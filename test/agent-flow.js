/*
 * Reproduce n8n's Tools Agent flow end to end against a fake DeepSeek:
 * AgentExecutor + createToolCallingAgent, run through streamEvents (what n8n
 * does with a streaming trigger) and through invoke, with parallel tool calls
 * on turn 1 and a single tool call on turn 2, then a final answer.
 * Every request that carries an assistant tool-call message must carry the
 * matching reasoning_content or the fake API throws the real 400.
 */
const assert = require('node:assert');
const { ChatOpenAI } = require('@langchain/openai');
const { AgentExecutor, createToolCallingAgent } = require('langchain/agents');
const { ChatPromptTemplate } = require('@langchain/core/prompts');
const { DynamicStructuredTool } = require('@langchain/core/tools');
const { z } = require('zod');
const { ChatDeepSeekThinking } = require('../dist/nodes/LmChatDeepSeekThinking/ChatDeepSeekThinking.js');

// Some OpenAI-compatible servers repeat the tool call id on every delta.
let REPEAT_ID_EVERY_DELTA = false;

const turns = [
  { reasoning: 'R1-think-about-both', calls: [['lookup', { q: 'a' }], ['lookup', { q: 'b' }]] },
  { reasoning: 'R2-one-more', calls: [['lookup', { q: 'c' }]] },
  { reasoning: 'R3-final', final: 'done' },
];

function toolCall(turn, i, name, args) {
  return { id: `call_${turn}_${i}`, type: 'function', function: { name, arguments: JSON.stringify(args) } };
}

async function* streamed(turn, t) {
  const chunks = [];
  chunks.push({ choices: [{ index: 0, delta: { role: 'assistant', content: '', reasoning_content: t.reasoning.slice(0, 4) } }] });
  chunks.push({ choices: [{ index: 0, delta: { reasoning_content: t.reasoning.slice(4) } }] });
  if (t.final) {
    chunks.push({ choices: [{ index: 0, delta: { content: t.final } }] });
    chunks.push({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
  } else {
    t.calls.forEach(([name, args], i) => {
      const full = toolCall(turn, i, name, args);
      chunks.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: full.id, type: 'function', function: { name, arguments: '' } }] } }] });
      const argStr = full.function.arguments;
      chunks.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, ...(REPEAT_ID_EVERY_DELTA ? { id: full.id } : {}), function: { arguments: argStr.slice(0, 3) } }] } }] });
      chunks.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, ...(REPEAT_ID_EVERY_DELTA ? { id: full.id } : {}), function: { arguments: argStr.slice(3) } }] } }] });
    });
    chunks.push({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] });
  }
  chunks.push({ choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } });
  for (const c of chunks) yield c;
}

function unstreamed(turn, t) {
  const message = t.final
    ? { role: 'assistant', content: t.final, reasoning_content: t.reasoning }
    : { role: 'assistant', content: '', reasoning_content: t.reasoning, tool_calls: t.calls.map(([n, a], i) => toolCall(turn, i, n, a)) };
  return { id: String(turn), model: 'm', usage: {}, choices: [{ index: 0, finish_reason: t.final ? 'stop' : 'tool_calls', message }] };
}

function installFakeApi(log) {
  let turn = 0;
  ChatOpenAI.prototype.completionWithRetry = async function (request) {
    const msgs = JSON.parse(JSON.stringify(request.messages));
    log.push({ stream: !!request.stream, messages: msgs });
    for (const m of msgs) {
      if (m.role === 'assistant' && m.tool_calls?.length) {
        const wantTurn = Number(m.tool_calls[0].id.split('_')[1]);
        const want = turns[wantTurn]?.reasoning;
        if (m.reasoning_content !== want) {
          throw new Error(`400 The \`reasoning_content\` in the thinking mode must be passed back to the API. (turn ${wantTurn}: got ${JSON.stringify(m.reasoning_content)}, ids ${m.tool_calls.map((c) => c.id).join(',')})`);
        }
      }
    }
    const t = turns[turn];
    const idx = turn;
    turn += 1;
    return request.stream ? streamed(idx, t) : unstreamed(idx, t);
  };
}

async function run(Cls, { streaming }) {
  const log = [];
  installFakeApi(log);
  const model = new Cls({ apiKey: 'k', model: 'deepseek-v4-flash', maxRetries: 0, configuration: { baseURL: 'https://api.deepseek.com' } });
  const tools = [
    new DynamicStructuredTool({ name: 'lookup', description: 'look something up', schema: z.object({ q: z.string() }), func: async ({ q }) => `result for ${q}` }),
  ];
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are helpful'],
    ['placeholder', '{chat_history}'],
    ['human', '{input}'],
    ['placeholder', '{agent_scratchpad}'],
  ]);
  const agent = createToolCallingAgent({ llm: model, tools, prompt, streamRunnable: false });
  const executor = AgentExecutor.fromAgentAndTools({ agent, tools, returnIntermediateSteps: true });
  const input = { input: 'hi', chat_history: [] };

  let output;
  if (streaming) {
    const events = executor.streamEvents(input, { version: 'v2' });
    for await (const ev of events) {
      if (process.env.DEBUG) console.log('EV', ev.event, ev.name, ev.event === 'on_chain_end' ? JSON.stringify(ev.data?.output)?.slice(0, 300) : '');
      // The executor is the top-level run; its end event carries the result.
      if (ev.event === 'on_chain_end' && ev.name === 'AgentExecutor') output = ev.data.output;
    }
  } else {
    output = await executor.invoke(input);
  }
  return { output, log };
}

(async () => {
  for (const [streaming, repeatId] of [[false, false], [true, false], [true, true]]) {
    REPEAT_ID_EVERY_DELTA = repeatId;
    let err = null;
    try { await run(ChatOpenAI, { streaming }); } catch (e) { err = e.message; }
    assert.ok(err && err.includes('reasoning_content'), `plain ChatOpenAI should fail (streaming=${streaming}), got ${err}`);
    console.log(`OK  plain ChatOpenAI reproduces the 400 through AgentExecutor (streaming=${streaming}${repeatId ? ', id on every delta' : ''})`);

    const { output, log } = await run(ChatDeepSeekThinking, { streaming });
    if (process.env.DEBUG) console.log('LEGS', JSON.stringify(log.map((l) => l.messages.map((m) => [m.role, m.tool_calls?.map((c) => c.id), m.reasoning_content]))));
    // invoke() returns { output }, streamEvents() ends with the bare output.
    const finalText = typeof output === 'string' ? output : output?.output;
    assert.strictEqual(finalText, 'done', `agent did not finish (streaming=${streaming}), ${log.length} legs`);
    assert.strictEqual(log.length, 3, `expected 3 legs, got ${log.length}`);
    assert.strictEqual(log[2].messages.filter((m) => m.role === 'assistant').length, 2);
    assert.ok(log.every((l) => l.stream === streaming), 'stream flag mismatch');
    console.log(`OK  subclass completes 2 parallel + 1 sequential tool calls through AgentExecutor (streaming=${streaming}${repeatId ? ', id on every delta' : ''})`);
  }
})().catch((e) => { console.error('FAILED:', e.message); process.exit(1); });
