/*
 * Regression tests for the tool-calling loop.
 *
 * DeepSeek requires `reasoning_content` to be handed back on every request that
 * carries `tools`. LangChain drops it, so the second leg of an agent loop fails
 * with a 400. This asserts:
 *
 *   1. plain ChatOpenAI still reproduces the failure, so the test is testing
 *      something real rather than passing vacuously
 *   2. our subclass carries the field back and the loop completes
 *   3. the same holds when the first leg is streamed, which is what n8n's Agent
 *      does whenever the workflow is triggered with a streaming response
 *   4. a long loop with many tool calls does not lose the reasoning of its
 *      earliest turns, which the API still demands for the whole round
 *
 * The DeepSeek API is faked at the transport layer, so no network or key.
 */

const assert = require('node:assert');
const { ChatOpenAI } = require('@langchain/openai');
const { AIMessage, HumanMessage, ToolMessage } = require('@langchain/core/messages');
const {
	ChatDeepSeekThinking,
	clearRememberedReasoning,
} = require('../dist/nodes/LmChatDeepSeekThinking/ChatDeepSeekThinking.js');

const REASONING = 'THOUGHT-ABC';
const TOOL_CALL_ID = 'call_42';

function firstLegMessage(callId, reasoning) {
	return {
		role: 'assistant',
		content: '',
		reasoning_content: reasoning,
		tool_calls: [{ id: callId, type: 'function', function: { name: 'lookup', arguments: '{}' } }],
	};
}

/** What the real API does when the field is missing on any tool-call turn. */
function assertReasoningPresent(request, expected) {
	for (const m of request.messages) {
		if (m.role !== 'assistant' || !m.tool_calls) continue;
		const want = expected(m);
		if (m.reasoning_content !== want) {
			throw new Error(
				'400 The `reasoning_content` in the thinking mode must be passed back to the API.',
			);
		}
	}
}

/** Streamed shape of the first leg: reasoning, then the tool call, in deltas. */
async function* streamedFirstLeg(callId, reasoning) {
	yield { choices: [{ index: 0, delta: { role: 'assistant', content: '', reasoning_content: reasoning.slice(0, 3) } }] };
	yield { choices: [{ index: 0, delta: { reasoning_content: reasoning.slice(3) } }] };
	yield {
		choices: [
			{
				index: 0,
				delta: {
					tool_calls: [
						{ index: 0, id: callId, type: 'function', function: { name: 'lookup', arguments: '' } },
					],
				},
			},
		],
	};
	yield { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{}' } }] }, finish_reason: 'tool_calls' }] };
}

function installFakeApi(captured) {
	let leg = 0;
	ChatOpenAI.prototype.completionWithRetry = async function (request) {
		leg += 1;
		captured.push(JSON.parse(JSON.stringify(request.messages)));

		if (leg === 1) {
			if (request.stream) return streamedFirstLeg(TOOL_CALL_ID, REASONING);
			return {
				id: '1',
				model: 'deepseek-v4-flash',
				usage: {},
				choices: [{ index: 0, finish_reason: 'tool_calls', message: firstLegMessage(TOOL_CALL_ID, REASONING) }],
			};
		}

		assertReasoningPresent(request, () => REASONING);

		return {
			id: '2',
			model: 'deepseek-v4-flash',
			usage: {},
			choices: [
				{
					index: 0,
					finish_reason: 'stop',
					message: { role: 'assistant', content: 'done', reasoning_content: 'THOUGHT-XYZ' },
				},
			],
		};
	};
}

function newModel(Cls) {
	return new Cls({
		apiKey: 'k',
		model: 'deepseek-v4-flash',
		maxRetries: 0,
		configuration: { baseURL: 'https://api.deepseek.com' },
	});
}

async function twoLegLoop(Cls, { streamFirstLeg = false } = {}) {
	const captured = [];
	installFakeApi(captured);
	const model = newModel(Cls);

	if (streamFirstLeg) {
		// streamEvents() in the agent ends up here: the model is streamed even
		// though the caller wants the whole message.
		const chunks = [];
		for await (const chunk of await model.stream([new HumanMessage('hi')])) chunks.push(chunk);
		assert.ok(chunks.length > 0, 'stream yielded nothing');
		const ids = chunks.flatMap((c) => c.tool_call_chunks ?? []).map((c) => c.id).filter(Boolean);
		assert.deepStrictEqual(ids, [TOOL_CALL_ID], 'stream did not pass tool call chunks through');
	} else {
		await model.invoke([new HumanMessage('hi')]);
	}

	// Exactly what an agent replays after running the tool.
	await model.invoke([
		new HumanMessage('hi'),
		new AIMessage({ content: '', tool_calls: [{ id: TOOL_CALL_ID, name: 'lookup', args: {} }] }),
		new ToolMessage({ tool_call_id: TOOL_CALL_ID, content: 'result' }),
	]);

	return captured[1].find((m) => m.role === 'assistant');
}

/** N tool-calling turns in one round, each with `parallel` tool calls. */
async function longLoop(turns, parallel) {
	let leg = 0;
	ChatOpenAI.prototype.completionWithRetry = async function (request) {
		leg += 1;
		assertReasoningPresent(request, (m) => `R-${m.tool_calls[0].id.split('_')[1]}`);
		if (leg > turns) {
			return { id: 'f', model: 'm', usage: {}, choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'done' } }] };
		}
		return {
			id: String(leg),
			model: 'm',
			usage: {},
			choices: [
				{
					index: 0,
					finish_reason: 'tool_calls',
					message: {
						role: 'assistant',
						content: '',
						reasoning_content: `R-${leg}`,
						tool_calls: Array.from({ length: parallel }, (_, i) => ({
							id: `call_${leg}_${i}`,
							type: 'function',
							function: { name: 'lookup', arguments: '{}' },
						})),
					},
				},
			],
		};
	};

	const model = newModel(ChatDeepSeekThinking);
	const history = [new HumanMessage('hi')];
	for (let t = 0; t <= turns; t += 1) {
		const reply = await model.invoke(history);
		if (!reply.tool_calls?.length) return t;
		history.push(new AIMessage({ content: '', tool_calls: reply.tool_calls }));
		for (const call of reply.tool_calls) {
			history.push(new ToolMessage({ tool_call_id: call.id, content: 'result' }));
		}
	}
	throw new Error('loop never finished');
}

(async () => {
	let controlError = null;
	try {
		await twoLegLoop(ChatOpenAI);
	} catch (error) {
		controlError = error.message;
	}
	assert.ok(
		controlError && controlError.includes('reasoning_content'),
		`expected plain ChatOpenAI to fail the loop, got: ${controlError ?? 'no error'}`,
	);
	console.log('OK  plain ChatOpenAI reproduces the 400');

	let assistant = await twoLegLoop(ChatDeepSeekThinking);
	assert.strictEqual(assistant.reasoning_content, REASONING, 'reasoning_content was not passed back on the second leg');
	assert.strictEqual(assistant.tool_calls?.[0]?.id, TOOL_CALL_ID, 'tool calls were not preserved');
	console.log('OK  subclass passes reasoning_content back and keeps tool calls');

	controlError = null;
	try {
		await twoLegLoop(ChatOpenAI, { streamFirstLeg: true });
	} catch (error) {
		controlError = error.message;
	}
	assert.ok(
		controlError && controlError.includes('reasoning_content'),
		`expected plain ChatOpenAI to fail the streamed loop, got: ${controlError ?? 'no error'}`,
	);
	console.log('OK  plain ChatOpenAI reproduces the 400 when the first leg is streamed');

	assistant = await twoLegLoop(ChatDeepSeekThinking, { streamFirstLeg: true });
	assert.strictEqual(assistant.reasoning_content, REASONING, 'reasoning_content collected from the stream was not passed back');
	console.log('OK  subclass collects reasoning_content from a streamed leg and passes it back');

	// A turn this instance never saw (e.g. an old install, or a rule the class
	// does not know) still fails, but the error now says what was missing.
	ChatOpenAI.prototype.completionWithRetry = async function (request) {
		assertReasoningPresent(request, () => 'never-seen');
		return { choices: [] };
	};
	let diagnostic = null;
	try {
		await newModel(ChatDeepSeekThinking).invoke([
			new HumanMessage('hi'),
			new AIMessage({ content: '', tool_calls: [{ id: 'call_unknown', name: 'lookup', args: {} }] }),
			new ToolMessage({ tool_call_id: 'call_unknown', content: 'result' }),
		]);
	} catch (error) {
		diagnostic = error.message;
	}
	assert.ok(
		diagnostic && diagnostic.includes('1 assistant message(s) without reasoning_content: #1 (current round, tool_calls call_unknown)'),
		`diagnostic missing from error: ${diagnostic}`,
	);
	assert.ok(diagnostic.includes('this instance saw 0 response(s)'), `diagnostic lacks instance counters: ${diagnostic}`);
	console.log('OK  an unrecoverable 400 names the assistant messages that had no reasoning to restore');

	// The instance replaying a turn is not always the one that saw the response
	// (n8n can supply a fresh model instance mid-run), so the store is shared.
	{
		const captured = [];
		installFakeApi(captured);
		const first = newModel(ChatDeepSeekThinking);
		await first.invoke([new HumanMessage('hi')]);
		const second = newModel(ChatDeepSeekThinking);
		await second.invoke([
			new HumanMessage('hi'),
			new AIMessage({ content: '', tool_calls: [{ id: TOOL_CALL_ID, name: 'lookup', args: {} }] }),
			new ToolMessage({ tool_call_id: TOOL_CALL_ID, content: 'result' }),
		]);
		assert.strictEqual(captured[1].find((m) => m.role === 'assistant').reasoning_content, REASONING);
		console.log('OK  a fresh model instance restores a turn another instance saw');
	}

	// One response with parallel tool calls can come back split into one
	// assistant message per call; each must get the reasoning of its turn.
	{
		let leg = 0;
		ChatOpenAI.prototype.completionWithRetry = async function (request) {
			leg += 1;
			if (leg === 1) {
				return {
					id: '1', model: 'm', usage: {},
					choices: [{ index: 0, finish_reason: 'tool_calls', message: {
						role: 'assistant', content: '', reasoning_content: 'R-split',
						tool_calls: ['call_00_a', 'call_01_b'].map((id) => ({ id, type: 'function', function: { name: 'lookup', arguments: '{}' } })),
					} }],
				};
			}
			assertReasoningPresent(request, () => 'R-split');
			return { id: '2', model: 'm', usage: {}, choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'done' } }] };
		};
		const model = newModel(ChatDeepSeekThinking);
		await model.invoke([new HumanMessage('hi')]);
		await newModel(ChatDeepSeekThinking).invoke([
			new HumanMessage('hi'),
			new AIMessage({ content: '', tool_calls: [{ id: 'call_00_a', name: 'lookup', args: {} }] }),
			new ToolMessage({ tool_call_id: 'call_00_a', content: 'a' }),
			new AIMessage({ content: '', tool_calls: [{ id: 'call_01_b', name: 'lookup', args: {} }] }),
			new ToolMessage({ tool_call_id: 'call_01_b', content: 'b' }),
		]);
		console.log('OK  parallel tool calls replayed as separate assistant messages each get their reasoning');
	}

	// Servers that name the field `reasoning` are captured too.
	{
		let leg = 0;
		ChatOpenAI.prototype.completionWithRetry = async function (request) {
			leg += 1;
			if (leg === 1) {
				return { id: '1', model: 'm', usage: {}, choices: [{ index: 0, finish_reason: 'tool_calls', message: { ...firstLegMessage('call_r', 'R-alt'), reasoning_content: undefined, reasoning: 'R-alt' } }] };
			}
			assertReasoningPresent(request, () => 'R-alt');
			return { id: '2', model: 'm', usage: {}, choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'done' } }] };
		};
		const model = newModel(ChatDeepSeekThinking);
		await model.invoke([new HumanMessage('hi')]);
		await model.invoke([
			new HumanMessage('hi'),
			new AIMessage({ content: '', tool_calls: [{ id: 'call_r', name: 'lookup', args: {} }] }),
			new ToolMessage({ tool_call_id: 'call_r', content: 'r' }),
		]);
		console.log('OK  a response that names the field `reasoning` is passed back as reasoning_content');
	}

	// n8n's Agent v3 hands tool execution to the workflow engine and re-runs the
	// agent node for each round: a fresh model instance, in what may be a fresh
	// process, and assistant messages rebuilt from stored metadata that includes
	// additional_kwargs.reasoning_content. So the returned message must carry
	// the reasoning, and a message carrying it must be enough on its own.
	for (const streamFirstLeg of [false, true]) {
		const captured = [];
		installFakeApi(captured);
		const first = newModel(ChatDeepSeekThinking);
		let reply;
		if (streamFirstLeg) {
			for await (const chunk of await first.stream([new HumanMessage('hi')])) {
				reply = reply ? reply.concat(chunk) : chunk;
			}
		} else {
			reply = await first.invoke([new HumanMessage('hi')]);
		}
		assert.strictEqual(reply.additional_kwargs.reasoning_content, REASONING, `returned message lacks reasoning (stream=${streamFirstLeg})`);
		assert.strictEqual(reply.tool_calls?.[0]?.id, TOOL_CALL_ID);

		// What the engine rebuilds, in a process that never saw the response.
		clearRememberedReasoning();
		const rebuilt = new AIMessage({
			content: '',
			tool_calls: [{ id: TOOL_CALL_ID, name: 'lookup', args: {} }],
			additional_kwargs: { reasoning_content: reply.additional_kwargs.reasoning_content },
		});
		await newModel(ChatDeepSeekThinking).invoke([
			new HumanMessage('hi'),
			rebuilt,
			new ToolMessage({ tool_call_id: TOOL_CALL_ID, content: 'result' }),
		]);
		assert.strictEqual(captured[1].find((m) => m.role === 'assistant').reasoning_content, REASONING);
		console.log(`OK  reasoning travels inside the message across a fresh process (first leg streamed=${streamFirstLeg})`);
	}

	const turns = await longLoop(40, 5);
	assert.strictEqual(turns, 40);
	console.log('OK  a 40-turn loop with 5 parallel tool calls per turn keeps every turn\'s reasoning');
})().catch((error) => {
	console.error('FAILED:', error.message);
	process.exit(1);
});
