/*
 * Regression test for the tool-calling loop.
 *
 * DeepSeek requires `reasoning_content` to be handed back on every request that
 * carries `tools`. LangChain drops it, so the second leg of an agent loop fails
 * with a 400. This asserts two things:
 *
 *   1. plain ChatOpenAI still reproduces the failure, so the test is testing
 *      something real rather than passing vacuously
 *   2. our subclass carries the field back and the loop completes
 *
 * The DeepSeek API is faked at the transport layer, so no network or key.
 */

const assert = require('node:assert');
const { ChatOpenAI } = require('@langchain/openai');
const { AIMessage, HumanMessage, ToolMessage } = require('@langchain/core/messages');
const {
	ChatDeepSeekThinking,
} = require('../dist/nodes/LmChatDeepSeekThinking/ChatDeepSeekThinking.js');

const REASONING = 'THOUGHT-ABC';
const TOOL_CALL_ID = 'call_42';

function installFakeApi(captured) {
	let leg = 0;
	ChatOpenAI.prototype.completionWithRetry = async function (request) {
		leg += 1;
		captured.push(JSON.parse(JSON.stringify(request.messages)));

		if (leg === 1) {
			return {
				id: '1',
				model: 'deepseek-v4-flash',
				usage: {},
				choices: [
					{
						index: 0,
						finish_reason: 'tool_calls',
						message: {
							role: 'assistant',
							content: '',
							reasoning_content: REASONING,
							tool_calls: [
								{
									id: TOOL_CALL_ID,
									type: 'function',
									function: { name: 'lookup', arguments: '{}' },
								},
							],
						},
					},
				],
			};
		}

		// What the real API does when the field is missing.
		const assistant = request.messages.find((m) => m.role === 'assistant');
		if (!assistant || assistant.reasoning_content !== REASONING) {
			throw new Error(
				'400 The `reasoning_content` in the thinking mode must be passed back to the API.',
			);
		}

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

async function twoLegLoop(Cls) {
	const captured = [];
	installFakeApi(captured);

	const model = new Cls({
		apiKey: 'k',
		model: 'deepseek-v4-flash',
		maxRetries: 0,
		configuration: { baseURL: 'https://api.deepseek.com' },
	});

	await model.invoke([new HumanMessage('hi')]);

	// Exactly what an agent replays after running the tool.
	await model.invoke([
		new HumanMessage('hi'),
		new AIMessage({ content: '', tool_calls: [{ id: TOOL_CALL_ID, name: 'lookup', args: {} }] }),
		new ToolMessage({ tool_call_id: TOOL_CALL_ID, content: 'result' }),
	]);

	return captured[1].find((m) => m.role === 'assistant');
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

	const assistant = await twoLegLoop(ChatDeepSeekThinking);
	assert.strictEqual(
		assistant.reasoning_content,
		REASONING,
		'reasoning_content was not passed back on the second leg',
	);
	assert.strictEqual(
		assistant.tool_calls?.[0]?.id,
		TOOL_CALL_ID,
		'tool calls were not preserved',
	);
	console.log('OK  subclass passes reasoning_content back and keeps tool calls');
})().catch((error) => {
	console.error('FAILED:', error.message);
	process.exit(1);
});
