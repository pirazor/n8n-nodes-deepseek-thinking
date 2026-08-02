import { ChatOpenAI } from '@langchain/openai';

/**
 * ChatOpenAI with DeepSeek's `reasoning_content` carried across turns.
 *
 * In thinking mode DeepSeek returns a `reasoning_content` field alongside the
 * assistant message, and for any request that carries `tools` it requires that
 * field to be handed back on every following request. LangChain does not know
 * about it: `openAIResponseToChatMessage` builds `additional_kwargs` from a
 * fixed literal of `function_call` and `tool_calls`, so `reasoning_content` is
 * dropped the moment the response is parsed and can never be sent back.
 *
 * The result is that a single-shot call works, but the second leg of an agent's
 * tool-calling loop fails with:
 *
 *   400 The `reasoning_content` in the thinking mode must be passed back to the API.
 *
 * `completionWithRetry` is the one class method that sees the fully built
 * request and the raw response, so both halves of the round trip are done here:
 * remember `reasoning_content` as it arrives, and put it back on the matching
 * assistant message on the way out.
 *
 * Messages are matched on tool call id, which survives LangChain's conversion
 * intact and is unique per turn. Assistant turns without tool calls fall back to
 * their content, which is what the API keys off anyway.
 */

type LooseMessage = {
	role?: string;
	content?: unknown;
	reasoning_content?: string;
	tool_calls?: Array<{ id?: string }>;
};

// A run of an agent is bounded, but the cache is per model instance and a model
// instance is per node execution, so keep it small rather than unbounded.
const MAX_REMEMBERED = 64;

export class ChatDeepSeekThinking extends ChatOpenAI {
	private readonly reasoningByKey = new Map<string, string>();

	/** Identifiers a turn can be recognised by on the next request. */
	private static keysFor(message: LooseMessage): string[] {
		const keys: string[] = [];

		for (const call of message.tool_calls ?? []) {
			if (call?.id) keys.push(`tool:${call.id}`);
		}

		if (typeof message.content === 'string' && message.content.trim() !== '') {
			keys.push(`content:${message.content.slice(0, 512)}`);
		}

		return keys;
	}

	private remember(message: LooseMessage): void {
		const reasoning = message.reasoning_content;
		if (typeof reasoning !== 'string' || reasoning === '') return;

		for (const key of ChatDeepSeekThinking.keysFor(message)) {
			this.reasoningByKey.set(key, reasoning);
		}

		while (this.reasoningByKey.size > MAX_REMEMBERED) {
			const oldest = this.reasoningByKey.keys().next();
			if (oldest.done) break;
			this.reasoningByKey.delete(oldest.value);
		}
	}

	private restore(messages: LooseMessage[]): void {
		for (const message of messages) {
			// Only assistant turns carry it, and never overwrite one already there.
			if (message?.role !== 'assistant' || message.reasoning_content) continue;

			for (const key of ChatDeepSeekThinking.keysFor(message)) {
				const reasoning = this.reasoningByKey.get(key);
				if (reasoning) {
					message.reasoning_content = reasoning;
					break;
				}
			}
		}
	}

	// The base method is overloaded on streaming; widen to cover both and let
	// the call sites keep their own narrowing.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	async completionWithRetry(request: any, options?: any): Promise<any> {
		if (Array.isArray(request?.messages)) {
			this.restore(request.messages as LooseMessage[]);
		}

		const response = await super.completionWithRetry(request, options);

		// A stream cannot be inspected without consuming it. Tool calling runs
		// unstreamed, which is the case that fails, so only capture there.
		if (!request?.stream) {
			const message = (response as { choices?: Array<{ message?: LooseMessage }> })
				?.choices?.[0]?.message;
			if (message) this.remember(message);
		}

		return response;
	}
}
