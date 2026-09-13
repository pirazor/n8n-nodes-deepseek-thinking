import { ChatOpenAI } from '@langchain/openai';

/**
 * ChatOpenAI with DeepSeek's `reasoning_content` carried across turns.
 *
 * In thinking mode DeepSeek returns a `reasoning_content` field alongside the
 * assistant message, and for any request that carries `tools` it requires that
 * field to be handed back on every following request in the same round. LangChain
 * does not know about it: `openAIResponseToChatMessage` builds `additional_kwargs`
 * from a fixed literal of `function_call` and `tool_calls`, so `reasoning_content`
 * is dropped the moment the response is parsed and can never be sent back.
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
 * Both transports are covered. Unstreamed responses are read off `choices[0]`.
 * Streamed responses (n8n's Agent runs the model through `streamEvents` when
 * the workflow is triggered with a streaming response, and LangChain then uses
 * `stream: true` even for the tool-calling legs) are wrapped so the
 * `reasoning_content` and tool call ids can be collected from the deltas as
 * they pass through, without changing what the caller sees.
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

type LooseDelta = LooseMessage & {
	tool_calls?: Array<{ id?: string; index?: number }>;
};

type LooseChunk = {
	choices?: Array<{ delta?: LooseDelta }>;
};

// Every tool-calling turn of the current round is replayed on every request of
// that round and each of them must carry its reasoning, so the cache has to
// outlast a whole agent run. It is per model instance (one per node execution)
// and is released with it, so the bound only guards against a runaway loop.
// Each turn stores one key per tool call plus one for its content.
const MAX_REMEMBERED = 1024;

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

	/**
	 * Pass a streamed response through untouched while assembling the assistant
	 * message from its deltas, then remember it once the stream is exhausted.
	 * The wrapper keeps the original stream's properties (`controller` for
	 * aborts) so it is a drop-in for what the OpenAI client returned.
	 */
	private captureStream<T extends AsyncIterable<LooseChunk>>(stream: T): T {
		const assembled: LooseMessage = { role: 'assistant', content: '', reasoning_content: '' };
		const toolCallIds: string[] = [];
		const self = this;

		async function* iterate(): AsyncGenerator<LooseChunk> {
			try {
				for await (const chunk of stream) {
					const delta = chunk?.choices?.[0]?.delta;
					if (delta) {
						if (typeof delta.reasoning_content === 'string') {
							assembled.reasoning_content += delta.reasoning_content;
						}
						if (typeof delta.content === 'string') {
							assembled.content += delta.content;
						}
						for (const call of delta.tool_calls ?? []) {
							if (call?.id) toolCallIds.push(call.id);
						}
					}
					yield chunk;
				}
			} finally {
				// Remember whatever arrived, even if the consumer stopped early:
				// the ids and reasoning stream first, and a partial record is
				// still better than a guaranteed 400 on the next leg.
				assembled.tool_calls = toolCallIds.map((id) => ({ id }));
				self.remember(assembled);
			}
		}

		const wrapped = iterate();
		// Carry over anything else the caller might reach for on the stream.
		for (const key of Object.keys(stream)) {
			if (!(key in wrapped)) {
				Object.defineProperty(wrapped, key, {
					get: () => (stream as unknown as Record<string, unknown>)[key],
					enumerable: true,
				});
			}
		}
		return wrapped as unknown as T;
	}

	/**
	 * When the API still rejects a request over `reasoning_content`, say which
	 * assistant messages had nothing to restore and where they sit, so a report
	 * of the error carries enough to tell an old install from an API rule this
	 * class does not know about.
	 */
	private explainReasoningError(error: unknown, request: { messages?: LooseMessage[] }): unknown {
		const message = (error as { message?: unknown })?.message;
		if (typeof message !== 'string' || !message.includes('reasoning_content')) return error;
		if (!Array.isArray(request?.messages)) return error;

		const messages = request.messages;
		let lastUser = -1;
		messages.forEach((m, i) => {
			if (m?.role === 'user') lastUser = i;
		});

		const missing = messages
			.map((m, i) => ({ m, i }))
			.filter(({ m }) => m?.role === 'assistant' && !m.reasoning_content)
			.map(({ m, i }) => {
				const ids = (m.tool_calls ?? []).map((c) => c?.id ?? '?').join(',');
				const where = i > lastUser ? 'current round' : 'earlier round';
				return `#${i} (${where}${ids ? `, tool_calls ${ids}` : ', no tool_calls'})`;
			});

		(error as { message: string }).message =
			`${message} [n8n-nodes-deepseek-thinking: ${missing.length} assistant message(s) without ` +
			`reasoning_content: ${missing.join('; ') || 'none'}; ${this.reasoningByKey.size} remembered]`;
		return error;
	}

	// The base method is overloaded on streaming; widen to cover both and let
	// the call sites keep their own narrowing.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	async completionWithRetry(request: any, options?: any): Promise<any> {
		if (Array.isArray(request?.messages)) {
			this.restore(request.messages as LooseMessage[]);
		}

		let response: any;
		try {
			response = await super.completionWithRetry(request, options);
		} catch (error) {
			throw this.explainReasoningError(error, request);
		}

		if (request?.stream) {
			if (response && typeof response[Symbol.asyncIterator] === 'function') {
				return this.captureStream(response);
			}
			return response;
		}

		const message = (response as { choices?: Array<{ message?: LooseMessage }> })
			?.choices?.[0]?.message;
		if (message) this.remember(message);

		return response;
	}
}
