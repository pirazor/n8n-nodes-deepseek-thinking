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
 * intact and is unique per call. Assistant turns without tool calls fall back to
 * their content, which is what the API keys off anyway.
 *
 * The store is process-wide rather than per instance. The instance that makes
 * a request is not always the one that saw the responses being replayed: n8n
 * can hand the agent a fresh model instance (a fallback model, a re-supplied
 * sub-node, history loaded from memory), and a per-instance cache then has
 * nothing to restore. Tool call ids are random and unique, so sharing the
 * store across instances cannot mix conversations up.
 */

type LooseMessage = {
	role?: string;
	content?: unknown;
	reasoning_content?: string;
	/** Some OpenAI-compatible servers use this name instead. */
	reasoning?: string;
	tool_calls?: Array<{ id?: string }>;
};

type LooseDelta = LooseMessage & {
	tool_calls?: Array<{ id?: string; index?: number }>;
};

type LooseChunk = {
	choices?: Array<{ delta?: LooseDelta }>;
};

// Every tool-calling turn of the current round is replayed on every request of
// that round and each of them must carry its reasoning, so the store has to
// outlast a whole agent run, and with several agents in flight, several. Each
// turn stores one key per tool call plus one for its content. Oldest entries
// go first once the bound is hit.
const MAX_REMEMBERED = 4096;
const reasoningByKey = new Map<string, string>();

function reasoningOf(message: LooseMessage): string | undefined {
	const value = message.reasoning_content ?? message.reasoning;
	return typeof value === 'string' && value !== '' ? value : undefined;
}

export class ChatDeepSeekThinking extends ChatOpenAI {
	/** Responses this instance has seen, and how many carried reasoning. */
	private seenResponses = 0;
	private seenWithReasoning = 0;

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
		this.seenResponses += 1;
		const reasoning = reasoningOf(message);
		if (!reasoning) return;
		this.seenWithReasoning += 1;

		for (const key of ChatDeepSeekThinking.keysFor(message)) {
			// Re-insert so a refreshed key moves to the young end of the map.
			reasoningByKey.delete(key);
			reasoningByKey.set(key, reasoning);
		}

		while (reasoningByKey.size > MAX_REMEMBERED) {
			const oldest = reasoningByKey.keys().next();
			if (oldest.done) break;
			reasoningByKey.delete(oldest.value);
		}
	}

	private restore(messages: LooseMessage[]): void {
		for (const message of messages) {
			// Only assistant turns carry it, and never overwrite one already there.
			if (message?.role !== 'assistant' || message.reasoning_content) continue;

			for (const key of ChatDeepSeekThinking.keysFor(message)) {
				const reasoning = reasoningByKey.get(key);
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
						const piece = delta.reasoning_content ?? delta.reasoning;
						if (typeof piece === 'string') {
							assembled.reasoning_content += piece;
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
	 * assistant messages had nothing to restore and where they sit, and what
	 * this instance and the shared store have seen, so a report of the error
	 * carries enough to tell an old install, a fresh instance, or a response
	 * without the field from an API rule this class does not know about.
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
			`reasoning_content: ${missing.join('; ') || 'none'}; this instance saw ` +
			`${this.seenResponses} response(s), ${this.seenWithReasoning} with reasoning; ` +
			`${reasoningByKey.size} key(s) remembered process-wide]`;
		return error;
	}

	// The base method is overloaded on streaming; widen to cover both and let
	// the call sites keep their own narrowing.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	async completionWithRetry(request: any, options?: any): Promise<any> {
		if (Array.isArray(request?.messages)) {
			this.restore(request.messages as LooseMessage[]);
		}

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
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
