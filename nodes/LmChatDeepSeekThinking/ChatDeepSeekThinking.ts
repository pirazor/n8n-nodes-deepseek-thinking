import { AIMessageChunk, type BaseMessage } from '@langchain/core/messages';
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs';
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
 * Two mechanisms carry it, and both are needed.
 *
 * 1. In the message. The reasoning is attached to the returned AIMessage as
 *    `additional_kwargs.reasoning_content` (for a stream, as a final empty chunk
 *    that LangChain merges into the aggregate), and any incoming assistant
 *    message that carries that field has it put back on the request. This is
 *    what n8n's Agent v3 relies on: it hands tool execution to the workflow
 *    engine and re-runs the agent node for each round with a freshly supplied
 *    model instance, rebuilding the assistant messages from stored metadata,
 *    including `additional_kwargs.reasoning_content`. Nothing has to survive in
 *    memory between calls.
 *
 * 2. In a process-wide store keyed by tool call id, for agents that replay
 *    LangChain's own message objects (Agent v2's AgentExecutor, fallback
 *    models). `completionWithRetry` is the one class method that sees the fully
 *    built request and the raw response, so it remembers `reasoning_content` as
 *    it arrives and restores it on matching assistant messages on the way out.
 *    Tool call ids are random and unique, so sharing the store across instances
 *    cannot mix conversations up.
 *
 * Both transports are covered. Unstreamed responses are read off `choices[0]`.
 * Streamed responses (n8n's Agent runs the model through `streamEvents` when
 * the workflow is triggered with a streaming response, and LangChain then uses
 * `stream: true` even for the tool-calling legs) are wrapped so the
 * `reasoning_content` and tool call ids can be collected from the deltas as
 * they pass through, without changing what the caller sees.
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

/** Test hook: forget everything, as a fresh n8n process would have. */
export function clearRememberedReasoning(): void {
	reasoningByKey.clear();
}

function reasoningOf(message: LooseMessage): string | undefined {
	const value = message.reasoning_content ?? message.reasoning;
	return typeof value === 'string' && value !== '' ? value : undefined;
}

/** Identifiers a turn can be recognised by on the next request. */
function keysFor(message: LooseMessage): string[] {
	const keys: string[] = [];

	for (const call of message.tool_calls ?? []) {
		if (call?.id) keys.push(`tool:${call.id}`);
	}

	if (typeof message.content === 'string' && message.content.trim() !== '') {
		keys.push(`content:${message.content.slice(0, 512)}`);
	}

	return keys;
}

function store(message: LooseMessage, reasoning: string): void {
	for (const key of keysFor(message)) {
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

function lookup(message: LooseMessage): string | undefined {
	for (const key of keysFor(message)) {
		const reasoning = reasoningByKey.get(key);
		if (reasoning) return reasoning;
	}
	return undefined;
}

/** The parts of a LangChain message this class reads; typed loosely on purpose. */
function asLoose(message: BaseMessage): LooseMessage {
	const m = message as unknown as {
		content?: unknown;
		tool_calls?: Array<{ id?: string }>;
		additional_kwargs?: Record<string, unknown>;
	};
	const kwargs = m.additional_kwargs ?? {};
	return {
		role: message._getType() === 'ai' ? 'assistant' : message._getType(),
		content: m.content,
		tool_calls: m.tool_calls ?? (kwargs.tool_calls as Array<{ id?: string }> | undefined),
		reasoning_content: kwargs.reasoning_content as string | undefined,
		reasoning: kwargs.reasoning as string | undefined,
	};
}

export class ChatDeepSeekThinking extends ChatOpenAI {
	/** Responses this instance has seen, and how many carried reasoning. */
	private seenResponses = 0;
	private seenWithReasoning = 0;

	private remember(message: LooseMessage): void {
		this.seenResponses += 1;
		const reasoning = reasoningOf(message);
		if (!reasoning) return;
		this.seenWithReasoning += 1;
		store(message, reasoning);
	}

	private restore(messages: LooseMessage[]): void {
		for (const message of messages) {
			// Only assistant turns carry it, and never overwrite one already there.
			if (message?.role !== 'assistant' || message.reasoning_content) continue;
			const reasoning = lookup(message);
			if (reasoning) message.reasoning_content = reasoning;
		}
	}

	/**
	 * Incoming assistant messages that already carry the reasoning (rebuilt by
	 * n8n's engine, or the very objects this class annotated earlier) feed the
	 * store, so `restore` finds them once the messages have been converted.
	 */
	private primeFrom(messages: BaseMessage[]): void {
		for (const message of messages) {
			if (message._getType() !== 'ai') continue;
			const loose = asLoose(message);
			const reasoning = reasoningOf(loose);
			if (reasoning) store(loose, reasoning);
		}
	}

	/** Put the reasoning of a just-produced message where callers can see it. */
	private static annotate(message: BaseMessage): void {
		const loose = asLoose(message);
		if (loose.reasoning_content) return;
		const reasoning = lookup(loose);
		if (!reasoning) return;
		const kwargs = (message as unknown as { additional_kwargs?: Record<string, unknown> })
			.additional_kwargs;
		if (kwargs) kwargs.reasoning_content = reasoning;
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

	// The two entry points that see LangChain messages: prime the store from
	// what comes in, and annotate what goes out.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	async _generate(messages: BaseMessage[], options: any, runManager?: any): Promise<ChatResult> {
		this.primeFrom(messages);
		const result = await super._generate(messages, options, runManager);
		for (const generation of result.generations ?? []) {
			if (generation?.message) ChatDeepSeekThinking.annotate(generation.message);
		}
		return result;
	}

	async *_streamResponseChunks(
		messages: BaseMessage[],
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		options: any,
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		runManager?: any,
	): AsyncGenerator<ChatGenerationChunk> {
		this.primeFrom(messages);

		// Track what the aggregate will be keyed by, to find its reasoning.
		const seen: LooseMessage = { role: 'assistant', content: '', tool_calls: [] };
		for await (const chunk of super._streamResponseChunks(messages, options, runManager)) {
			const message = chunk.message as AIMessageChunk;
			if (typeof message.content === 'string') seen.content += message.content;
			for (const call of message.tool_call_chunks ?? []) {
				if (call?.id) seen.tool_calls!.push({ id: call.id });
			}
			yield chunk;
		}

		// The raw stream has been drained by now, so the reasoning is stored.
		// A trailing empty chunk carries it; LangChain merges additional_kwargs
		// when it concatenates chunks, so the aggregate ends up annotated.
		const reasoning = lookup(seen);
		if (reasoning) {
			yield new ChatGenerationChunk({
				text: '',
				message: new AIMessageChunk({
					content: '',
					additional_kwargs: { reasoning_content: reasoning },
				}),
			});
		}
	}
}
