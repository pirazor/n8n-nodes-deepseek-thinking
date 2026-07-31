import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { Serialized } from '@langchain/core/load/serializable';
import type { LLMResult } from '@langchain/core/outputs';
import {
	NodeConnectionTypes,
	NodeOperationError,
	type ISupplyDataFunctions,
} from 'n8n-workflow';

/**
 * Reports model runs back to n8n.
 *
 * n8n's built-in LLM sub-nodes attach an internal `N8nLlmTracing` callback.
 * That callback is what makes the sub-node show its input and output in the
 * UI, display the running indicator, and report token usage. It is not
 * exported to community packages, so this is a minimal equivalent built on the
 * public `addInputData` / `addOutputData` methods of ISupplyDataFunctions.
 *
 * Without it the model still answers, but n8n sees nothing: the sub-node shows
 * no output and the agent behaves as if the model never produced a result.
 */
export class N8nTracing extends BaseCallbackHandler {
	name = 'N8nDeepSeekTracing';

	// Ensures n8n has recorded the data before the run is torn down.
	awaitHandlers = true;

	private runIndexByRunId = new Map<string, number>();

	constructor(private readonly ctx: ISupplyDataFunctions) {
		super();
	}

	handleLLMStart(_llm: Serialized, prompts: string[], runId: string): void {
		const { index } = this.ctx.addInputData(NodeConnectionTypes.AiLanguageModel, [
			[{ json: { messages: prompts } }],
		]);
		this.runIndexByRunId.set(runId, index);
	}

	handleLLMEnd(output: LLMResult, runId: string): void {
		const runIndex = this.runIndexByRunId.get(runId) ?? 0;
		this.runIndexByRunId.delete(runId);

		// Surface reasoning tokens too, which is the point of this node.
		const usage = (output.llmOutput?.tokenUsage ?? output.llmOutput?.estimatedTokenUsage) as
			| Record<string, number>
			| undefined;

		const generations = output.generations?.map((generation) =>
			generation.map((item) => ({
				text: item.text,
				generationInfo: item.generationInfo,
			})),
		);

		this.ctx.addOutputData(NodeConnectionTypes.AiLanguageModel, runIndex, [
			[
				{
					json: {
						response: { generations },
						tokenUsage: usage,
					},
				},
			],
		]);
	}

	handleLLMError(error: Error, runId: string): void {
		const runIndex = this.runIndexByRunId.get(runId) ?? 0;
		this.runIndexByRunId.delete(runId);
		// addOutputData expects an n8n error type, not a bare Error.
		this.ctx.addOutputData(
			NodeConnectionTypes.AiLanguageModel,
			runIndex,
			new NodeOperationError(this.ctx.getNode(), error),
		);
	}
}
