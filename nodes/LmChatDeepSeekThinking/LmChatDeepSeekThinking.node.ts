import { ChatOpenAI } from '@langchain/openai';
import {
	NodeConnectionTypes,
	type ILoadOptionsFunctions,
	type INodePropertyOptions,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

/**
 * DeepSeek chat model with an exposed thinking level.
 *
 * n8n's built-in DeepSeek node instantiates ChatOpenAI (DeepSeek's API is
 * OpenAI-compatible) but never surfaces `reasoning_effort`, so thinking depth
 * cannot be controlled. This node is the same wiring plus a Thinking Level
 * option that is forwarded as a top-level `reasoning_effort` field.
 *
 * DeepSeek semantics: any value other than `none` enables thinking mode.
 * `low`/`medium` are normalised to `high` and `xhigh` to `max` server-side;
 * they are offered here so the field matches other providers' vocabulary.
 */

interface DeepSeekOptions {
	frequencyPenalty?: number;
	maxTokens?: number;
	presencePenalty?: number;
	temperature?: number;
	topP?: number;
	timeout?: number;
	maxRetries?: number;
	responseFormat?: 'text' | 'json_object';
}

export class LmChatDeepSeekThinking implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'DeepSeek Chat Model (Thinking)',
		name: 'lmChatDeepSeekThinking',
		icon: 'file:deepseek.svg',
		group: ['transform'],
		version: 1,
		description: 'DeepSeek chat model with a configurable thinking level',
		defaults: { name: 'DeepSeek Chat Model (Thinking)' },
		codex: {
			categories: ['AI'],
			subcategories: { AI: ['Language Models', 'Root Nodes'] },
			resources: {
				primaryDocumentation: [
					{ url: 'https://api-docs.deepseek.com/guides/thinking_mode/' },
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],
		credentials: [{ name: 'deepSeekApi', required: true }],
		requestDefaults: {
			ignoreHttpStatusErrors: true,
			baseURL: '={{ $credentials?.url }}',
		},
		properties: [
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				typeOptions: { loadOptionsMethod: 'getModels' },
				default: 'deepseek-reasoner',
				description:
					'The DeepSeek model to use. Thinking is only produced by models that support it (e.g. deepseek-reasoner).',
			},
			{
				displayName: 'Thinking Level',
				name: 'thinkingLevel',
				type: 'options',
				default: 'default',
				description:
					'Sent as reasoning_effort. "API default" omits the field entirely; "None" disables thinking. DeepSeek maps low/medium to high and xhigh to max.',
				options: [
					{ name: 'API Default (Do Not Send)', value: 'default' },
					{ name: 'None (Thinking Disabled)', value: 'none' },
					{ name: 'Low', value: 'low' },
					{ name: 'Medium', value: 'medium' },
					{ name: 'High', value: 'high' },
					{ name: 'Extra High', value: 'xhigh' },
					{ name: 'Max', value: 'max' },
				],
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Frequency Penalty',
						name: 'frequencyPenalty',
						type: 'number',
						default: 0,
						typeOptions: { minValue: -2, maxValue: 2 },
						description:
							'Positive values penalise tokens by their existing frequency, reducing verbatim repetition',
					},
					{
						displayName: 'Maximum Number of Tokens',
						name: 'maxTokens',
						type: 'number',
						default: -1,
						typeOptions: { maxValue: 32768 },
						description:
							'Maximum number of tokens to generate. -1 leaves it to the model default.',
					},
					{
						displayName: 'Presence Penalty',
						name: 'presencePenalty',
						type: 'number',
						default: 0,
						typeOptions: { minValue: -2, maxValue: 2 },
						description:
							'Positive values penalise tokens that already appeared, increasing topic diversity',
					},
					{
						displayName: 'Response Format',
						name: 'responseFormat',
						type: 'options',
						default: 'text',
						options: [
							{ name: 'Text', value: 'text' },
							{ name: 'JSON', value: 'json_object' },
						],
						description:
							'JSON mode requires the word "json" to appear in your prompt, and is not compatible with thinking on some models',
					},
					{
						displayName: 'Sampling Temperature',
						name: 'temperature',
						type: 'number',
						default: 0.7,
						typeOptions: { minValue: 0, maxValue: 2, numberPrecision: 1 },
						description: 'Higher values produce more random output',
					},
					{
						displayName: 'Timeout (Ms)',
						name: 'timeout',
						type: 'number',
						default: 360000,
						description:
							'Maximum time a request may take. Thinking responses can be slow, so keep this generous.',
					},
					{
						displayName: 'Max Retries',
						name: 'maxRetries',
						type: 'number',
						default: 2,
						description: 'Maximum number of retries per request',
					},
					{
						displayName: 'Top P',
						name: 'topP',
						type: 'number',
						default: 1,
						typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 1 },
						description:
							'Nucleus sampling threshold. Change this or temperature, not both.',
					},
				],
			},
		],
	};

	methods = {
		loadOptions: {
			async getModels(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
				const credentials = await this.getCredentials('deepSeekApi');
				const baseUrl = ((credentials.url as string) ?? 'https://api.deepseek.com').replace(
					/\/+$/,
					'',
				);

				try {
					const response = (await this.helpers.httpRequest({
						method: 'GET',
						url: `${baseUrl}/models`,
						headers: { Authorization: `Bearer ${credentials.apiKey as string}` },
						json: true,
					})) as { data?: Array<{ id: string }> };

					const models = (response.data ?? [])
						.map((model) => ({ name: model.id, value: model.id }))
						.sort((a, b) => a.name.localeCompare(b.name));

					if (models.length > 0) return models;
				} catch {
					// Fall through to the static list if /models is unreachable.
				}

				return [
					{ name: 'deepseek-chat', value: 'deepseek-chat' },
					{ name: 'deepseek-reasoner', value: 'deepseek-reasoner' },
				];
			},
		},
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials('deepSeekApi');
		const modelName = this.getNodeParameter('model', itemIndex) as string;
		const thinkingLevel = this.getNodeParameter('thinkingLevel', itemIndex, 'default') as string;
		const options = this.getNodeParameter('options', itemIndex, {}) as DeepSeekOptions;

		// Anything not natively supported by the ChatOpenAI constructor is passed
		// straight through into the request body via modelKwargs.
		const modelKwargs: Record<string, unknown> = {};
		if (thinkingLevel !== 'default') {
			modelKwargs.reasoning_effort = thinkingLevel;
		}
		if (options.responseFormat) {
			modelKwargs.response_format = { type: options.responseFormat };
		}

		const { responseFormat, maxTokens, ...rest } = options;

		const model = new ChatOpenAI({
			apiKey: credentials.apiKey as string,
			model: modelName,
			...rest,
			// ChatOpenAI treats -1 as an explicit cap, so only forward real limits.
			...(maxTokens && maxTokens > 0 ? { maxTokens } : {}),
			timeout: options.timeout ?? 360000,
			maxRetries: options.maxRetries ?? 2,
			configuration: {
				baseURL: ((credentials.url as string) ?? 'https://api.deepseek.com').replace(/\/+$/, ''),
			},
			...(Object.keys(modelKwargs).length > 0 ? { modelKwargs } : {}),
		});

		return { response: model };
	}
}
