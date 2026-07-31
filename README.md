<p align="center">
  <img src="nodes/LmChatDeepSeekThinking/deepseek.png" width="84" alt="DeepSeek">
</p>

<h1 align="center">n8n-nodes-deepseek-thinking</h1>

<p align="center">
  <b>DeepSeek chat model for n8n — with reasoning effort control.</b><br>
  <sub>The one setting the built-in node never exposed.</sub>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/n8n-nodes-deepseek-thinking"><img src="https://img.shields.io/npm/v/n8n-nodes-deepseek-thinking.svg" alt="npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"></a>
  <a href="https://docs.n8n.io/integrations/community-nodes/"><img src="https://img.shields.io/badge/n8n-community%20node-FF6D5A.svg" alt="n8n community node"></a>
</p>

---

## The problem

n8n's built-in DeepSeek node builds a LangChain `ChatOpenAI` client but never
sends [`reasoning_effort`](https://api-docs.deepseek.com/guides/thinking_mode/).
Every call runs at the server default — no way to think harder on a hard task,
or skip thinking on a trivial one.

Routing the built-in **OpenAI Chat Model** node to DeepSeek's base URL doesn't
help: its `reasoningEffort` field is hidden behind a model-name regex
(`^o1…`, `^o[3-9].*`, `^gpt-5.*`) that DeepSeek models never match, and it
accepts only `low`/`medium`/`high` — leaving `max` and `none` unreachable.

## The fix

Same node, plus a **Thinking Level** dropdown sent as top-level `reasoning_effort`.

| Setting | Sent as | Behaviour |
|---|---|---|
| API Default | *(omitted)* | Server decides |
| None | `none` | Thinking off — fastest, cheapest |
| Low / Medium | `low` / `medium` | Normalised to `high` |
| High | `high` | Default thinking effort |
| Extra High | `xhigh` | Normalised to `max` |
| Max | `max` | Maximum effort |

Any value except `none` enables thinking mode.

## Models

Graded effort is honoured by **DeepSeek V4 Pro** and **V4 Flash**, which map
`low`/`medium` → `high` and `xhigh` → `max`. `deepseek-reasoner` always thinks;
`deepseek-chat` responds directly. The model list is loaded live from your
account's `/models` endpoint.

## Install

**Community Nodes UI** — Settings → Community nodes → Install:

```
n8n-nodes-deepseek-thinking
```

**Self-hosted, from source:**

```bash
git clone https://github.com/pirazor/n8n-nodes-deepseek-thinking.git
cd n8n-nodes-deepseek-thinking && npm install && npm run build
cp -r . ~/.n8n/custom/n8n-nodes-deepseek-thinking   # then restart n8n
```

## Usage

Add **DeepSeek Chat Model (Thinking)**, select your existing DeepSeek
credential, pick a model and Thinking Level, and connect it to an AI Agent's
**Model** input.

Reuses n8n's built-in `deepSeekApi` credential — nothing new to configure.
Temperature, topP, maxTokens, penalties, timeout, retries and JSON response
format all work as before.

> **Tip:** thinking costs latency and tokens. Setting **None** on classification
> and extraction steps is often the biggest speedup in an agent pipeline.

## Notes

- Token usage won't appear in n8n's log panel — the tracing helpers are internal
  to `@n8n/n8n-nodes-langchain` and aren't exported to community packages. The
  model itself is unaffected.
- JSON response format and thinking are incompatible on some models; use
  **None** if a model rejects the pair.
- If the node doesn't appear in an Agent's Model dropdown, set
  `N8N_COMMUNITY_PACKAGES_ALLOW_TOOL_USAGE=true` and restart n8n.

## License

[MIT](LICENSE) © 2026 Dr. Enes Karaaslan — [Connected Wise LLC](https://connectedwise.com)

<sub>Not affiliated with or endorsed by DeepSeek or n8n GmbH.</sub>
