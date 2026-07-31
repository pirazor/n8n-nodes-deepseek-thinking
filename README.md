# 🧠 n8n-nodes-deepseek-thinking

### Make DeepSeek think as hard as your workflow deserves.

[![npm version](https://img.shields.io/npm/v/n8n-nodes-deepseek-thinking.svg)](https://www.npmjs.com/package/n8n-nodes-deepseek-thinking)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![n8n community node](https://img.shields.io/badge/n8n-community%20node-FF6D5A.svg)](https://docs.n8n.io/integrations/community-nodes/)

DeepSeek can reason at wildly different depths — a quick answer or a long,
deliberate chain of thought. n8n's built-in DeepSeek node gives you **no way to
say which**. Every call runs at the server default, whether you're extracting a
date or designing a system.

**This node adds the dial.** One dropdown, from *thinking off* to *max effort*.

```
Thinking Level:  None ─ Low ─ Medium ─ High ─ Extra High ─ Max
                  ↓                                         ↓
              fast & cheap                        slow & rigorous
```

---

## Why this exists

n8n's built-in DeepSeek node instantiates LangChain's `ChatOpenAI` (DeepSeek's
API is OpenAI-compatible) and exposes only `temperature`, `topP`, `maxTokens`,
the penalties, `timeout`, `maxRetries` and `responseFormat`. It never sends
[`reasoning_effort`](https://api-docs.deepseek.com/guides/thinking_mode/), so
thinking depth is simply unreachable.

The obvious workaround doesn't work either. Pointing the built-in **OpenAI Chat
Model** node at DeepSeek's base URL looks promising — that node *does* have a
`reasoningEffort` option — but it's hidden behind a model-name regex
(`^o1…`, `^o[3-9].*`, `^gpt-5.*`) that DeepSeek model names never match, and it
only accepts `low`/`medium`/`high`, putting `max` and `none` permanently out of
reach.

So: same wiring as the official node, plus the one field that was missing.

## Thinking levels

| Option | Sent as | DeepSeek behaviour |
|---|---|---|
| **API Default** | *(omitted)* | Whatever the server decides |
| **None** | `none` | Thinking **disabled** — fastest, cheapest |
| **Low** / **Medium** | `low` / `medium` | Normalised to `high` server-side |
| **High** | `high` | Default effort in thinking mode |
| **Extra High** | `xhigh` | Normalised to `max` server-side |
| **Max** | `max` | Maximum reasoning effort |

Any value other than `none` enables thinking mode. Graded effort applies to
models that support it (e.g. DeepSeek V4 Pro / Flash); `deepseek-reasoner`
always thinks. Reference: [DeepSeek — Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/).

## Features

- 🎚️ **Thinking Level** — the whole point, exposed as `reasoning_effort`
- 🔑 **Zero new credentials** — reuses n8n's built-in `deepSeekApi` credential
- 🔄 **Live model list** — pulled from your account's `/models` endpoint
- 🧩 **Drop-in** — connects to any AI Agent's *Model* input like any chat model
- ⚙️ **Full parity** — keeps temperature, topP, maxTokens, penalties, timeout,
  retries and JSON response format

## Installation

### Community Nodes UI (recommended)

**Settings → Community nodes → Install**, then enter:

```
n8n-nodes-deepseek-thinking
```

### Custom extensions folder (self-hosted, no npm)

```bash
git clone https://github.com/pirazor/n8n-nodes-deepseek-thinking.git
cd n8n-nodes-deepseek-thinking
npm install && npm run build

mkdir -p ~/.n8n/custom
cp -r . ~/.n8n/custom/n8n-nodes-deepseek-thinking
# restart n8n
```

### Docker

```yaml
environment:
  - N8N_CUSTOM_EXTENSIONS=/home/node/.n8n/custom
volumes:
  - ./n8n-nodes-deepseek-thinking:/home/node/.n8n/custom/n8n-nodes-deepseek-thinking
```

## Usage

1. Add **DeepSeek Chat Model (Thinking)** to your canvas.
2. Select your existing **DeepSeek** credential.
3. Pick a model (e.g. `deepseek-reasoner`) and a **Thinking Level**.
4. Connect it to an AI Agent's **Model** input.

That's it — the agent now reasons at exactly the depth you chose.

### Picking a level

| Use case | Suggested level |
|---|---|
| Classification, extraction, routing | **None** |
| Summarising, drafting, rewriting | **High** |
| Multi-step analysis, code, planning | **Max** |

Thinking costs latency and tokens. Turning it **off** for simple steps is often
the biggest win in an agent pipeline.

## Development

```bash
npm install
npm run build     # → dist/nodes/LmChatDeepSeekThinking/
npm run dev       # watch mode
```

## Known limitations

- **No token-usage tracing in the n8n log panel.** The helpers that report token
  counts (`N8nLlmTracing`) are internal to `@n8n/n8n-nodes-langchain` and are not
  exported to community packages. The model works normally; only n8n's usage
  display is affected.
- **JSON response format + thinking** are incompatible on some DeepSeek models.
  Set Thinking Level to *None* if a model rejects the combination.
- Thinking responses can be slow — the default timeout is 360 000 ms.

## Contributing

Issues and PRs welcome at
[github.com/pirazor/n8n-nodes-deepseek-thinking](https://github.com/pirazor/n8n-nodes-deepseek-thinking).

## Author

**Dr. Enes Karaaslan** — Principal AI Scientist
Founder, [Connected Wise LLC](https://connectedwise.com) · [eneskaraaslan.com](https://eneskaraaslan.com)

## License

[MIT](LICENSE) © 2026 Dr. Enes Karaaslan — Connected Wise LLC

---

<sub>Not affiliated with or endorsed by DeepSeek or n8n GmbH. "n8n" is a trademark of n8n GmbH.</sub>
