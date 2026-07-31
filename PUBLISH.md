# Publishing checklist

## 1. Create the GitHub repo and push (one command)

```bash
cd n8n-nodes-deepseek-thinking
gh repo create pirazor/n8n-nodes-deepseek-thinking \
  --public \
  --description "🧠 Make DeepSeek think as hard as your workflow deserves, an n8n community node that unlocks DeepSeek's Thinking Level (reasoning_effort: none to max) that the built-in node never exposed." \
  --source=. --remote=origin --push
```

Without the `gh` CLI: create an **empty** repo named `n8n-nodes-deepseek-thinking`
on github.com (no README/license, the commit already has them), then:

```bash
git remote add origin https://github.com/pirazor/n8n-nodes-deepseek-thinking.git
git push -u origin main
```

Suggested repo topics: `n8n`, `n8n-community-node-package`, `deepseek`,
`llm`, `ai-agent`, `reasoning`, `langchain`

## 2. Publish to npm  (required for the Community Nodes UI)

GitHub alone does **not** make the node installable from n8n's
Settings → Community nodes. n8n installs from the npm registry, and it only
lists packages whose `keywords` contain `n8n-community-node-package`
(already set).

```bash
npm login
npm publish --access public     # prepublishOnly runs the build automatically
```

Then anyone can install it by entering `n8n-nodes-deepseek-thinking`
in Settings → Community nodes.

## 3. Optional, get it verified by n8n

Verified nodes appear in the in-app node search for every user, including
n8n Cloud. Submit at: https://github.com/n8n-io/n8n/discussions or via the
community-node verification process in n8n's docs.

## Release hygiene

- Bump `version` in `package.json` before each `npm publish`.
- Tag releases: `git tag v0.1.0 && git push --tags`.
- CI (`.github/workflows/ci.yml`) builds and asserts that
  `reasoning_effort` reaches the request payload on every push/PR.
