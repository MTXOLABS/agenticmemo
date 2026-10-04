# Escape Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅        |
| < 2.0   | ❌        |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub issue.

- Email: **animaytiwari123@gmail.com** with subject line `[SECURITY] Escape (agenticmemo)`
- Include: affected version, reproduction steps, and impact assessment if known.

You will receive an acknowledgement within 72 hours. We ask for up to 90 days to
ship a fix before public disclosure.

## Security model — read this before deploying

Escape provides memory for existing agents and a bundled LLM agent runtime.
The `AgentMemory` plug-in supplies context and records application-approved data;
it does not execute retrieved code or tools. The bundled `Agent` runtime can
execute LLM-driven workflows. Security depends on **how you deploy and connect
these components**. Know these boundaries:

### 1. `PythonReplTool` executes arbitrary code — unsandboxed

`PythonReplTool` runs LLM-generated Python via `exec()` **in your process, with your
permissions**. This is by design (it's a REPL tool), but it means:

- Never enable it on tasks containing untrusted input (user uploads, scraped web
  content, third-party documents) without OS-level sandboxing.
- For production, run the agent inside a container/VM with minimal privileges,
  no secrets in the environment, and restricted network egress.

### 2. `FileReadTool` / `FileWriteTool` have no path restrictions

The built-in file tools can read and write **any path the process can access**.
If your threat model includes prompt injection (it should — see below), wrap them
with your own allowlist or run the agent in a chroot/container before exposing them.

### 3. Prompt injection and memory poisoning

Agent memory introduces a specific risk: **content retrieved from memory is injected
into future prompts**. If an attacker can influence what gets stored (e.g., via a
scraped webpage that manipulates the agent's trajectory), they can plant instructions
that resurface in later, unrelated tasks ("stored prompt injection").

The plug-in excludes failed, unknown, and stale experience from retrieval, but
passing validation is a declaration by the calling application. It is not proof
that content is safe. Retrieved context must remain untrusted data. Scope labels
partition records; the application must authorize scope access.

In the bundled runtime, tool outputs are truncated, and trajectory filtering and
outcome verification can help assess which experiences to retain.
Mitigations you should add for hostile-input deployments: review memory contents
periodically, use a separate memory file per trust domain, and never share a memory
pool between trusted and untrusted workloads.

### 4. Memory files are plaintext

Persisted memory (SQLite databases and legacy `*.json` files) may contain task
text, approved actions and metadata, tool inputs/outputs, and answers.
Treat these files with the same sensitivity as application logs: exclude them from
backups you don't control, don't commit them, and encrypt at rest if tasks contain
confidential data.

### 5. API keys

Keys are passed to the official Anthropic/OpenAI SDKs and are never logged or
persisted as configuration by Escape. Prefer environment variables over hardcoding
keys, and exclude credentials from task text, tool output, and memory payloads.
