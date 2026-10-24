# ScoutX

ScoutX is an agentic, keyboard-first desktop file discovery and recommendation system. It combines multi-tool search with an adaptive reflection router, optional dense vector retrieval, grounded LLM explanations, optional Laya shadow advice, and an on-demand Spotlight-style desktop launcher.

The core CLI package is entirely self-contained and operates with **zero mandatory third-party runtime dependencies**. Modern GUI support via PySide6 (Qt) is fully supported as an optional extra.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Search Configuration](#search-configuration)
- [Optional Providers](#optional-providers)
  - [LLM Explanations](#llm-explanations)
  - [Dense Vector Embeddings](#dense-vector-embeddings)
  - [Laya Shadow Mode](#laya-shadow-mode)
- [Configuration and Data Locations](#configuration-and-data-locations)
- [Privacy and Security](#privacy-and-security)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Development and Testing](#development-and-testing)

---

## Overview

Traditional desktop search engines rely on rigid keyword matches or uncalibrated full-system indexing. ScoutX uses an adaptive multi-lane routing agent to iteratively observe, reflect, and retrieve:

- **Adaptive Routing Loop**: Classifies queries into `fast`, `standard`, or `deep` lanes. It reflects on candidate relevance, applies keyword/filetype/temporal constraints, and escalates lanes only when confidence is low.
- **Dense & Lexical Retrieval**: Combines BM25-style keyword search, fuzzy matching, and recency/frequency scoring with persistent SQLite-backed vector embeddings.
- **Grounded LLM Explanations**: Generates concise, privacy-bounded explanations for why a file matches the user's intent, with automatic degraded offline fallback if the provider is unavailable.
- **Laya System-1 Shadow Telemetry**: Connects optionally to the non-autoregressive Laya decision service to record shadow routing advice without allowing external predictions to alter ScoutX's routing decisions.
- **Spotlight-Style Desktop Launcher**: A frameless, keyboard-driven floating search dialog with 150 ms debounce, immediate generation invalidation, and native file execution without shell risk.

---

## Installation

ScoutX requires **Python 3.11 or higher**.

### 1. Core CLI Installation (Zero Dependencies)

The core engine and command-line interface require only the Python standard library:

```bash
# Clone the repository
https://github.com/MD-KAMRAAN-AJMAL/ScoutX.git
cd ScoutX

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core CLI
pip install .
```

### 2. Optional Desktop GUI Installation

To enable the desktop Spotlight launcher (`scoutx launch` / `scoutx gui`), install the package with the `gui` extra, which installs `PySide6`:

```bash
pip install '.[gui]'
```

---

## Quick Start

### Check CLI Availability

```bash
scoutx --help
```

### Search Files via the Command Line

Search your current directory using the autonomous routing agent:

```bash
# Basic lexical search
scoutx search "quarterly revenue report" --root .

# Retrieve up to 10 results with router trace diagnostics printed to stderr
scoutx search "revenue" --root . --limit 10 --trace
```

### Record a File-Open Event (Personalization)

Log opened files to boost relevant documents in future searches:

```bash
scoutx log-open path/to/document.pdf --usage-db ~/Library/Application\ Support/ScoutX/usage.db --topic finance
```

### Open the Desktop Launcher

If PySide6 is installed, launch the floating search bar:

```bash
# Launch using settings defaults
scoutx launch

# Alternative alias
scoutx gui

# Launch targeting a specific directory
scoutx launch --root ~/Documents
```

---

## Search Configuration

The search engine accepts explicit paths or defaults to standard system directories:

| Flag | Default | Description |
|---|---|---|
| `--root PATH` | Current working directory | Root directory scanned recursively for matching files. |
| `--limit N` | `5` (CLI) / `8` (GUI) | Maximum recommendations returned (clamped between 1 and 50). |
| `--usage-db PATH` | OS data directory `usage.db` | SQLite database storing query history and user profile scores. |
| `--index-db PATH` | OS data directory `index.db` | Dedicated SQLite database storing file metadata and vector index embeddings. |
| `--trace` | Disabled | Emits internal routing iterations, lane choices, candidate counts, and reflection confidence to stderr. |

---

## Optional Providers

ScoutX supports optional external providers for dense embeddings, LLM reasoning, and shadow routing. All secrets must be supplied via **environment variables**; API keys are never passed as literal command-line arguments.

### LLM Explanations

ScoutX can generate grounded explanations using Ollama, OpenAI-compatible APIs, or Google Gemini. When enabled, prompts are constructed under a strict total character budget (`max_input_chars`, default: 2,400 characters, clamped [256, 32,768]). Candidate snippets sent to the model are compact excerpts starting at a maximum of 240 characters per candidate (compacting down to 120, 60, 24, or 0 characters if needed). Host filesystem paths are strictly excluded.

```bash
# 1. Local Ollama (no API key required)
scoutx search "project roadmap" --root ~/Projects \
  --llm-provider ollama \
  --llm-model llama3 \
  --llm-base-url http://127.0.0.1:11434/v1

# 2. OpenAI-compatible API (default env: OPENAI_API_KEY, default timeout: 3.0s)
export OPENAI_API_KEY="your-api-key"
scoutx search "client invoice" --root ~/Documents \
  --llm-provider openai_compatible \
  --llm-model gpt-4o-mini \
  --llm-api-key-env OPENAI_API_KEY \
  --llm-timeout 3.0

# 3. Google Gemini (default env: GEMINI_API_KEY, default timeout: 3.0s)
export GEMINI_API_KEY="your-api-key"
scoutx search "financial audit" --root ~/Documents \
  --llm-provider gemini \
  --llm-model gemini-1.5-flash \
  --llm-api-key-env GEMINI_API_KEY
```

- `--require-llm`: Fails fast with exit code 2 if the LLM provider is unreachable.
- `--offline-fallback`: Enables graceful fallback to deterministic heuristic explanations if the provider is down.

### Dense Vector Embeddings

By default, ScoutX uses a built-in zero-dependency deterministic hash embedding (`local_hash`). You can configure dense neural embeddings for semantic similarity. Individual text inputs are truncated to `max_input_chars` (default: 4,000 characters) and batched in groups of 16:

```bash
# 1. Local Hash (Default, 64-dim, zero dependencies)
scoutx search "budget" --root ~/Documents --embedding-provider local_hash

# 2. Local Ollama Embeddings
scoutx search "budget" --root ~/Documents \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text \
  --embedding-base-url http://127.0.0.1:11434/v1

# 3. OpenAI-compatible Embeddings (default env: SCOUTX_OPENAI_API_KEY)
export SCOUTX_OPENAI_API_KEY="your-api-key"
scoutx search "budget" --root ~/Documents \
  --embedding-provider openai_compatible \
  --embedding-model text-embedding-3-small \
  --embedding-base-url http://127.0.0.1:8000/v1 \
  --embedding-api-key-env SCOUTX_OPENAI_API_KEY

# 4. Google Gemini Embeddings (default env: SCOUTX_GEMINI_API_KEY)
export SCOUTX_GEMINI_API_KEY="your-api-key"
scoutx search "budget" --root ~/Documents \
  --embedding-provider gemini \
  --embedding-model text-embedding-004 \
  --embedding-api-key-env SCOUTX_GEMINI_API_KEY \
  --embedding-timeout 5.0
```

### Laya Shadow Mode

ScoutX can evaluate query complexity against a running [Laya](https://github.com/NandhaKishorM/laya) non-autoregressive service. This operates in **strict shadow telemetry mode**; Laya advice is recorded in trace logs but never overrides ScoutX's internal router decisions.

The query string and its constructed state description are capped by `max_query_chars` (default: 500 characters) and contain only sanitized query metadata (the raw query string, intent, keywords, filetypes, filename hints, and complexity score). The serialized HTTP request payload additionally contains fixed question definitions (`LAYA_ROUTING_QUESTIONS`). Host filesystem paths, document contents, candidate snippets, and profile statistics are strictly excluded.

```bash
# Enable Laya shadow advice with a 500 ms client deadline
scoutx search "search query" --root . \
  --laya-shadow \
  --laya-base-url http://127.0.0.1:8001 \
  --laya-timeout 0.5 \
  --trace
```

---

## Configuration and Data Locations

ScoutX adheres to OS-standard directory conventions. The configuration file is separated from persistent databases to support independent backup and reset workflows:

### macOS

- **Configuration**: `~/Library/Application Support/ScoutX/config.json`
- **Persistent Databases**: `~/Library/Application Support/ScoutX/usage.db` and `index.db`

### Windows

- **Configuration**: `%APPDATA%\ScoutX\config.json` (e.g. `C:\Users\<User>\AppData\Roaming\ScoutX\config.json`)
- **Persistent Databases**: `%LOCALAPPDATA%\ScoutX\usage.db` and `index.db` (e.g. `C:\Users\<User>\AppData\Local\ScoutX\`)

### Linux / Unix

- **Configuration**: `$XDG_CONFIG_HOME/scoutx/config.json` (defaults to `~/.config/scoutx/config.json`)
- **Persistent Databases**: `$XDG_DATA_HOME/scoutx/usage.db` and `index.db` (defaults to `~/.local/share/scoutx/`)

### Launcher Settings File (`config.json`)

The desktop launcher automatically creates and reads `config.json`. Example structure:

```json
{
  "search_root": "/Users/user/Documents",
  "result_limit": 8,
  "debounce_ms": 150,
  "hide_on_focus_loss": true,
  "embedding_provider": "local_hash",
  "llm_provider": null,
  "laya_shadow": false
}
```

*Note: ScoutX rejects API keys stored directly inside `config.json`. Only environment variable names (e.g. `"llm_api_key_env": "MY_KEY"`) are accepted.*

---

## Privacy and Security

ScoutX is designed with a local-first privacy boundary:

1. **No Raw Document Storage**: The SQLite index stores document metadata (`id`, `path`, `title`, `extension`, `mtime`, `size`, `content_hash`) and vector embeddings. Raw document bodies are never saved into SQLite databases.
2. **Local Ingestion Limits**: Document ingestion (`DocumentLoader`) reads plain-text files up to a 2 MiB file-size cap (200,000 extracted characters) and compressed Office archives up to 10 MiB (8 MiB uncompressed XML member cap). Files exceeding these caps or identified as binary are safely skipped.
3. **Zero Document Exposure to Laya**: In Laya shadow mode, only query-level metadata is transmitted: the raw query string (capped at 500 characters by `max_query_chars`), intent keywords, and numerical complexity score, alongside fixed routing schema questions. Document names, filesystem paths, snippets, and profile data are **strictly excluded**.
4. **Bounded LLM Excerpts (Path-Free)**: When an external LLM provider is explicitly enabled, ScoutX does **not** transmit full documents or 2 MiB files. Instead, it constructs a compact prompt bounded by `max_input_chars` (default: 2,400 characters, clamped [256, 32,768]). Candidate snippets are compact excerpts capped at at most 240 characters per candidate (compacting down to 120, 60, 24, or 0 characters to fit the prompt budget). Full host filesystem paths are strictly excluded. Detected secrets and credentials are automatically redacted. If you require zero data egress, use a local provider like Ollama or rely solely on ScoutX's built-in offline lexical/hash search.
5. **Safe Subprocess Execution**: System file opening uses direct system calls (`open` on macOS, `os.startfile` on Windows, `xdg-open` on Linux) with tokenized argument arrays and `shell=False`, preventing shell injection attacks.

---

## Troubleshooting

### 1. `Error: GUI dependencies not installed`
- **Cause**: `scoutx launch` or `scoutx gui` was executed in an environment without PySide6.
- **Resolution**: Install the optional GUI extra via `pip install 'scoutx[gui]'` (or `pip install PySide6>=6.5.0`). The command exits with code 2 without emitting a traceback.

### 2. Zero Results Returned
- **Cause**: Search root directory does not contain files matching the query keywords, or files exceed local ingestion size limits (plain-text files exceeding 2 MiB or Office archives exceeding 10 MiB are skipped by `DocumentLoader`).
- **Resolution**: Broaden your query, run with `--trace` to view candidate generation across tools, or verify that the target directory is passed via `--root`.

### 3. Provider Connection / Timeout Errors
- **Cause**: Remote LLM, embedding endpoint, or Laya service is not running or takes longer than the configured timeout.
- **Resolution**:
  - For LLMs, pass `--offline-fallback` to allow degraded heuristic scoring when offline.
  - Verify endpoints (`http://127.0.0.1:11434` for Ollama, `http://127.0.0.1:8001` for Laya).
  - Check timeouts (`--llm-timeout 3.0`, `--embedding-timeout 5.0`, `--laya-timeout 0.5`).

### 4. Malformed Settings Recovery
- If `config.json` becomes corrupted or invalid, ScoutX logs a diagnostic warning and loads safe defaults without crashing. Delete or fix `config.json` in your OS configuration directory to reset preferences.

---

## Known Limitations

- **No System-Wide Global Hotkey Daemon**: ScoutX does not install a persistent background daemon for global OS shortcuts (e.g. `Cmd+Space` or `Alt+Space`). You can invoke `scoutx launch` from your terminal or bind the command to a custom OS hotkey using macOS Shortcuts/Automator, Windows Shortcut Properties, or Linux Desktop Keybinding settings.
- **No In-Window Document Preview**: The desktop launcher renders file metadata, match scores, path context, and grounded explanations. Double-clicking or pressing `Enter` opens the document in your OS's default application.
- **Laya Integration is Shadow-Only**: In compliance with safety standards, Laya suggestions are collected strictly for comparison telemetry and do not modify the autonomous router's decision logic.

---

## Development and Testing

The ScoutX test suite is written using standard library `unittest` with zero mandatory dependencies.

### Run All Unit and Integration Tests

```bash
# Run complete test suite (Slices 1 to 6)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

### Run Tests by Slice

```bash
# Baseline engine and retrieval tests (Slice 1)
python3 -m unittest tests/test_baseline.py -v

# Adaptive router tests (Slice 2)
python3 -m unittest tests/test_router.py -v

# LLM provider and explanation tests (Slice 3)
python3 -m unittest tests/test_llm.py -v

# Persistent SQLite storage and embeddings tests (Slice 4)
python3 -m unittest tests/test_slice4_storage_and_embeddings.py -v

# Laya shadow mode advisory tests (Slice 5)
python3 -m unittest tests/test_slice5_laya.py -v

# Desktop launcher and GUI lifecycle tests (Slice 6)
python3 -m unittest tests/test_slice6_launcher.py -v
```

### Check Git Formatting

```bash
git diff --check
```
