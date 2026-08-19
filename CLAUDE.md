# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Saiga Bot is a Telegram bot that provides a unified interface to multiple LLM providers. It supports multi-model conversations, image processing/generation, agentic tool use (via smolagents CodeAct), document uploads (PDF/CSV/TXT), user subscriptions with payments (Telegram Stars and Yookassa), and Russian language localization.

## Commands

```bash
make install        # Install dependencies with uv
make validate       # Run black, flake8, mypy (strict) on src/ and tests/
make test           # Run pytest -s
make serve-test     # Run bot with test config (uses test_db.sqlite)
make serve          # Run bot with production config
```

Single test: `uv run pytest tests/test_provider.py::test_name -s`

## Architecture

**Entry point:** `src/bot.py::main()` - Uses `fire` CLI. The `LlmBot` class orchestrates all components.

**Core flow:**
1. Telegram message → aiogram dispatcher → handler in `bot.py`
2. Rate limit check → history retrieval from DB → `LLMProvider.reply()`
3. Optional agent execution (`src/agents.py`) with tools from `src/tools/`
4. Optional content filtering (`src/llm_filter.py`) for group chats
5. Response sent back, conversation saved to SQLite

**Key modules:**
- `src/bot.py` - Main bot class, command handlers (~1100 lines)
- `src/provider.py` - LLMProvider abstracts OpenAI-compatible API calls
- `src/agents.py` - smolagents CodeAgent integration
- `src/database.py` - SQLAlchemy ORM models (Message, ToolCall, Conversation, Payment, etc.)
- `src/configs.py` - Dataclasses for all JSON configs
- `src/payments.py` - Yookassa payment processing
- `src/tools/` - Agent tools: web search, image generation/editing, webpage visiting

**Configuration:** All configs are JSON files in `configs/`. Separate `*_prod.json` files for production. Provider configs define model capabilities (`can_handle_images`, `can_handle_tools`), token limits, and rate limits per subscription tier.

**Templates:** `templates/ru_*.jinja` - Jinja2 templates for Russian localization
**Prompts:** `prompts/` - YAML/Jinja2 agent system prompts

## Code Style

- Black formatter with 120 char line length
- Strict mypy type checking
- All async/await for I/O operations
- Flake8 ignores: E203, F403, E501, SIM115, PIE786, W503
