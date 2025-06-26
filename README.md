# Saiga: Telegram bot

## Description

This is a Telegram bot designed to interact with various language models. It offers support for system prompts and image processing capabilities.

## Features

- Integration with the Telegram platform
- Support for any number of language models (compatible with OpenAI API format)
- Ability to handle system prompts, generation parameters, and characters
- Image processing and generation (with gpt-image-1)
- Function calling with CodeAct (from [smolagents](https://github.com/huggingface/smolagents))

## Installation

To install and run the bot, follow these steps:

1. Clone the repository:
```
git clone https://github.com/IlyaGusev/saiga_bot
```

2. Navigate to the project directory:
```
cd saiga_bot
```

3. Install the required dependencies:
```
uv venv
make install
```

4. Add a token from [@BotFather](https://t.me/botfather) and an admin's user name and ID to [configs/bot.json](https://github.com/IlyaGusev/saiga_bot/blob/master/configs/bot.json).
5. Add URLs and tokens to [configs/providers.json](https://github.com/IlyaGusev/saiga_bot/blob/master/configs/providers.json). Modify all other configs if needed.
6. Run the bot with `make serve`
```
make serve
```

Or with an explicit command:
```
uv run python -m src.bot --bot-config-path configs/bot_prod.json --providers-config-path configs/providers.json --db-path db.sqlite --characters-path configs/characters.json --yookassa-config-path configs/yookassa.json --localization-config-path configs/localization.json --tools-config-path configs/tools_prod.json
```

## Contact

For any questions or feedback, please open an issue on the GitHub repository or contact [@YallenGusev](https://t.me/YallenGusev).
