#!/bin/bash
set -euo pipefail

python3 -m src.bot --bot-config-path $1 --providers-config-path configs/providers.json --db-path $2 --characters-path configs/characters.json --tools-config-path configs/tools.json --yookassa-config-path configs/yookassa.json --localization-config-path configs/localization.json
