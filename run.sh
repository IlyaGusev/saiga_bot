#!/bin/bash
set -euo pipefail

python3 -m src.bot --bot-token $1 --client-config-path configs/providers.json --db-path db.sqlite --characters-path configs/characters.json --tools-config-path configs/tools.json --yookassa-config-path configs/yookassa.json --localization-config-path configs/localization.json
