#!/bin/bash
set -euo pipefail

python3 -m src.bot --bot-token $1 --client-config-path client_config.json --db-path db.sqlite --characters-path characters.json
