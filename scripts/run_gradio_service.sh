#!/bin/zsh

set -euo pipefail

cd "/Users/sergenunlu/Developer/active/apps/Corio ECG"

export CORIO_GRADIO_ENABLE_DEWARP_RETRY=0

exec .venv/bin/python -u -m src.web.app
