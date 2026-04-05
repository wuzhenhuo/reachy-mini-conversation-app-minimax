#!/bin/bash
# Startup script for Reachy Mini Conversation App with MiniMax

PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
CERTIFI=$($PYTHON -c "import certifi; print(certifi.where())")

export SSL_CERT_FILE="$CERTIFI"
export REQUESTS_CA_BUNDLE="$CERTIFI"
export OPENAI_API_KEY="sk-api-F1AgRpGcM85w0UbPtTLaMlH0-N890rUQ7MF0GqtLbTES1v9OjuIG2itCpo_bGwk8s843EWaNCq2qu2rUSsYbsECtybKws6CHHeaNdgwoyAnJKDdHq65eJZs"
export OPENAI_BASE_URL="https://api.minimax.chat/v1"
export MODEL_NAME="MiniMax-M2.7"
export REACHY_MINI_CUSTOM_PROFILE="example"

cd "$(dirname "$0")"

exec /Library/Frameworks/Python.framework/Versions/3.12/bin/reachy-mini-conversation-app --gradio --no-camera "$@"
