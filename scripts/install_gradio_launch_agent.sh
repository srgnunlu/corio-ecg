#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.sergenunlu.corio-ecg.gradio.plist"
LABEL="com.sergenunlu.corio-ecg.gradio"
SOURCE_PLIST="${SCRIPT_DIR}/launchd/${PLIST_NAME}"
TARGET_DIR="${HOME}/Library/LaunchAgents"
TARGET_PLIST="${TARGET_DIR}/${PLIST_NAME}"
DOMAIN="gui/$(id -u)"

mkdir -p "${TARGET_DIR}"
cp "${SOURCE_PLIST}" "${TARGET_PLIST}"

launchctl bootout "${DOMAIN}" "${TARGET_PLIST}" >/dev/null 2>&1 || true
launchctl bootstrap "${DOMAIN}" "${TARGET_PLIST}"

cat <<EOF
LaunchAgent installed:
  ${TARGET_PLIST}

Start the service:
  launchctl kickstart -k ${DOMAIN}/${LABEL}

Check status:
  launchctl print ${DOMAIN}/${LABEL}

Stop and unload:
  launchctl bootout ${DOMAIN} ${TARGET_PLIST}
EOF
