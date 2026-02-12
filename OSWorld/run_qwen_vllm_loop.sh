#!/bin/bash

# Simple loop to run qwen_vllm_generate_and_verify every 15 minutes.
# You can override the Python binary by exporting PYTHON_BIN, e.g.:
#   export PYTHON_BIN=/path/to/venv/bin/python

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="/gpfs/radev/home/jw3278/project/GUI_agent/OSWorld/qwen_vllm_generate_and_verify.py"
LOG_DIR="/gpfs/radev/home/jw3278/project/GUI_agent/OSWorld/logs"
LOG_FILE="${LOG_DIR}/qwen_vllm_loop.log"

mkdir -p "${LOG_DIR}"

echo "Starting qwen_vllm_generate_and_verify loop. Logging to ${LOG_FILE}"

while true; do
  START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$START_TS] Starting qwen_vllm_generate_and_verify.py" | tee -a "${LOG_FILE}"

  "${PYTHON_BIN}" "${SCRIPT_PATH}" >> "${LOG_FILE}" 2>&1

  END_TS="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$END_TS] Finished run, sleeping 900 seconds (15 minutes)." | tee -a "${LOG_FILE}"

  sleep 900
done


