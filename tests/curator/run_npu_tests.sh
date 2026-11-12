#!/bin/bash
# End-to-end test script for NPU operation adaptation
# Usage:
#   bash tests/run_npu_tests.sh              # Run all tests
#   bash tests/run_npu_tests.sh --test 1 2 3 # Run only tests 1, 2, and 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== Configuration (adjust for your environment) =====
MODEL_DIR="../../models"
# ========================================

echo "Project directory: ${PROJECT_ROOT}"
echo ""

cd "${PROJECT_ROOT}"

# Suppress torch_npu CANN directory-owner warnings and CANN SyntaxWarning
export PYTHONWARNINGS="ignore::UserWarning:torch_npu.utils.collect_env,ignore::SyntaxWarning"

CMD=(python3 tests/test_npu_operators.py --model_dir "${MODEL_DIR}")

# Forward all additional arguments
if [ $# -gt 0 ]; then
    CMD+=("$@")
fi

exec "${CMD[@]}"
