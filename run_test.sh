#!/usr/bin/env bash
set -euo pipefail

# Remove system CUDA lib path to avoid picking up mismatched cuDNN
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/usr/local/cuda-12.4/lib64' | paste -sd:)"
fi

# Run tests (adjust to your project)
python demo.py --dataset=Indian --patches=7 --flag_test=test --model_name=MSF_Heat3D --epoches=500 --batch_size=64
