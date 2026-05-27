#!/usr/bin/env bash
set -e
echo "=== installing torch + deps (CUDA 12) ==="
pip install --quiet --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
pip install --quiet --break-system-packages numpy scipy matplotlib tqdm certifi gudhi 2>&1 | tail -3
pip install --quiet --break-system-packages adversarial-robustness-toolbox autoattack 2>&1 | tail -3
echo "=== verify ==="
python3 -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python3 -c "import gudhi; print('gudhi', gudhi.__version__)"
python3 -c "from art.attacks.evasion import FastGradientMethod, SquareAttack, ProjectedGradientDescent; print('ART OK')"
python3 -c "from autoattack import AutoAttack; print('autoattack OK')" 2>&1 | head -1
echo "=== done ==="
