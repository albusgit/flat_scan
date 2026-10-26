#!/usr/bin/env bash
# One-time setup of a rented GPU box (Ubuntu, PyTorch CUDA image, e.g. pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel).
#   bash setup.sh            (from flat_scan/tools/train/vast, after unpacking flatscan_code.tar and flatscan_data.tar)
# FS_DATA (default /data): where flatscan_data.tar was unpacked; WORK (default /work): pools, banks, runs.
set -euo pipefail
FS_DATA=${FS_DATA:-/data}; WORK=${WORK:-/work}
HERE=$(cd "$(dirname "$0")" && pwd); TOOLS=$(cd "$HERE/../.." && pwd)
mkdir -p "$WORK"

# fonts: the rule-net generator's Linux list (DejaVu, Liberation, FreeFont), Noto for the multilingual text pages
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq fonts-dejavu-core fonts-liberation fonts-freefont-ttf fonts-noto-core fonts-noto-cjk \
    fonts-noto-extra libgl1 libglib2.0-0 libfontconfig1 libxkbcommon0 libegl1 libdbus-1-3 > /dev/null
pip install -q -r "$TOOLS/train/requirements.txt"

# every font in one folder (the generators take a font directory): the system's plus the bundle's Google fonts
mkdir -p "$FS_DATA/fonts"
find /usr/share/fonts "$FS_DATA/round4/assets/gfonts" -type f \( -iname '*.ttf' -o -iname '*.otf' -o -iname '*.ttc' \) \
    -exec cp -n {} "$FS_DATA/fonts/" \;
echo "fonts: $(ls "$FS_DATA/fonts" | wc -l)"

# the multilingual corpus with the fonts of this box (the bundled one names Windows fonts)
python "$TOOLS/text_net/corpus.py" "$FS_DATA/textnet/udhr2" "$FS_DATA/textnet/corpus_linux.json" "$FS_DATA/fonts"

python - <<'EOF'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'x',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
EOF
echo "setup done. Next: run_shade.sh, run_rules.sh, run_flat.sh (README.md)"
