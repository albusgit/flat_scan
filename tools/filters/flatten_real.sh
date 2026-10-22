#!/bin/bash
# Flattened pages (as the app makes them: page finder, both nets, ShadeNet) for the filter eval:
# flatten_real.sh OUTDIR PHOTO...   writes OUTDIR/<name>_page.png and <name>_paper.png
R=$(cd "$(dirname "$0")/../.." && pwd)
B=$R/tools/desktop_check/build/cpu/pipeline_cli.exe
export PATH="/c/Users/User/Android/opencv/build/x64/vc16/bin:$PATH"
O=$1; shift; mkdir -p $O
for P in "$@"; do
  n=$(basename "${P%.*}")
  [ -f $O/${n}_page.png ] && continue
  FS_NETS=both FS_SHADE_NET=$R/app/src/main/assets/models/shadenet.onnx FS_ORT_THREADS=4 FS_PAGE_PNG=$O/$n \
    $B "$P" $O/$n.jpg net > $O/$n.json 2>$O/$n.err || echo "fail $n"
done
