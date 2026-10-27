Finger detection runs (tools/fingers/eval.py). JSON per run: summary and per-photo rows.

| run | rules | mean IoU | pooled IoU | detected | control FP rate | control area filled |
|---|---|---|---|---|---|---|
| r0 | before: holes only | 0.077 | 0.047 | 5 / 32 | 0.130 (46 photos) | 0.72 % |
| r1 | + contact | 0.077 | 0.047 | 5 / 32 | 0.128 (47 photos) | 0.71 % |
| r4 | + finger width | 0.012 | 0.001 | 1 / 32 | 0.043 (46 photos) | 0.03 % |
| g8 | + skin regions | 0.127 | 0.167 | 11 / 32 | 0.152 (46 photos) | 1.20 % |
| g12 | + skin regions + width | 0.064 | 0.131 | 8 / 32 | 0.065 (46 photos) | 0.51 % |
| k16 | colour + sides | 0.064 | 0.046 | 4 / 32 | 0.065 (46 photos) | 0.08 % |
| k24 | skin regions + colour + sides | 0.136 | 0.167 | 11 / 32 | 0.065 (46 photos) | 0.08 % |
| default | default build (= skin regions + colour + sides) | 0.136 | 0.167 | 11 / 32 | 0.065 (46 photos) | 0.08 % |
