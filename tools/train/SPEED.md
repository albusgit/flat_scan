# Training speed on this PC (round 19 bake-off, 2026-10-27)

RTX 5060 8 GB (Blackwell, bf16), i5-14400F (16 threads), 32 GB RAM, torch 2.11 cu128. All numbers measured here.

## Page finder v2 (`pagefind2/train.py`)
| Setting | Photos/s | Peak VRAM | Loss over the timed steps | Verdict |
|---|---|---|---|---|
| fp32, batch 16 (TF32 on or off) | - | out of memory | - | no |
| fp32, batch 8 (reference) | 13.2 | 4.4 GB | 0.330 / last 50: 0.284 | reference |
| bf16, batch 8 | 98.4 | 3.1 GB | 0.336 / 0.290 | matches the reference |
| bf16, batch 16 | 71.6 | 5.4 GB | - | (before channels_last) |
| **bf16, batch 16, channels_last** | **110.5** | 5.4 GB | 0.277 / 0.218 | **used** |
| + torch.compile | 108.6 | 4.0 GB | 0.276 / 0.218 | no gain |
| bf16, batch 24 | - | out of memory | - | no |

- **Bug found:** the GPU JPEG decode (`torchvision.io.decode_jpeg(device='cuda')`, batched) runs on its own stream. Without a `torch.cuda.synchronize()` before it, the weights went NaN within about 10 steps, every run.
  - A sync before the decode fixes it; a sync after does not. Speed unchanged (6.94 vs 7.0 steps/s).
  - The earlier "bf16 NaN" and the torch.compile `cudaErrorAssert` were this bug.
- **Evaluation:** odd batch sizes made cudnn.benchmark re-tune: 77 s per evaluation. Now it runs at the training batch size with the last batch padded: 4 s.
- **Startup:** the first 100 steps take about 75 s (tuning), once per launch.

## SD-HR (`flat2/train_run2.py`, `flat2/sdhr_bench.py`)
| Micro-batch | channels_last | compile (encoder) | ms / sample | Peak VRAM |
|---|---|---|---|---|
| 2 | - | - | 24.7 | 1.2 GB |
| 4 | - | - | 24.3 | 1.9 GB |
| 6 | - | - | 24.4 | 2.5 GB |
| 4 | yes | - | 22.8 | 1.9 GB |
| 4 | - | yes | 22.6 | 1.5 GB |
| **4** | **yes** | **yes** | **20.3** | 1.5 GB |

- The loss is the same in every variant (43.0283 on fixed random inputs).
- In the real trainer: 0.36 s per 16-sample step (GPU bound).
- **Also changed:**
  - fused AdamW;
  - a non-finite step is skipped instead of stopping the run;
  - a rolling `last.pt` every 500 steps;
  - the encoder is compiled in place, so checkpoint keys are unchanged.
- **RAM:** a memory-mapped UVDoc grid file (868 MB) was copied into every DataLoader worker, because Windows pickles the dataset. Now each worker maps it itself (`Mixed.__getstate__`). The trainer's RAM went from 12.9 to 7.2 GB.

## Synthetic generator (`flat2/make_both.py`)
| Workers | Photos/s | CPU |
|---|---|---|
| 1 | 0.75 | - |
| 4 | 1.70 | 35 % |
| **8** | **2.03** | 60 % |
| 12 | 1.98 | - |

- Each worker uses a full core but runs about 3× slower when 8 run together. It is memory-bandwidth bound: dozens of full-frame float32 passes per photo.
- 8 workers make as much as 12, with 4 GB less RAM.
- The rasteriser now sizes its pixel box per face group, not to the largest face. The output is identical (0 pixels differ), but it is only 4 % faster: not the bottleneck.

## Memory, measured (committed / in RAM)
| Job | Committed | In RAM |
|---|---|---|
| Generator, 8 workers | 12.9 GB | 9.2 GB |
| Page-finder trainer | 10.1 GB | 3.3 GB |
| SD-HR trainer (5–6 workers) | 15.7 GB | 7.2 GB |

`overnight_r19.py`:
- caps each job's committed memory (Windows job objects) at about 1.35× these;
- suspends the generator when free RAM falls below 3 GB;
- restarts any stage that dies from its newest checkpoint.

## The generator beside the trainer (round 19, 2026-10-28)
Page finder, batch 16, 150 timed steps; SD-HR, counts 4,3,3 + 10 generated, micro 4, steps 300-400 (after the warm-up).

| Setup | Page finder steps/s | SD-HR s/step |
|---|---|---|
| generator inside the trainer (8 / 4 fresh per step) | 3.16 | 0.52 |
| no fresh samples (the ring only: the ceiling) | 6.31 | 0.34 |
| generator in its own process (`--gen_dev`), on the CPU so it takes no GPU time | 5.93 | 0.46 (the CPU generator competes with the 6 data workers for the CPU) |
| generator in its own process on the same GPU | 0.83 | - |

- Making fresh samples is half of the page finder's step and a third of SD-HR's.
- `gpugen/feeder.py` runs the generator in a child process; the trainer takes finished batches from shared memory without waiting (overhead about 6 %).
- On a GPU of its own (`--gen_dev cuda:1`, a 4-GPU box): about 1.9 × for the page finder and up to 1.5 × for SD-HR expected; not measured (one GPU here).
- Two processes on one GPU are 4 × slower (Windows time-slicing), so 1-2-GPU boxes keep the generator inside the trainer.
