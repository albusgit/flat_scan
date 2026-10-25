# Speed and size, round 3 (round 2's "Not done")

Round 2's list of what was not done (`round2.md`), worked through in order. Each item is its own commit, with what
was measured. Same check as round 2 for the items that keep the output: the emulator's debug batch of 4 photos
(`--es batch t4 --es accel cpu --ez rules true`, Android 15 x86_64 AVD, 4 cores, every net on LiteRT's CPU)
against round 2's outputs (pages, page-finder logits, net grids: 20 files byte-identical). Nothing here was
measured on a phone.

## 1. XNNPACK weight cache: measured, not kept

LiteRT 1.4.0's `xnnpack_delegate.h` is vendored now (`core/third_party/litert`), and the CPU interpreters'
options are its struct, with static asserts on its LP64 layout. The hand-written mirror called the second field
`flags`; in 1.4.0 that offset is `runtime_flags`, and `flags` is the third. Only `num_threads` was ever written,
so nothing changed (batch: 20 files identical), but a flag set through the old mirror would have gone to the
wrong field.

The weight cache itself was tried: `weight_cache_file_path` = `<cache dir>/xnn_<net>_<model sha256>_<LiteRT
version>.cache`, one file per model, held by one session at a time. Results:

- **UVDoc: the build hangs.** LiteRT logs `Unknown constant buffer passed to BuildPackIdentifier` and the batch
  never finished (10 min, process idle). Turning it on for every fixed-shape net would hang the CPU path.
- **Page finder: no gain.** It built and loaded correctly (12.1 MB file; outputs identical), but its session was
  ready in 280, 341 and 319 ms from the cache against 169 and 296 ms without it. The SHA-256 of the 12.6 MB
  model plus reading the cache cost as much as packing the weights.
- The models are fp32, not fp16, so a cache is about the size of its model (12.1 MB vs 12.6 MB), not twice
  it. Caching every net would put about 70 MB in `no_backup`.
- The sessions are cheap on the emulator anyway: UVDoc 180-414 ms, SyntheticDoc high-res 63-109 ms, both made
  beside the CPU stages.

The gain is well under the 0.3 s gate, so the cache path stays unset.
