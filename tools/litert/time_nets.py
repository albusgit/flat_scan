"""CPU time per invoke of .tflite files on the desktop (tensorflow 2.19.1's tf.lite interpreter: XNNPACK delegate, and
its builtin kernels), interleaved so that the shared machine's load hits every file alike. Indicative only.
  python tools/litert/time_nets.py THREADS ROUNDS FILE.tflite ...   (random input of the file's shape)"""
import sys
import time

import numpy as np
import tensorflow as tf

th, rounds, files = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
its = {}
for f in files:
    for mode in ('xnnpack', 'builtin'):
        kw = {} if mode == 'xnnpack' else {
            'experimental_op_resolver_type': tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
        try:
            it = tf.lite.Interpreter(model_path=f, num_threads=th, **kw)
            it.allocate_tensors()
        except Exception as e:
            print(f'{f} {mode}: {str(e)[:120]}')
            continue
        d = it.get_input_details()[0]
        it.set_tensor(d['index'], np.random.default_rng(0).random(d['shape'], dtype=np.float32))
        it.invoke()   # warm-up
        its[(f, mode)] = (it, [])
for r in range(rounds):
    for k, (it, ts) in its.items():
        t0 = time.perf_counter()
        it.invoke()
        ts.append(1000 * (time.perf_counter() - t0))
for (f, mode), (it, ts) in its.items():
    print(f'{np.median(ts):8.0f} ms  min {min(ts):6.0f}  {mode:8s} {f}')
