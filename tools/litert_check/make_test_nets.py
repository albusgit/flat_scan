# Tiny UVDoc-shaped test nets for tools/litert_check: input NCHW [1,3,712,488] -> output [1,2,45,31].
# Writes OUT/full/{uvdoc.onnx, uvdoc_static.tflite} (Conv, ReLU, Tanh) and OUT/partial/{uvdoc.onnx,
# uvdoc_static.tflite} (a CUMSUM added, which XNNPACK has no kernel for: the whole-graph check's test).
# tflite: NHWC I/O, fp16 weights (float16 post-training quantisation), fp32 compute.
#   python -I make_test_nets.py OUT     (TensorFlow 2.19.1 + onnx, e.g. C:/Users/User/Android/litert_venv)
# The test nets of docs/results/litert/runtime.txt; litert_check MODEL_DIR = OUT/full or OUT/partial, NET uvdoc.
import os, sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import tensorflow as tf

out = sys.argv[1]
rng = np.random.default_rng(0)
w1 = (rng.standard_normal((8, 3, 16, 16)) * 0.05).astype(np.float32)
b1 = (rng.standard_normal(8) * 0.1).astype(np.float32)
w2 = (rng.standard_normal((2, 8, 3, 3)) * 0.2).astype(np.float32)
b2 = (rng.standard_normal(2) * 0.1).astype(np.float32)

def onnx_model(partial):
    nodes = [
        helper.make_node("Conv", ["image", "w1", "b1"], ["c1"], kernel_shape=[16, 16], strides=[16, 16], pads=[4, 4, 4, 4]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2" if partial else "t"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    ]
    inits = [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(b1, "b1"), numpy_helper.from_array(w2, "w2"), numpy_helper.from_array(b2, "b2")]
    if partial:
        nodes.append(helper.make_node("CumSum", ["c2", "axis"], ["t"]))
        inits.append(numpy_helper.from_array(np.array(3, dtype=np.int64), "axis"))
    nodes.append(helper.make_node("Tanh", ["t"], ["grid"]))
    g = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 712, 488])],
                          [helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 45, 31])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

def tflite_model(partial):
    k1 = tf.constant(w1.transpose(2, 3, 1, 0))   # OIHW -> HWIO
    k2 = tf.constant(w2.transpose(2, 3, 1, 0))

    @tf.function(input_signature=[tf.TensorSpec([1, 712, 488, 3], tf.float32, name="image")])
    def f(x):
        x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]])
        x = tf.nn.relu(tf.nn.conv2d(x, k1, 16, "VALID") + b1)
        x = tf.nn.conv2d(x, k2, 1, "SAME") + b2
        if partial:
            x = tf.math.cumsum(x, axis=2)   # NCHW axis 3 (width) = NHWC axis 2
        return tf.math.tanh(x)

    c = tf.lite.TFLiteConverter.from_concrete_functions([f.get_concrete_function()], f)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
    return c.convert()

for partial in (False, True):
    d = os.path.join(out, "partial" if partial else "full")
    os.makedirs(d, exist_ok=True)
    onnx.save(onnx_model(partial), os.path.join(d, "uvdoc.onnx"))
    with open(os.path.join(d, "uvdoc_static.tflite"), "wb") as fh:
        fh.write(tflite_model(partial))
    i = tf.lite.Interpreter(model_path=os.path.join(d, "uvdoc_static.tflite"))
    print(d, [x["shape"].tolist() for x in i.get_input_details()], [x["shape"].tolist() for x in i.get_output_details()],
          sorted({o["op_name"] for o in i._get_ops_details()}))
