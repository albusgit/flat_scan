"""Static-shape copies of the app's nets for the NPU / NNAPI execution providers (docs/accel/PLAN.md, Stream A).

  python tools/accel/prepare_models.py [MODELS_DIR] [OUT_DIR]
      MODELS_DIR default app/src/main/assets/models; OUT_DIR default MODELS_DIR
      -> uvdoc_static.onnx, syntheticdoc_hr_static.onnx (the originals are not touched)

What changes (make_static):
  * every graph input gets its fixed size; outputs not used by the app are removed (syntheticdoc_hr's
    `shading`) together with the nodes only they needed;
  * the shape arithmetic is folded: a Shape of a tensor whose size is now known becomes a constant, and so
    does every node whose inputs are all constants and whose outputs are integer tensors (the Gather /
    Concat / Slice / Unsqueeze chains that feed Resize sizes, Reshape, Pad and Expand). So no node sees a
    shape it can only learn at run time (NNAPI and QNN need fixed shapes; the mobile usability checker
    counts such nodes as "dynamic shape");
  * floating-point nodes are never evaluated here: the fp16 weights keep their Cast -> fp32 nodes (ONNX
    Runtime folds those at load time, as for the originals), so the files stay half size and the
    arithmetic the CPU runs is the original graph's. This is the "onnxsim or equivalent" of the plan:
    onnxsim's folding would also turn the fp16 weights into fp32 initializers (2x the file size) and runs
    graph rewrites (BatchNorm fusion) of its own;
  * an Expand to a now constant shape (ShadeNet's global-mean branch) becomes a Mul by a ones tensor of
    the target size: the same broadcast, exact (x * 1.0f == x), and an op NNAPI has;
  * a Resize given its output `sizes` gets empty tensors for its unused roi / scales inputs instead of ''
    (the usability checker counts '' as a dynamic input and splits the partition there);
  * every intermediate tensor gets its static shape in value_info.
The IR version and opset stay those of the original file. tools/accel/compare_outputs.py checks that the
outputs are bit-identical to the originals on the CPU. Also used by tools/shadow_net/export.py --static."""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INT_TYPES = {TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8, TensorProto.UINT8, TensorProto.BOOL}
# ops whose outputs are shapes or shape-sized index tensors; the only ones folded besides Shape
FOLDABLE = {'Gather', 'Concat', 'Slice', 'Unsqueeze', 'Squeeze', 'Cast', 'Add', 'Sub', 'Mul', 'Div', 'Reshape',
            'ConstantOfShape', 'Range', 'Equal', 'Where', 'Identity', 'Neg', 'Transpose', 'Expand', 'Floor', 'Ceil',
            'Max', 'Min', 'ReduceProd', 'Mod', 'Abs', 'Not', 'And', 'Or', 'Greater', 'Less'}
SHAPE_INPUTS = {'Resize': (1, 2, 3), 'Reshape': (1,), 'Expand': (1,), 'Pad': (1, 2), 'Slice': (1, 2, 3, 4),
                'ConstantOfShape': (0,), 'Tile': (1,), 'Upsample': (1,), 'Unsqueeze': (1,), 'Squeeze': (1,)}


def _dims(vi):
    t = vi.type.tensor_type
    if not t.HasField('shape'):
        return None
    d = [x.dim_value if x.HasField('dim_value') else None for x in t.shape.dim]
    return None if any(v is None for v in d) else d


def _set_dims(vi, dims):
    t = vi.type.tensor_type
    t.ClearField('shape')
    for v in dims:
        t.shape.dim.add().dim_value = int(v)


def _infer(m):
    return shape_inference.infer_shapes(m, check_type=True, strict_mode=True, data_prop=True)


def _value_infos(m):
    g = m.graph
    return {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}


def _constants(g):
    c = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == 'Constant' and len(n.attribute) == 1 and n.attribute[0].name == 'value':
            c[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return c


def _eval_node(node, inputs):
    """One node on constant inputs (integer shape arithmetic only), with the ONNX reference evaluator."""
    from onnx.reference import ReferenceEvaluator
    names = [f'i{k}' for k in range(len(node.input))]
    n2 = helper.make_node(node.op_type, [names[k] if node.input[k] else '' for k in range(len(node.input))],
                          [f'o{k}' for k in range(len(node.output))], domain=node.domain)
    n2.attribute.extend(node.attribute)
    ins = [helper.make_tensor_value_info(names[k], helper.np_dtype_to_tensor_dtype(inputs[k].dtype), inputs[k].shape)
           for k in range(len(node.input)) if node.input[k]]
    outs = [helper.make_tensor_value_info(f'o{k}', TensorProto.UNDEFINED, None) for k in range(len(node.output))]
    gm = helper.make_model(helper.make_graph([n2], 'fold', ins, outs), opset_imports=[helper.make_opsetid('', 17)])
    feeds = {names[k]: inputs[k] for k in range(len(node.input)) if node.input[k]}
    return ReferenceEvaluator(gm).run(None, feeds)


def _prune(m):
    """Drop nodes no graph output needs, and initializers no node uses."""
    g = m.graph
    need = {o.name for o in g.output}
    keep = []
    for n in reversed(list(g.node)):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(i for i in n.input if i)
    keep.reverse()
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    inits = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(inits)
    vis = [v for v in g.value_info if v.name in used]
    del g.value_info[:]
    g.value_info.extend(vis)


def make_static(m, input_dims, keep_outputs=None, log=print, expand_as_mul=True, resize_inputs=True):
    """m: onnx.ModelProto (modified and returned). input_dims: {input name: [dims]}; keep_outputs: names.
    expand_as_mul: an Expand to a constant shape becomes a Mul by ones (bit-identical; NNAPI has no Expand).
    resize_inputs: a sized Resize's '' roi / scales become empty tensors (bit-identical; see below)."""
    g = m.graph
    for i in g.input:
        if i.name in input_dims:
            _set_dims(i, input_dims[i.name])
        if _dims(i) is None:
            raise ValueError(f'input {i.name}: no fixed size given')
    if keep_outputs is not None:
        outs = [o for o in g.output if o.name in keep_outputs]
        if len(outs) != len(keep_outputs):
            raise ValueError(f'outputs {keep_outputs} not all in the model')
        dropped = [o.name for o in g.output if o.name not in keep_outputs]
        del g.output[:]
        g.output.extend(outs)
        if dropped:
            log(f'  removed outputs {dropped}')
    for o in g.output:   # the exporter's symbolic output sizes go; inference sets the real ones
        o.type.tensor_type.ClearField('shape')
    del g.value_info[:]
    n0 = len(g.node)
    _prune(m)
    log(f'  nodes {n0} -> {len(g.node)} after removing unused outputs')
    folded = 0
    while True:
        mi = _infer(m)
        vi = _value_infos(mi)
        types = {k: v.type.tensor_type.elem_type for k, v in vi.items()}
        consts = _constants(g)
        new_nodes, changed = [], False
        for n in g.node:
            vals = None
            if n.op_type == 'Shape' and _dims(vi[n.input[0]]) is not None:
                d = _dims(vi[n.input[0]])
                at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
                vals = [np.array(d[at.get('start', 0):at.get('end', len(d))], np.int64)]
            elif (n.op_type in FOLDABLE and n.domain in ('', 'ai.onnx') and all(i == '' or i in consts for i in n.input)
                  and all(types.get(o) in INT_TYPES for o in n.output)):
                vals = _eval_node(n, [consts.get(i) for i in n.input])
            if vals is None:
                new_nodes.append(n)
                continue
            for name, v in zip(n.output, vals):
                new_nodes.append(helper.make_node('Constant', [], [name], value=numpy_helper.from_array(np.asarray(v), name)))
            folded += 1
            changed = True
        del g.node[:]
        g.node.extend(new_nodes)
        _prune(m)
        if not changed:
            break
    if expand_as_mul:
        # Expand (ShadeNet's global-mean branch, expand_as) is not an NNAPI op: x * ones(target) is the
        # same broadcast, and x * 1.0f == x exactly, so the CPU output does not change
        mi = _infer(m)
        vi = _value_infos(mi)
        consts = _constants(g)
        nodes = []
        for n in g.node:
            if n.op_type == 'Expand' and n.input[1] in consts and _dims(vi[n.input[0]]) is not None:
                d, s = _dims(vi[n.input[0]]), [int(v) for v in consts[n.input[1]]]
                if len(d) == len(s) and all(a == b or a == 1 for a, b in zip(d, s)):
                    ones = numpy_helper.from_array(np.ones([1 if a == b else b for a, b in zip(d, s)], np.float32),
                                                   n.output[0] + '_ones')
                    g.initializer.append(ones)
                    nodes.append(helper.make_node('Mul', [n.input[0], ones.name], [n.output[0]], name=(n.name or 'Expand') + '_as_mul'))
                    log(f'  Expand {n.name} {d} -> {s} as Mul by ones{list(ones.dims)}')
                    continue
            nodes.append(n)
        del g.node[:]
        g.node.extend(nodes)
        _prune(m)
    if resize_inputs:
        # a Resize given `sizes` has '' for roi and scales; the usability checker counts an empty input name
        # as a dynamic shape (the node then splits the partition). Empty float tensors are the opset 11
        # spelling of the same thing: the CPU kernel takes sizes when scales has no element (checked
        # bit-identical by compare_outputs.py). Not scales = sizes / input: 90 -> 121 (syntheticdoc_hr)
        # is 1.3444444f, and 1.3444444f * 90 = 120.99999f gives 120 rows.
        consts = _constants(g)
        empty = {}
        for n in g.node:
            if n.op_type == 'Resize' and len(n.input) == 4 and n.input[3] in consts:
                for k, tag in ((1, 'roi'), (2, 'scales')):
                    if n.input[k] == '':
                        if tag not in empty:
                            empty[tag] = numpy_helper.from_array(np.zeros([0], np.float32), f'resize_empty_{tag}')
                            g.initializer.append(empty[tag])
                        n.input[k] = empty[tag].name
                log(f'  Resize {n.name}: roi, scales as empty tensors (sizes {[int(v) for v in consts[n.input[3]]]})')
    # the final sizes into value_info and the outputs
    mi = _infer(m)
    vi = _value_infos(mi)
    for v in mi.graph.value_info:
        if _dims(v) is not None:
            g.value_info.append(v)
    for o in g.output:
        o.CopyFrom(vi[o.name])
    # what is left must be static: no Shape node, every shape input a constant, every tensor sized
    consts = _constants(g)
    left = [n.name or n.output[0] for n in g.node if n.op_type == 'Shape']
    for n in g.node:
        for k in SHAPE_INPUTS.get(n.op_type, ()):
            if k < len(n.input) and n.input[k] and n.input[k] not in consts:
                left.append(f'{n.op_type} {n.name} input {k}')
    unsized = [v.name for v in mi.graph.value_info if _dims(v) is None] + [o.name for o in g.output if _dims(o) is None]
    if left or unsized:
        raise ValueError(f'not static: {left} {unsized[:10]}')
    onnx.checker.check_model(m)
    log(f'  folded {folded} shape nodes; nodes now {len(g.node)}; outputs ' +
        ', '.join(f'{o.name} {_dims(o)}' for o in g.output))
    return m


MODELS = [   # (file, output names the app reads)
    ('uvdoc.onnx', ['grid']),
    ('syntheticdoc_hr.onnx', ['grid']),
]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models')
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    os.makedirs(dst, exist_ok=True)
    for f, outs in MODELS:
        p = os.path.join(src, f)
        m = onnx.load(p)
        dims = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
        print(f, 'input', dims)
        make_static(m, dims, outs)
        q = os.path.join(dst, f[:-5] + '_static.onnx')
        onnx.save(m, q)
        print(f'  -> {q} {os.path.getsize(q)} bytes (original {os.path.getsize(p)})')


if __name__ == '__main__':
    main()
