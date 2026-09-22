# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Export sparse desktop-style disparity sampling as a weight-free ONNX graph.

python plugins/onnx/export_stereo_sampler.py --check
Requires onnx; --check also requires numpy and onnxruntime.

Inputs: disparity float32 [H,W] (cached model-resolution disparity),
points float32 [N,2] (x,y in original rectified-image pixels),
source_size float32 [2] (original height,width).
Outputs: disparities float32 [N] (original-image pixels; zero means invalid),
fractions float32 [N] (valid fraction of clipped neighbourhood).

Each point gathers only a 7x7 native-pixel neighbourhood. Bilinear sampling
emulates OpenCV's half-pixel resize without upscaling the full disparity image.
The graph selects desktop's 90th-percentile order statistic among finite positive
values. It does not rerun the stereo model or create a full-frame window tensor.
The same graph works in desktop ONNX Runtime and browser WASM.
"""
import argparse
from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h


def build_model():
    nodes, constants = [], []

    def const(name, values, dtype=T.FLOAT, dims=None):
        dims = [] if dims is None else dims
        constants.append(h.make_tensor(name, dtype, dims, values))
        return name

    def op(kind, *inputs, **attrs):
        name = f"{kind.lower()}_{len(nodes)}"
        nodes.append(h.make_node(kind, list(inputs), [name], **attrs))
        return name

    zero = const("zero", [0])
    one = const("one", [1])
    half = const("half", [.5])
    percentile = const("percentile", [.9])
    infinity = const("infinity", [float('inf')])
    i0 = const("i0", [0], T.INT64)
    i1 = const("i1", [1], T.INT64)
    axis1 = const("axis1", [1], T.INT64, [1])
    flat = const("flat", [-1], T.INT64, [1])
    k49 = const("k49", [49], T.INT64, [1])
    dx = const("dx", [x for y in range(-3, 4) for x in range(-3, 4)], dims=[1, 49])
    dy = const("dy", [y for y in range(-3, 4) for x in range(-3, 4)], dims=[1, 49])
    f32 = lambda x: op("Cast", x, to=T.FLOAT)
    i64 = lambda x: op("Cast", x, to=T.INT64)
    finite = lambda x: op("Not", op("Or", op("IsNaN", x), op("IsInf", x)))
    clamp = lambda x, maximum: op("Min", op("Max", x, zero), maximum)
    within = lambda x, maximum: op("And", op("GreaterOrEqual", x, zero), op("Less", x, maximum))

    shape = f32(op("Shape", "disparity"))
    width = op("Gather", shape, i1, axis=0)
    height = op("Gather", shape, i0, axis=0)
    source_w = op("Gather", "source_size", i1, axis=0)
    source_h = op("Gather", "source_size", i0, axis=0)
    scale_x = op("Div", width, source_w)
    scale_y = op("Div", height, source_h)
    x = op("Unsqueeze", op("Gather", "points", i0, axis=1), axis1)
    y = op("Unsqueeze", op("Gather", "points", i1, axis=1), axis1)
    finite_points = op("And", finite(x), finite(y))
    # Cast truncates toward zero, matching DenseStereoGrid's int(point + .5).
    cx = f32(i64(op("Add", op("Where", finite_points, x, zero), half)))
    cy = f32(i64(op("Add", op("Where", finite_points, y, zero), half)))
    centers_ok = op("And", finite_points, op("And", within(cx, source_w), within(cy, source_h)))
    px, py = op("Add", cx, dx), op("Add", cy, dy)
    in_bounds = op("And", within(px, source_w), within(py, source_h))
    last_x, last_y = op("Sub", width, one), op("Sub", height, one)
    gx = clamp(op("Sub", op("Mul", op("Add", px, half), scale_x), half), last_x)
    gy = clamp(op("Sub", op("Mul", op("Add", py, half), scale_y), half), last_y)
    ix, iy = op("Floor", gx), op("Floor", gy)
    jx, jy = op("Min", op("Add", ix, one), last_x), op("Min", op("Add", iy, one), last_y)
    fx, fy = op("Sub", gx, ix), op("Sub", gy, iy)
    grid = op("Reshape", "disparity", flat)

    def pixel(x, y):
        return op("Gather", grid, i64(op("Add", op("Mul", y, width), x)), axis=0)

    a, b, c, d = pixel(ix, iy), pixel(jx, iy), pixel(ix, jy), pixel(jx, jy)
    top = op("Add", op("Mul", op("Sub", one, fx), a), op("Mul", fx, b))
    bottom = op("Add", op("Mul", op("Sub", one, fx), c), op("Mul", fx, d))
    value = op("Div", op("Add", op("Mul", op("Sub", one, fy), top), op("Mul", fy, bottom)), scale_x)
    valid = op("And", in_bounds, op("And", finite(value), op("Greater", value, zero)))
    count = op("ReduceSum", f32(valid), axis1, keepdims=1)
    considered = op("ReduceSum", f32(in_bounds), axis1, keepdims=1)
    sortable = op("Where", valid, value, infinity)
    nodes.append(h.make_node("TopK", [sortable, k49], ["sorted", "indices"], axis=1, largest=0, sorted=1))
    index = i64(op("Floor", op("Mul", count, percentile)))
    selected = op("GatherElements", "sorted", index, axis=1)
    accepted = op("And", centers_ok, op("Greater", count, zero))
    values = op("Where", accepted, selected, zero)
    fractions = op("Where", accepted, op("Div", count, op("Max", considered, one)), zero)
    nodes += [h.make_node("Squeeze", [values, axis1], ["disparities"]),
              h.make_node("Squeeze", [fractions, axis1], ["fractions"])]
    inputs = [h.make_tensor_value_info("disparity", T.FLOAT, ["height", "width"]),
              h.make_tensor_value_info("points", T.FLOAT, ["points", 2]),
              h.make_tensor_value_info("source_size", T.FLOAT, [2])]
    outputs = [h.make_tensor_value_info(name, T.FLOAT, ["points"]) for name in ("disparities", "fractions")]
    graph = h.make_graph(nodes, "sparse_stereo_sampler", inputs, outputs, constants)
    model = h.make_model(graph, producer_name="VIAME", opset_imports=[h.make_opsetid("", 18)], ir_version=9)
    model.doc_string = __doc__
    onnx.checker.check_model(model, full_check=True)
    return model


def check(model):
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(93)
    checked = 0
    for height, width, source_h, source_w in ((7, 7, 7, 7), (11, 19, 80, 120), (1, 1, 4, 8)):
        grid = rng.uniform(0, 40, (height, width)).astype(np.float32)
        grid[rng.random(grid.shape) < .3] = 0
        points = np.array([[0, 0], [source_w - 1, source_h - 1], [-2, 0], [np.nan, 1], [np.inf, 0],
                           [-.6, .4], [source_w, 0], [source_w / 2, source_h / 2]], np.float32)
        ds, fractions = session.run(None, {"disparity": grid, "points": points,
                                           "source_size": np.array([source_h, source_w], np.float32)})
        for i, (x, y) in enumerate(points):
            expected, fraction = 0, 0
            if np.isfinite(x) and np.isfinite(y):
                cx, cy = int(float(x) + .5), int(float(y) + .5)
                if 0 <= cx < source_w and 0 <= cy < source_h:
                    samples, total = [], 0
                    sx, sy = width / source_w, height / source_h
                    for py in range(max(0, cy - 3), min(source_h, cy + 4)):
                        for px in range(max(0, cx - 3), min(source_w, cx + 4)):
                            gx = min(width - 1, max(0, (px + .5) * sx - .5))
                            gy = min(height - 1, max(0, (py + .5) * sy - .5))
                            ix, iy = int(gx), int(gy)
                            jx, jy = min(width - 1, ix + 1), min(height - 1, iy + 1)
                            fx, fy = gx - ix, gy - iy
                            v = ((1 - fy) * ((1 - fx) * grid[iy, ix] + fx * grid[iy, jx])
                                 + fy * ((1 - fx) * grid[jy, ix] + fx * grid[jy, jx])) / sx
                            total += 1
                            if np.isfinite(v) and v > 0:
                                samples.append(v)
                    if samples:
                        expected = sorted(samples)[int(len(samples) * .9)]
                        fraction = len(samples) / total
            np.testing.assert_allclose(ds[i], expected, rtol=2e-6, atol=1e-5)
            np.testing.assert_allclose(fractions[i], fraction, atol=1e-7)
            checked += 1
    print(f"ONNX Runtime matches sparse reference ({checked} cases)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("stereo_sample.onnx"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    model = build_model()
    if args.check:
        check(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
