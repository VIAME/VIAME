# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Export shared SAM2/SAM3 mask post-processing without model weights.

Run: python plugins/onnx/export_sam_postprocess.py --check
Requires onnx; --check additionally requires numpy, onnxruntime and opencv-python.

Inputs: logits float32 [1, candidates, low_h, low_w], scores float32 [candidates],
original_size / reshaped_size / padded_size int64 [height, width].
Outputs: mask uint8 [original_h, original_w], score float32 scalar.

This graph can run unchanged in browser or desktop ONNX Runtime, or be composed
with a SAM decoder. It keeps encoder embeddings reusable between clicks. The
highest-IoU candidate is selected BEFORE resize; only one full-size mask is made.
Resizing uses SAM's bilinear half-pixel convention, followed by unpadding and a
second resize to the original image, then a strict logits > 0 threshold.
"""
import argparse
from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h
from viame import image_kernels


def build_model():
    def constant(name, dtype, dims, values):
        return h.make_tensor(name, dtype, dims, values)

    initializers = [
        constant("channel_axis", T.INT64, [1], [1]),
        constant("batch_channel", T.INT64, [2], [1, 1]),
        constant("spatial_axes", T.INT64, [2], [2, 3]),
        constant("start", T.INT64, [2], [0, 0]),
        constant("squeeze_axes", T.INT64, [2], [0, 1]),
        constant("threshold", T.FLOAT, [], [0]),
    ]
    nodes = [
        h.make_node("ArgMax", ["scores"], ["best"], axis=0, keepdims=0, select_last_index=0),
        h.make_node("Gather", ["scores", "best"], ["score"], axis=0),
        h.make_node("Gather", ["logits", "best"], ["selected"], axis=1),
        h.make_node("Unsqueeze", ["selected", "channel_axis"], ["selected_4d"]),
        h.make_node("Concat", ["batch_channel", "padded_size"], ["padded_4d"], axis=0),
        h.make_node("Resize", ["selected_4d", "", "", "padded_4d"], ["padded"],
                    mode="linear", coordinate_transformation_mode="half_pixel"),
        h.make_node("Slice", ["padded", "start", "reshaped_size", "spatial_axes"], ["cropped"]),
        h.make_node("Concat", ["batch_channel", "original_size"], ["original_4d"], axis=0),
        h.make_node("Resize", ["cropped", "", "", "original_4d"], ["resized"],
                    mode="linear", coordinate_transformation_mode="half_pixel"),
        h.make_node("Greater", ["resized", "threshold"], ["binary"]),
        h.make_node("Cast", ["binary"], ["bytes"], to=T.UINT8),
        h.make_node("Squeeze", ["bytes", "squeeze_axes"], ["mask"]),
    ]
    inputs = [h.make_tensor_value_info("logits", T.FLOAT, [1, "candidates", "low_h", "low_w"]),
              h.make_tensor_value_info("scores", T.FLOAT, ["candidates"])]
    inputs += [h.make_tensor_value_info(name, T.INT64, [2]) for name in
               ("original_size", "reshaped_size", "padded_size")]
    outputs = [h.make_tensor_value_info("mask", T.UINT8, ["original_h", "original_w"]),
               h.make_tensor_value_info("score", T.FLOAT, [])]
    graph = h.make_graph(nodes, "sam_mask_postprocess", inputs, outputs, initializers)
    model = h.make_model(graph, producer_name="VIAME", opset_imports=[h.make_opsetid("", 18)], ir_version=9)
    model.doc_string = __doc__
    onnx.checker.check_model(model, full_check=True)
    return model


def check(model):
    import cv2
    from viame import image_kernels
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(19)
    for candidates in (1, 3):
        for original, reshaped, padded in (((21, 37), (32, 32), (32, 32)),
                                            ((17, 29), (18, 32), (32, 32)),
                                            ((1, 31), (1, 32), (32, 32))):
            logits = rng.normal(size=(1, candidates, 8, 8)).astype(np.float32)
            scores = rng.random(candidates).astype(np.float32)
            feeds = {"logits": logits, "scores": scores}
            feeds.update({name: np.asarray(size, dtype=np.int64) for name, size in
                          (("original_size", original), ("reshaped_size", reshaped), ("padded_size", padded))})
            mask, score = session.run(None, feeds)
            best = int(scores.argmax())
            ref = image_kernels.resize(logits[0, best], padded[1], padded[0])
            ref = ref[:reshaped[0], :reshaped[1]]
            ref = image_kernels.resize(ref, original[1], original[0]) > 0
            np.testing.assert_array_equal(mask, ref)
            np.testing.assert_equal(score, scores[best])
    print("ONNX Runtime matches reference selection, resize, crop and threshold (6 cases)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("sam_postprocess.onnx"))
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
