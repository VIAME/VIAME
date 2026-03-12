#!/usr/bin/env python3.10
"""
yolo-mit_to_onnx - Convert a pytorch lightning yolov9 model to ONNX.
"""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from yolo.model.yolo import create_model
from yolo.utils.logger import logger


def yolomit_to_onnx(model_path: Path, config_path: Path, output_onnx: Path):
    lightning_model = torch.load(model_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" not in lightning_model.keys():
        raise ValueError("The provided model is not a PyTorch Lightning checkpoint.")

    yolo_config_dir = config_path.parent.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(yolo_config_dir)):
        cfg = compose(config_name=config_path.name)

    model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=model_path).eval()
    dummy_input = torch.ones((1, 3, *cfg.image_size))
    onnx_file = output_onnx.with_suffix(".onnx")
    #TODO: reduce ONNX model to use only main branch from yolov9: https://github.com/MultimediaTechLab/YOLO/pull/174
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f":white_check_mark: Success ONNX export to {onnx_file}")
