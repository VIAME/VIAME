# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from pathlib import Path
import warnings

import zipfile
import json
import ast
import sys

from torch import load


class OnnxConverter(KwiverProcess):
    """
    This process convert a yolo-darknet/crcnn-mmdet model to onnx in the
    config step.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.declare_configuration_key("model_path", "", "Path to the trained model (yolo-mit or darknet backend)")
        self.declare_configuration_key("onnx_model_prefix", "", "Output onnx model path prefix")

    # ----------------------------------------------
    def _configure(self):

        # Get config parameters
        model_path = self.config_value("model_path")
        onnx_model_prefix = self.config_value("onnx_model_prefix")
        batch_size = 1

        # Models conversion
        if (model_path.endswith(".weights")):
            from darknet2onnx.export import export_darknet_to_onnx
            from darknet2onnx.darknet2pytorch.model import Darknet

            weights_file = Path(model_path)
            cfg_file = weights_file.with_suffix(".cfg")
            onnx_file = Path(onnx_model_prefix).with_suffix(".onnx")

            model = Darknet(cfg_file)
            model.load_weights(weights_file)
            export_darknet_to_onnx(model, batch_size, onnx_filepath=onnx_file)

            print(f"The generated onnx model was written to: {onnx_file}")
        elif (model_path.endswith(".ckpt") or model_path.endswith(".pth")):
            import yaml
            print("Detected pytorch model!")
            model_path_ = Path(model_path)
            config_path = Path(model_path_.parent / "train_config.yaml")
            if not config_path.exists():
                raise ValueError("Detected pytorch model without associated configuration!")
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            # if cfg.get('litdet_version'):  #TODO
                # print("Detected LitDet model!")
            if cfg.get('name') == "viame-mit-yolo-detector":
                print("Detected yolo-mit lightning model!")
                model_path = Path(model_path)
                config_path = model_path.parent / "train_config.yaml"
                output_onnx = Path(onnx_model_prefix).with_suffix(".onnx")
                from viame.pytorch.onnx_exporters.yolomit_to_onnx import yolomit_to_onnx
                yolomit_to_onnx(model_path, config_path, output_onnx)
            else:
                warnings.warn("Detected a pytorch YAML configuration that is not valid.")

        elif (model_path.endswith(".zip")):
            print("Detected netharn model, export may fail!")
            net_shape = ()
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                all_files_in_zip = zip_ref.namelist()
                train_info_files = [file for file in all_files_in_zip if file.endswith("train_info.json")]
                if len(train_info_files) > 1:
                    raise ValueError(f"There should be only one JSON confugration, detected {len(train_info_files)}")
                with zip_ref.open(train_info_files[0]) as file:
                    content = file.read()
                    json_content = json.loads(content.decode('utf-8'))
                    config = ast.literal_eval(json_content["extra"]["config"])
                    net_shape = (config['window_dims'][0], config['window_dims'][1], 3)

            from viame.pytorch.onnx_exporters.crcnn_to_onnx import crcnn_to_onnx
            crcnn_to_onnx(model_path, net_shape, batch_size, onnx_model_prefix)
            print(f'The generated onnx model was written to: {onnx_model_prefix}.onnx')

        else:
            raise ValueError(f"The model {model_path} is not curently supported, only darknet and yolo-mit backends are supported!")

        self._base_configure()
        self.mark_process_as_complete()

    # ----------------------------------------------
    def _step(self):
        self._base_step()


# ==================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = "python:viame.pytorch.convert_to_onnx_process"

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process("convert_to_onnx", "Convert a VIAME model to onnx", OnnxConverter)

    process_factory.mark_process_module_as_loaded(module_name)
