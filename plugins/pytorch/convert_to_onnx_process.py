# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from pathlib import Path
import warnings

import zipfile
import json
import ast


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
        model_path = Path(self.config_value("model_path"))
        onnx_model_prefix = Path(self.config_value("onnx_model_prefix"))
        batch_size = 1

        # Models conversion
        match model_path.suffix.lower():
            case ".weights":  # darknet backend
                print("Detected darknet model.")
                from darknet2onnx.export import export_darknet_to_onnx
                from darknet2onnx.darknet2pytorch.model import Darknet

                cfg_file = model_path.with_suffix(".cfg")
                onnx_file = onnx_model_prefix.with_suffix(".onnx")

                model = Darknet(cfg_file)
                model.load_weights(model_path)
                export_darknet_to_onnx(model, batch_size, onnx_filepath=onnx_file)

                print(f"The generated onnx model was written to: {onnx_file}")
            case ".ckpt" | ".pth":  # pytorch backend
                print("Detected pytorch model.")
                import yaml
                config_path = model_path.parent / "train_config.yaml"
                if not config_path.exists():
                    raise ValueError("Detected pytorch model without associated configuration!")
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                # if cfg.get('litdet_version'):  #TODO
                    # print("Detected LitDet model!")
                if cfg.get('name') == "viame-mit-yolo-detector":
                    print("Detected yolo-mit lightning model!")
                    output_onnx = Path(onnx_model_prefix).with_suffix(".onnx")
                    from viame.pytorch.yolomit_to_onnx import yolomit_to_onnx
                    yolomit_to_onnx(model_path, config_path, output_onnx)
                else:
                    raise ValueError(f"Detected an invalid YAML configuration at {config_path}")
            case ".zip":  # netharn backend
                print("Detected netharn model.")
                warnings.warn("Exporting a netharn model to ONNX is currently in beta and may be unstable!")
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

                from viame.pytorch.crcnn_to_onnx import crcnn_to_onnx
                crcnn_to_onnx(model_path, net_shape, batch_size, onnx_model_prefix)
                print(f'The generated onnx model was written to: {onnx_model_prefix}.onnx')
            case _:
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
