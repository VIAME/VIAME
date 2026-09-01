# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from pathlib import Path


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
            case ".zip":  # netharn / bioharn mmdet backend
                print("Detected netharn model.")
                from viame.pytorch.netharn_mmdet_to_onnx import netharn_mmdet_to_onnx

                # The exporter reads the training window size out of the zip's
                # train_info.json and writes a .modelspec.json sidecar next to
                # the graph, which is what the "onnx" detector reads.
                onnx_file = Path(onnx_model_prefix).with_suffix(".onnx")
                netharn_mmdet_to_onnx(str(model_path), str(onnx_file))
                print(f'The generated onnx model was written to: {onnx_file}')
            case _:
                raise ValueError(f"The model {model_path} is not currently supported; the darknet, yolo-mit and netharn backends are.")

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
