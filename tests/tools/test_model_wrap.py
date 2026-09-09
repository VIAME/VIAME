"""model_wrap classifies bare model files and wraps them in a detector pipeline.

The checkpoints here are hand-built torch-style archives: a zip holding a
data.pkl whose pickle carries the strings a real checkpoint would, and no
tensor data. Nothing imports a deep learning framework.
"""
import io
import os
import pickle
import pickletools
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

import model_wrap  # noqa: E402


def torch_archive(path, obj=None, raw_pickle=None):
    data = raw_pickle if raw_pickle is not None else pickle.dumps(obj, protocol=2)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", data)
        zf.writestr("archive/version", "3\n")
    return path


def global_pickle(module, name):
    """A dict {'model': <module.name>} pickle; pickletools reads the global
    reference without importing it."""
    return (
        b"\x80\x02}q\x00(X\x05\x00\x00\x00modelq\x01c"
        + module.encode() + b"\n" + name.encode() + b"\nq\x02u."
    )


@pytest.fixture
def templates(tmp_path):
    pipelines = tmp_path / "pipelines"
    (pipelines / "templates").mkdir(parents=True)
    (pipelines / "templates" / "detector_default.pipe").write_text(
        "config _scheduler\n"
        "  :type pythread_per_process\n\n"
        "include $ENV{VIAME_INSTALL}/configs/pipelines/common_default_input_with_downsampler.pipe\n\n"
        "process detector_input\n"
        "  :: image_filter\n\n"
        "connect from downsampler.output_1\n"
        "        to   detector_input.image\n\n"
        "process detector1\n"
        "  :: image_object_detector\n"
        "  [-DETECTOR-IMPL-]\n\n"
        "connect from detector_input.image\n"
        "        to   detector1.image\n\n"
        "process detector_output\n"
        "  :: refine_detections\n\n"
        "process detector_writer\n"
        "  :: detected_object_output\n"
        "  :file_name computed_detections.csv\n\n"
        "connect from detector_output.detected_object_set\n"
        "        to   detector_writer.detected_object_set\n"
    )
    (pipelines / "templates" / "detector_onnx.pipe").write_text(
        "process detector1\n"
        "  :: image_object_detector\n"
        "  :detector:type onnx\n"
        "  block detector:onnx\n"
        "    :model                                     [-MODEL-]\n"
        "  endblock\n"
    )
    (pipelines / "templates" / "detector_netharn_clfr.pipe").write_text(
        "include $ENV{VIAME_INSTALL}/configs/pipelines/common_default_input_with_downsampler.pipe\n\n"
        "process classifier1\n"
        "  :: image_object_detector\n"
        "  :detector:type                               netharn_classifier\n\n"
        "  block detector:netharn_classifier\n"
        "    :mode                                      frame_classifier\n"
        "    relativepath deployed =                    [-DEPLOYED-]\n"
        "  endblock\n\n"
        "connect from detector_input.image\n"
        "        to   classifier1.image\n"
    )
    return pipelines


def netharn_zip(path, model_class, topology="MM_CascadeRCNN_2810fc.py"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("deploy_x/train_info.json",
                    '{"hyper": {"model": ["%s", {}]}}' % model_class)
        zf.writestr("deploy_x/deploy_snapshot.pt", "")
        zf.writestr("deploy_x/" + topology, "")
    return path


class TestIdentifyCheckpoints:
    def test_rf_detr_deployed(self, tmp_path):
        path = torch_archive(tmp_path / "det.pth", {
            "model": {"transformer.decoder.layers.0.w": 0, "class_embed.weight": 0},
            "args": {"resolution": 704},
        })
        info = model_wrap.identify(str(path))
        assert info.runnable
        assert info.impl == "rf_detr"
        assert info.keys == {"weight": str(path)}

    def test_rf_detr_lightning(self, tmp_path):
        path = torch_archive(tmp_path / "last.ckpt", {
            "pytorch-lightning_version": "2.6.5",
            "state_dict": {"model.transformer.decoder.layers.0.w": 0,
                           "model.class_embed.weight": 0},
        })
        assert model_wrap.identify(str(path)).impl == "rf_detr"

    def test_ultralytics(self, tmp_path):
        path = torch_archive(
            tmp_path / "yolo.pt",
            raw_pickle=global_pickle("ultralytics.nn.tasks", "DetectionModel"))
        info = model_wrap.identify(str(path))
        assert info.impl == "ultralytics"
        assert info.mode == "original_and_resized"

    def test_litdet(self, tmp_path):
        path = torch_archive(
            tmp_path / "det.ckpt",
            raw_pickle=global_pickle(
                "lightning_hydra_detection.tasks.detect_module", "DetectLitModule"))
        info = model_wrap.identify(str(path))
        assert info.impl == "litdet"
        assert info.keys == {"checkpoint": str(path)}

    def test_mit_yolo_needs_train_config(self, tmp_path):
        path = torch_archive(
            tmp_path / "best.ckpt",
            raw_pickle=global_pickle("yolo.config.config", "ModelConfig"))
        info = model_wrap.identify(str(path))
        assert not info.runnable
        assert "train_config.yaml" in info.reason

        (tmp_path / "train_config.yaml").write_text("model:\n  name: v9-c\n")
        info = model_wrap.identify(str(path))
        assert info.runnable
        assert info.impl == "mit_yolo"
        assert info.keys["model"] == "v9-c"

    def test_netharn_snapshot_is_not_runnable(self, tmp_path):
        path = torch_archive(tmp_path / "best_snapshot.pt", {
            "epoch": 3, "model_state_dict": {"layer.w": 0},
        })
        info = model_wrap.identify(str(path))
        assert not info.runnable
        assert "deployed" in info.reason

    def test_mmdet_needs_companions(self, tmp_path):
        path = torch_archive(tmp_path / "trained.pth", {
            "meta": {"CLASSES": ("fish",)}, "state_dict": {"backbone.w": 0},
        })
        assert not model_wrap.identify(str(path)).runnable

        (tmp_path / "trained.py").write_text("model = dict()\n")
        (tmp_path / "trained.lbl").write_text("fish\n")
        info = model_wrap.identify(str(path))
        assert info.impl == "mmdet"
        assert info.keys["net_config"].endswith("trained.py")
        assert info.keys["class_names"].endswith("trained.lbl")

    def test_detectron2_needs_its_yaml(self, tmp_path):
        path = torch_archive(tmp_path / "model_final.pth", {
            "model": {"roi_heads.w": 0}, "iteration": 10,
        })
        info = model_wrap.identify(str(path))
        assert not info.runnable
        assert "config" in info.reason

        (tmp_path / "model_config.yaml").write_text("MODEL: {}\n")
        info = model_wrap.identify(str(path))
        assert info.impl == "detectron2"
        assert info.keys == {"checkpoint_fpath": str(path),
                             "cfg": str(tmp_path / "model_config.yaml")}

    def test_darknet_weights_with_companions(self, tmp_path):
        path = tmp_path / "seal.weights"
        path.write_bytes(b"\x00" * 16)
        assert not model_wrap.identify(str(path)).runnable

        (tmp_path / "seal.cfg").write_text("[net]\n")
        (tmp_path / "seal.lbl").write_text("seal\n")
        info = model_wrap.identify(str(path))
        assert info.impl == "darknet"
        assert not info.windowed
        assert info.keys == {"net_config": str(tmp_path / "seal.cfg"),
                             "weight_file": str(path),
                             "class_names": str(tmp_path / "seal.lbl")}

    def test_companion_by_stem_wins_over_others(self, tmp_path):
        path = torch_archive(tmp_path / "trained.pth", {
            "meta": {}, "state_dict": {"backbone.w": 0},
        })
        (tmp_path / "trained.py").write_text("")
        (tmp_path / "other.py").write_text("")
        (tmp_path / "trained.lbl").write_text("fish\n")
        info = model_wrap.identify(str(path))
        assert info.keys["net_config"] == str(tmp_path / "trained.py")

    def test_unknown_contents(self, tmp_path):
        path = torch_archive(tmp_path / "resnet.pt", {"conv1.weight": 0})
        assert not model_wrap.identify(str(path)).runnable

    def test_garbage_file(self, tmp_path):
        path = tmp_path / "junk.pt"
        path.write_bytes(b"not a checkpoint")
        assert not model_wrap.identify(str(path)).runnable


class TestIdentifyZips:
    def test_netharn_deployed_detector(self, tmp_path):
        path = netharn_zip(tmp_path / "trained_detector.zip",
                           "bioharn.models.mm_models.MM_CascadeRCNN")
        info = model_wrap.identify(str(path))
        assert info.impl == "netharn"
        assert not info.classifier
        assert info.keys == {"deployed": str(path)}

    def test_netharn_deployed_classifier(self, tmp_path):
        path = netharn_zip(tmp_path / "trained_classifier.zip",
                           "viame.pytorch.netharn.clf_fit.ClfModel", "ClfModel_ab12.py")
        info = model_wrap.identify(str(path))
        assert info.impl == "netharn_classifier"
        assert info.classifier
        assert "full-frame classifier" in info.describe()

    def test_netharn_classifier_by_topology_name_alone(self, tmp_path):
        path = tmp_path / "clf.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("deploy_x/train_info.json", "{}")
            zf.writestr("deploy_x/deploy_snapshot.pt", "")
            zf.writestr("deploy_x/ClfModel_ab12.py", "")
        assert model_wrap.identify(str(path)).classifier

    def test_onnx_detector_package(self, tmp_path):
        path = tmp_path / "fish_deim.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("pkg/model.onnx", "")
            zf.writestr("pkg/model.modelspec.json",
                        '{"postprocess": {"decoder": "detr"}}')
        info = model_wrap.identify(str(path))
        assert info.impl == "onnx"
        assert not info.classifier

    def test_onnx_classifier_package(self, tmp_path):
        path = tmp_path / "clf.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("pkg/model.onnx", "")
            zf.writestr("pkg/model.modelspec.json",
                        '{"postprocess": {"decoder": "classifier"}, '
                        '"meta": {"task": "classification"}}')
        info = model_wrap.identify(str(path))
        assert info.impl == "onnx_classifier"
        assert info.classifier

    def test_bare_onnx_reads_its_sidecar(self, tmp_path):
        path = tmp_path / "boulder.onnx"
        path.write_bytes(b"")
        assert model_wrap.identify(str(path)).impl == "onnx"

        (tmp_path / "boulder.modelspec.json").write_text('{"meta": {"task": "classification"}}')
        info = model_wrap.identify(str(path))
        assert info.impl == "onnx_classifier"
        assert info.keys == {"model": str(path)}

    def test_packaged_pipeline(self, tmp_path):
        path = tmp_path / "trained.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("detector.pipe", "")
            zf.writestr("trained_detector.pth", "")
        info = model_wrap.identify(str(path))
        assert info.kind == "pipeline_zip"
        assert info.pipes == ["detector.pipe"]
        assert info.pipe_in_zip == "detector.pipe"

    def test_several_pipelines_are_all_listed(self, tmp_path):
        path = tmp_path / "addon.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("configs/pipelines/tracker_x.pipe", "")
            zf.writestr("configs/pipelines/detector_x.pipe", "")
            zf.writestr("configs/pipelines/models/x.pth", "")
        info = model_wrap.identify(str(path))
        assert info.runnable
        assert info.pipes == [
            "configs/pipelines/detector_x.pipe", "configs/pipelines/tracker_x.pipe"]
        assert info.pipe_in_zip == ""
        assert "2 pipelines" in info.describe()

    def test_bundled_darknet(self, tmp_path):
        path = tmp_path / "seal.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("seal/seal.weights", b"\x00" * 16)
            zf.writestr("seal/seal.cfg", "[net]\n")
            zf.writestr("seal/seal.names", "seal\n")
        work = tmp_path / "work"
        work.mkdir()
        info = model_wrap.identify(str(path), str(work))
        assert info.runnable
        assert info.impl == "darknet"
        assert info.kind == "zip of Darknet weights"
        assert info.describe().startswith("seal.zip (seal.weights): zip of Darknet weights")
        assert info.keys["weight_file"] == str(work / "seal" / "seal" / "seal.weights")
        assert info.keys["class_names"].endswith("seal.names")

    def test_bundled_detectron2(self, tmp_path):
        ckpt = torch_archive(tmp_path / "model_final.pth", {
            "model": {"roi_heads.w": 0}, "iteration": 10,
        })
        path = tmp_path / "d2.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.write(ckpt, "model_final.pth")
            zf.writestr("model_config.yaml", "MODEL: {}\n")
        info = model_wrap.identify(str(path), str(tmp_path))
        assert info.impl == "detectron2"
        assert info.keys["cfg"] == str(tmp_path / "d2" / "model_config.yaml")

    def test_bundled_mit_yolo_finds_train_config_anywhere(self, tmp_path):
        ckpt = torch_archive(
            tmp_path / "best.ckpt",
            raw_pickle=global_pickle("yolo.config.config", "ModelConfig"))
        path = tmp_path / "yolo.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.write(ckpt, "checkpoints/best.ckpt")
            zf.writestr("train_config.yaml", "model:\n  name: v9-s\n")
        info = model_wrap.identify(str(path), str(tmp_path))
        assert info.impl == "mit_yolo"
        assert info.keys["model"] == "v9-s"

    def test_bundle_without_companions_reports_the_first_reason(self, tmp_path):
        ckpt = torch_archive(tmp_path / "best_snapshot.pt", {
            "model_state_dict": {"w": 0},
        })
        path = tmp_path / "snap.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.write(ckpt, "best_snapshot.pt")
        info = model_wrap.identify(str(path), str(tmp_path))
        assert not info.runnable
        assert info.kind == "zip of netharn training snapshot"

    def test_bundle_makes_its_own_work_dir_when_none_given(self, tmp_path):
        path = tmp_path / "seal.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("seal.weights", b"\x00")
            zf.writestr("seal.cfg", "")
            zf.writestr("seal.lbl", "")
        info = model_wrap.identify(str(path))
        assert info.runnable
        assert os.path.isfile(info.keys["weight_file"])

    def test_empty_archive(self, tmp_path):
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("readme.txt", "")
        assert not model_wrap.identify(str(path)).runnable


class TestBuildPipeline:
    def test_checkpoint_renders_nested_windowed_block(self, tmp_path, templates):
        ckpt = torch_archive(tmp_path / "det.pth", {
            "model": {"transformer.decoder.w": 0, "class_embed.weight": 0},
        })
        info = model_wrap.identify(str(ckpt))
        work = tmp_path / "work"
        work.mkdir()
        pipe = model_wrap.build_pipeline(info, str(work), str(templates))
        text = open(pipe).read()

        assert "[-DETECTOR-IMPL-]" not in text
        assert "  :detector:type                             ocv_windowed" in text
        assert "    :detector:type                           rf_detr" in text
        assert "    :mode                                    disabled" in text
        assert "block detector:rf_detr" in text
        assert ":weight" in text and str(ckpt) in text
        assert "chip_width" not in text

    def test_resized_mode_adds_chip_size(self, tmp_path, templates):
        ckpt = torch_archive(
            tmp_path / "yolo.pt",
            raw_pickle=global_pickle("ultralytics.nn.tasks", "DetectionModel"))
        info = model_wrap.identify(str(ckpt))
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        text = open(pipe).read()
        assert ":mode                                    original_and_resized" in text
        assert ":chip_width" in text

    def test_darknet_is_not_nested_in_the_windowed_detector(self, tmp_path, templates):
        for name in ("seal.weights", "seal.cfg", "seal.lbl"):
            (tmp_path / name).write_bytes(b"\x00")
        info = model_wrap.identify(str(tmp_path / "seal.weights"))
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        text = open(pipe).read()
        assert "  :detector:type                             darknet" in text
        assert "ocv_windowed" not in text
        assert "    :net_config" in text and ":weight_file" in text

    def test_classifier_uses_the_frame_classifier_template(self, tmp_path, templates):
        path = netharn_zip(tmp_path / "trained_classifier.zip",
                           "viame.pytorch.netharn.clf_fit.ClfModel", "ClfModel_ab12.py")
        info = model_wrap.identify(str(path))
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        text = open(pipe).read()
        assert pipe.endswith("classifier.pipe")
        assert "[-DEPLOYED-]" not in text and "relativepath" not in text
        assert "  block detector:netharn_classifier\n" in text
        assert "    :deployed                                " + str(path) in text
        assert "  endblock" in text
        assert "process classifier1" in text

    def test_onnx_classifier_swaps_the_implementation(self, tmp_path, templates):
        path = tmp_path / "clf.onnx"
        path.write_bytes(b"")
        (tmp_path / "clf.modelspec.json").write_text('{"postprocess": {"decoder": "classifier"}}')
        info = model_wrap.identify(str(path))
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        text = open(pipe).read()
        assert "  :detector:type                             onnx_classifier" in text
        assert "  block detector:onnx_classifier\n" in text
        assert "netharn_classifier" not in text
        assert "    :model                                   " + str(path) in text

    def test_onnx_uses_its_own_template(self, tmp_path, templates):
        path = tmp_path / "pkg.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("m.onnx", "")
        info = model_wrap.identify(str(path))
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        text = open(pipe).read()
        assert ":model                                     " + str(path) in text

    def test_packaged_pipeline_is_spliced_between_reader_and_writer(
        self, tmp_path, templates
    ):
        path = tmp_path / "trained.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("detector.pipe", "process detector_input\n  :: image_filter\n")
            zf.writestr("trained_detector.pth", "")
        info = model_wrap.identify(str(path))
        work = tmp_path / "work"
        work.mkdir()
        pipe = model_wrap.build_pipeline(info, str(work), str(templates))
        text = open(pipe).read()

        inner = work / "trained" / "detector.pipe"
        assert inner.exists()
        assert (work / "trained" / "trained_detector.pth").exists()
        assert "include " + str(inner) in text
        assert "common_default_input_with_downsampler" in text
        assert "to   detector_input.image" in text
        assert "process detector_writer" in text
        assert "process detector1" not in text

    def test_complete_pipeline_in_zip_runs_as_is(self, tmp_path, templates):
        path = tmp_path / "addon.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "configs/pipelines/detector_x.pipe",
                "include $ENV{VIAME_INSTALL}/configs/pipelines/common_default_input.pipe\n"
                "process detector\n  :: image_object_detector\n")
        info = model_wrap.identify(str(path))
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        assert pipe == str(tmp_path / "addon" / "configs" / "pipelines" / "detector_x.pipe")

    def test_several_pipelines_must_be_narrowed_first(self, tmp_path, templates):
        path = tmp_path / "two.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("a.pipe", "")
            zf.writestr("b.pipe", "")
        info = model_wrap.identify(str(path))
        with pytest.raises(ValueError):
            model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        info.pipes = ["b.pipe"]
        pipe = model_wrap.build_pipeline(info, str(tmp_path), str(templates))
        assert (tmp_path / "two" / "b.pipe").exists()
        assert "include " + str(tmp_path / "two" / "b.pipe") in open(pipe).read()

    def test_not_runnable_raises(self, tmp_path, templates):
        path = tmp_path / "junk.pt"
        path.write_bytes(b"nope")
        info = model_wrap.identify(str(path))
        with pytest.raises(ValueError):
            model_wrap.build_pipeline(info, str(tmp_path), str(templates))
