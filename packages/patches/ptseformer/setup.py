# VIAME patch: upstream PTSEFormer ships no packaging. This installs its model
# code under a "ptseformer" package so the generic top-level names it uses
# ("src", "util") do not land in site-packages. The dataset layer, training
# scripts and the compiled deformable-attention sources are left out: the
# VIAME plugin (plugins/pytorch/ptseformer_*.py) provides its own.

from setuptools import setup

MODULES = [
    "src",
    "src.models",
    "src.models.backbone",
    "src.models.detr_util",
    "src.models.matcher",
    "src.models.ops",
    "src.models.ops.functions",
    "src.models.ops.modules",
    "src.models.transformer",
    "util",
]

setup(
    name="ptseformer",
    version="0.0.0+979c023",
    description="PTSEFormer: Progressive Temporal-Spatial Enhanced TransFormer "
                "Towards Video Object Detection (ECCV 2022)",
    url="https://github.com/Hon-Wong/PTSEFormer",
    license="MIT",
    packages=["ptseformer"] + ["ptseformer." + name for name in MODULES],
    package_dir={"ptseformer": "."},
    python_requires=">=3.8",
)
