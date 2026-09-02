Sea Lion Background Classifier Add-On
=====================================

SVM classifiers that categorize sea lion survey imagery by background type
(open water, coastal, cloudy, land, seaweed/water), used by the two pipelines
in this folder to gate or suppress detections by scene content.

These models were part of the original SEA-LION pack and were dropped when
that pack was slimmed to the fusion detector models, which orphaned the two
pipelines that need them. They now live in their own pack so the sea lion
detector add-on no longer installs pipelines whose models it does not carry.

Model pack contents (``configs/pipelines/models/sea_lion_v3_bg_classifiers``)::

    all_land.svm  cloudy.svm  coastal.svm  open_water.svm  seaweed_water.svm

Upload note: the pack zip was rebuilt from a pre-slimming install
(md5 ``05646b0d005886ee74f0f64985cc7686``). Until it is uploaded to
viame.kitware.com and the URL in ``cmake/download_viame_addons.csv`` is
updated, enabling ``VIAME_DOWNLOAD_MODELS-SEA-LION-BG-CLASSIFIER`` will fail
at the download step.
