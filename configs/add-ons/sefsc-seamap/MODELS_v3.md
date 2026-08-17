# sefsc-seamap v3 model pack (staging)

Drop these into the add-on's `models/` directory. Not checked in.

## Present here

| file | what | source |
|---|---|---|
| `gfit_groups_rf_detr_1728.pth` | group RF-DETR, detector 1 of the fusion | `SEFSC/Models/rgb_1728_groups/trained_detector.pth`, 46 classes, 247,828 steps |
| `fishtrack_motion_rf_detr_1728.pth` | motion RF-DETR, detector 2 of the fusion | install `fish_with_motion_rf_detr.pth`, 316,992 steps |
| `gfit_groups_enetv2m_large.zip` | groups classifier | `SEFSC/Models/classifiers/large_fish_enetv2m.zip` |
| `gfit_species_enetv2m_large.zip` | species classifier | `large_fish_enetv2m_species.zip` |
| `fishtrack_srnn_siamese.pt` | SRNN appearance head | FishTrack23 round 3, `siamese_model.pt` |
| `fishtrack_srnn_rnn_f.pt` | SRNN targetRNN AIM | `target_lstm_F.pt` |
| `fishtrack_srnn_rnn_v.pt` | SRNN targetRNN AIM V | `target_lstm_V.pt` |
| `sam2_hbp.pt` | segmentation, measurement pipes | install `configs/pipelines/models/` |
| `gfit_cal.json` | stereo calibration | `~/Desktop`; T_y -176.9 mm matches the vertical-baseline note in the stereo pipes |

`gfit_` replaces the old `sefsc_` / `seamap_` / `seamap-` prefixes only. The
FishTrack23-derived models keep `fishtrack_`, and `sam2_hbp.pt` and the
`.CamCAL` files keep their upstream names.

## Still needed

| file | why |
|---|---|
| `SC6_camera3_2024.CamCAL`, `SC6_satelliteA_2024.CamCAL` | seagis measurement pipes only. Not in either `VIAME-SEFSC-SEAMAP-Models*.zip` (34 entries each, all pipes and models) and not anywhere on this machine. |

Detection, classification and tracking are complete, as are both `vme`
measurement pipes. Only the two `seagis` pipes remain blocked.

`measurement_seamap_groups_v3_vme_fast_fdn.pipe` was removed from the pack
pending the Fast Foundation Stereo ONNX export.

## Chosen settings

Single large classifier with `average_prior true` and `prior_weight 0.30`.
On a 52-sequence SEFSC subset, group mAP@50:

| | all classes | top 10 |
|---|---|---|
| detector alone, no reclassification | 0.248 | 0.339 |
| v2.5 chain (enet2s large+small) | 0.331 | 0.470 |
| single enetv2m large, no prior | 0.562 | 0.690 |
| **single enetv2m large + 0.30 prior** | **0.574** | **0.707** |

A 4-way classifier ensemble reached 0.580 but costs four classifier passes, so
it is not used here.
