# sefsc-seamap v3 model pack (staging)

Drop these into the add-on's `models/` directory. Not checked in.

## Present here

| file | what | source |
|---|---|---|
| `fishtrack_motion_rf_detr_1728.pth` | motion RF-DETR, detector 2 of the fusion | install `fish_with_motion_rf_detr.pth`, 316,992 steps |
| `seamap_groups_enetv2m_large.zip` | groups classifier | `SEFSC/Models/classifiers/large_fish_enetv2m.zip` |
| `seamap_species_enetv2m_large.zip` | species classifier | `large_fish_enetv2m_species.zip` |
| `fishtrack_srnn_siamese.pt` | SRNN appearance head | FishTrack23 round 3, `siamese_model.pt` |
| `fishtrack_srnn_rnn_f.pt` | SRNN targetRNN AIM | `target_lstm_F.pt` |
| `fishtrack_srnn_rnn_v.pt` | SRNN targetRNN AIM V | `target_lstm_V.pt` |

## Still needed

| file | why |
|---|---|
| `sefsc_groups_rf_detr_1728.pth` | detector 1 of the fusion. The SEFSC-trained group RF-DETR is not on this machine -- its detections were copied in from elsewhere. |
| `sam2_hbp.pt` | segmentation stage, measurement pipes |
| `seamap-cal.json`, `SC6_camera3_2024.CamCAL`, `SC6_satelliteA_2024.CamCAL` | stereo calibration, measurement pipes |
| `fast-fdn-stereo.onnx` | fast-fdn measurement pipe |

Carried over from the v2.5 pack; they were not in the install tree.

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
