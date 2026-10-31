# Model provenance registry

Every weight file used by this project is recorded here: source, license, and
SHA-256 as downloaded on 2026-11-01. Weight files themselves are git-ignored
(`models/*.onnx`); fetch them with the commands below and verify the hashes.

| File | Role | Source | License | SHA-256 |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | Face detection + 5 landmarks | OpenCV Zoo `face_detection_yunet` | MIT | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | Face embedding (128-d) | OpenCV Zoo `face_recognition_sface` | Apache-2.0 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `object_detection_yolox_2022nov.onnx` | Person/object detection | OpenCV Zoo `object_detection_yolox` | Apache-2.0 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

All three are commercially usable. InsightFace model packs and Ultralytics
weights are deliberately absent (non-commercial / AGPL — see the research
report in `docs/`).

## Fetch commands

```bash
cd models
curl -sL -o face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
curl -sL -o object_detection_yolox_2022nov.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx
sha256sum *.onnx   # compare with the table above
```

## Field note (Labkom 2 sample frame, 1920×1080)

Only **2 faces** in the whole classroom frame pass the runner's
identification gate (width ≥ 40 px and detector quality ≥ 0.6). The encoder
returns weaker detections too, but they only inform `unresolved_reason`.
This measured result is the basis for the walk-in capture + look-up burst +
`id_spot` identification strategy documented in the research report.
