# Frozen inference assets

All three final branches are versioned in this repository:

- `spatial.pt`: EfficientNet V2-S Hybrid, epoch 5 (Git LFS).
- `temporal.pt`: R(2+1)D-18 full-frame motion flow, epoch 14 (Git LFS).
- `manifest.json`: versions, preprocessing recipes and SHA-256 of both.

- `spectral.json`: trained `spectral-face-v3` classifiers, normalization,
  thresholds and preprocessing configuration. This is the actual third
  model, not an instruction to train it.
- `face_detection_yunet_2026may.onnx`: OpenCV Zoo YuNet face locator used by
  the spectral branch. It does not classify videos as real or fake.
  SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
  Its MIT license is in `YUNET_LICENSE.txt`; upstream:
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

`demo.spectral.SpectralDetector` uses these assets before falling back to
the training cache. No datasets or training cache are required for inference.
The pinned preprocessing versions must match `demo-requirements.txt` and
the FFmpeg build recorded in `spectral.json`. A different FFmpeg version
can cause the branch to be unavailable: do not silently bypass that check.

Install Git LFS before cloning, or run these commands in an existing checkout:

```powershell
git lfs install
git pull --ff-only origin main
git lfs pull
git lfs fsck
```

Do not use GitHub's source ZIP as the model installation: it may contain
LFS pointer files instead of the actual weights. No retraining is needed.
The full `releases/TRACE-final-windows.zip` is another way to transfer the
application, including the matching FFmpeg binaries; then run `setup-demo.cmd`.

Install the recorded FFmpeg build and validate all branches with `setup-demo.cmd`. Then run
`start-demo.cmd` and check a new video; old reports are not recomputed.
