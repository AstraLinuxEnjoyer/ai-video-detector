# Dataset locations

The downloaded archive directory moved from `C:/Users/Ivan/Downloads/dataset`
to `D:/dataset`. The project's `datasets/archive` directory junction points to
that new location without copying any files. It is excluded from Git.

`training.dataset_paths.ARCHIVE_DATASET_DIR` uses this junction by default.
Set `AI_VIDEO_ARCHIVE_ROOT` or pass `--dataset-root` to override the archive root
on another machine. No machine-specific D drive is hardcoded in the Python code.

GenVideo source names resolve to `GenVideo-100K/extracted/train_*` underneath
the archive root. For example, inspect an import plan without changing data:

```powershell
python -m training.import_videos --dataset GenVideo_OpenSora --label fake --source GenVideo_OpenSora --limit 1000 --dry-run
```

The legacy explicit `--input` command remains supported. The archive directory
is NOT the training directory: working videos still live under `data/real` and
`data/fake`. Existing imported filenames, frozen partitions, weights and
checkpoints have not been modified. Existing run commands remain unchanged;
no rebuild or fresh training is required solely because archives moved.

Hardlinks left under the project's `data` remain real files on C after the
archive directory moves to D. They continue to occupy C disk space. Cross-volume
hardlinks are impossible; new imports refuse to silently copy from D back to C.
Use `--mode copy` only deliberately, or separately migrate the working `data`
directory with integrity checks and reconnect its old path before importing.
Do not delete `data` thinking the relocated archives replace its contents.

CocoGlide remains a separate junction to `C:/Users/Ivan/Downloads/CocoGlide`.
FakeParts remains under `datasets/FakeParts`. Neither is included in existing
frozen training runs merely by relocating or downloading files.
