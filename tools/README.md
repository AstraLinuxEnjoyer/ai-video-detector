# Frozen TRACE decoder runtime

Place the matching Windows x64 **full build** of `ffmpeg.exe` and `ffprobe.exe`
in this folder. The required exact FFmpeg version line is recorded in
`demo_weights/spectral.json` (currently 8.1.1-full_build-www.gyan.dev).
Keep the build license, GPL text and source/build information next to the files.
Upstream builds: https://www.gyan.dev/ffmpeg/builds/.

Both the launcher and direct `uvicorn demo.app:app` execution prefer this complete
local pair over the Windows PATH. Global FFmpeg can remain at a different version.
The selection changes only the current process environment.

Before use or packaging, run from the project directory:

```powershell
.\.demo-venv\Scripts\python.exe -B -m demo.check_runtime --json
```

Expect `status: ready` and `spectral.status: ready`. The preflight verifies
frozen spatial/motion checksums, the spectral recipe, NumPy/OpenCV/FFmpeg
versions and YuNet checksum. Never edit spectral.json to silence an error.

The EXE files are intentionally excluded from ordinary Git. Transfer this folder
with the tested portable bundle or the future installer; a source clone with
LFS weights alone is not a complete decoder installation. Keep executables
pinned to the model contract when updating the application.
