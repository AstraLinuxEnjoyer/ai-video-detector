"""One-command spatial training with audited data and isolated resumable runs."""

import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
# Keep dataset transfers on the HTTP path governed by the timeouts above.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from training.spatial.run import main


if __name__ == "__main__":
    main()
