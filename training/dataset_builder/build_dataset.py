from pathlib import Path
import random
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]

REAL_DIR = ROOT / "data" / "real"
FAKE_DIR = ROOT / "data" / "fake"

OUTPUT = ROOT / "training" / "datasets" / "spatial"

TRAIN_RATIO = 0.8

FRAMES_PER_VIDEO = 8

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
]


def find_videos(folder: Path):

    videos = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.rglob(f"*{ext}"))

    return sorted(videos)


def extract_frames(video_path: Path, output_folder: Path):

    output_folder.mkdir(parents=True, exist_ok=True)

    pattern = output_folder / "frame_%03d.jpg"

    command = [

        "ffmpeg",

        "-y",

        "-loglevel",

        "error",

        "-i",

        str(video_path),

        "-vf",

        f"select='not(mod(n\\,10))'",

        "-vsync",

        "vfr",

        str(pattern)

    ]

    subprocess.run(command)

    frames = sorted(output_folder.glob("*.jpg"))

    if len(frames) > FRAMES_PER_VIDEO:

        step = len(frames) / FRAMES_PER_VIDEO

        keep = []

        for i in range(FRAMES_PER_VIDEO):

            keep.append(frames[int(i * step)])

        for frame in frames:

            if frame not in keep:
                frame.unlink()

        frames = sorted(output_folder.glob("*.jpg"))


def process_class(video_list, split, label):

    root = OUTPUT / split / label

    for index, video in enumerate(video_list):

        video_name = f"{index:06d}"

        out_dir = root / video_name

        extract_frames(video, out_dir)

        print(
            f"{label.upper()} [{split}] {index+1}/{len(video_list)}",
            end="\r"
        )


def main():

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    real = find_videos(REAL_DIR)
    fake = find_videos(FAKE_DIR)

    print("Real videos :", len(real))
    print("Fake videos :", len(fake))

    random.shuffle(real)
    random.shuffle(fake)

    real = real[:len(fake)]

    print()
    print("Balanced dataset")
    print("Real :", len(real))
    print("Fake :", len(fake))
    print()

    real_split = int(len(real) * TRAIN_RATIO)
    fake_split = int(len(fake) * TRAIN_RATIO)

    process_class(
        real[:real_split],
        "train",
        "real"
    )

    process_class(
        real[real_split:],
        "val",
        "real"
    )

    process_class(
        fake[:fake_split],
        "train",
        "fake"
    )

    process_class(
        fake[fake_split:],
        "val",
        "fake"
    )

    print()
    print()
    print("Dataset successfully created.")


if __name__ == "__main__":
    main()