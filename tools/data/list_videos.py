import sys

sys.path.append(".")

import os
from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.argument(
    "video-path",
    nargs=1,
    required=False,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
def main(video_path):
    ROOT = Path.cwd()
    OUT_FILE = video_path / "list.txt"
    N_VIDEOS = sum(1 for f in video_path.glob(f"**/*.*") if f.parent != video_path)

    print("Input:", video_path.relative_to(ROOT))
    print(
        "Output:",
        OUT_FILE.relative_to(ROOT),
        "(exists)" if OUT_FILE.exists() else "(not exists)",
    )
    print("Σ videos:", N_VIDEOS)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    data = []
    bar = tqdm(total=N_VIDEOS, dynamic_ncols=True)
    class_id = 0

    for action in sorted(video_path.iterdir()):
        if action.is_file():
            continue

        for file in sorted(action.iterdir()):
            if file.is_dir():
                continue

            line = f"{file.relative_to(video_path)} {class_id}"

            data.append(line)
            bar.update(1)

        class_id += 1

    bar.close()

    with open(OUT_FILE, "w") as f:
        f.write(os.linesep.join(data))


if __name__ == "__main__":
    main()
