import sys

sys.path.append(".")

import os
import pathlib

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
        path_type=pathlib.Path,
    ),
)
def main(video_path):
    out_file = video_path / "list.txt"
    n_videos = sum(1 for f in video_path.glob(f"**/*") if f.is_file())

    print("Input:", video_path)
    print("Output:", out_file)
    print("Σ videos:", n_videos)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    data = []
    bar = tqdm(total=n_videos, dynamic_ncols=True)
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

    with open(out_file, "w") as f:
        f.write(os.linesep.join(data))


if __name__ == "__main__":
    main()
