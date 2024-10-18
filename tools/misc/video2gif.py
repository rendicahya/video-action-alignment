import pathlib
import sys

import click
from moviepy.editor import VideoFileClip

sys.path.append(".")


@click.command()
@click.argument(
    "path",
    nargs=1,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
)
def main(path):
    clip = VideoFileClip(str(path))
    name = f"{path.stem}.gif"

    clip.write_gif(name, fps=clip.fps, program="ffmpeg")


if __name__ == "__main__":
    main()
