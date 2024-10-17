import sys

sys.path.append(".")

import pathlib

import click
import cv2
import numpy as np
from python_video import frames_to_video


@click.command()
@click.argument(
    "mask-path",
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
@click.option(
    "--output_path",
    "-o",
    default="mask.mp4",
    help="Output path for the generated video.",
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.option(
    "--fps",
    "-f",
    default=30,
    help="Frames per second (FPS) for the output video.",
    type=int,
)
def main(mask_path, output_path, fps):
    mask = np.load(mask_path)["arr_0"]
    out_frames = []

    for i, frame in enumerate(mask):
        frame = np.repeat(np.expand_dims(frame, axis=2), 3, axis=2)
        frame = cv2.putText(
            frame,
            str(i),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_frames.append(frame)

    frames_to_video(
        out_frames,
        output_path,
        writer="moviepy",
        fps=fps,
    )


if __name__ == "__main__":
    main()
