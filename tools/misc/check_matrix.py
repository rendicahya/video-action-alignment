import sys

sys.path.append(".")

import click
import numpy as np


@click.command()
@click.argument(
    "matrix-path",
    nargs=1,
    required=False,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
    ),
)
def main(matrix_path):
    matrix = np.load(matrix_path)["arr_0"]
    check_value = matrix[-2, -1]

    if check_value == 0:
        print("Matrix is not valid.")
    else:
        print("Matrix is valid. Check value:", check_value)


if __name__ == "__main__":
    main()
