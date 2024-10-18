import sys

sys.path.append(".")

import pathlib

import click


@click.command()
@click.argument(
    "path",
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
def main(path):
    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    for action in path.iterdir():
        if action.is_file():
            continue

        action.rename(action.parent / action.name.replace(" ", "_"))


if __name__ == "__main__":
    main()
