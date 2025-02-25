import re
from pathlib import Path

import click


def print_log(lines, pattern, mark):
    for line in lines:
        if not mark in line:
            continue

        match = pattern.search(line)

        if not match:
            continue

        progress_current = match.group(2)
        progress_total = match.group(3)

        if progress_current != progress_total:
            continue

        epoch = match.group(1)
        acc_top1 = match.group(4)

        print(acc_top1)


@click.command()
@click.argument(
    "log-path",
    nargs=1,
    required=False,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(log_path):
    if log_path.suffix != ".log":
        print("Please enter a .log file")
        exit()

    with open(log_path) as f:
        lines = f.readlines()

    train_pattern = re.compile(
        r"Epoch\(train\) *\[(\d+)\]\[(\d+)\/(\d+)\].*?top1_acc: (\d\.\d+)"
    )
    val_pattern = re.compile(
        r"Epoch\(val\) \[(\d+)\]\[(\d+)/(\d+)\].*?acc/top1: (\d\.\d+)"
    )

    print("Train:")
    print_log(lines, train_pattern, "Epoch(train)\n")

    print("Validation:")
    print_log(lines, val_pattern,"Epoch(val)")


if __name__ == "__main__":
    main()
