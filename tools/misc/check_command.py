from os.path import basename, exists, splitext

command = input("Paste your command in one line (no new lines): ").strip()
skip = "train", "test", "confusion_matrix"

for single in command.split(";"):
    if not single.strip():
        continue

    for segment in single.strip().split():
        ext = splitext(segment)[1]
        base = basename(segment)
        stem = splitext(base)[0]

        if not ext or ext not in (".py", ".pth") or stem in skip:
            continue

        print(base, exists(segment))
