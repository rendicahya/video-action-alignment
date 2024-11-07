from os.path import basename, exists

command = input("Paste your command in one line (no new lines): ").strip()

for single in command.split(";"):
    if not single.strip():
        continue

    config = single.split()[2]
    base = basename(config)

    print(base, exists(config))
