"""Simple stdout logging for the classifier CLI."""


def log(*args) -> None:
    print(*args, flush=True)