import sys

from .driver.fastrun import try_warm


def main() -> int:
    """`ppy`: a remembered `ppy run FILE` runs before the command line is even
    built; everything else goes through it."""
    warm = try_warm(sys.argv[1:])
    if warm is not None:
        return warm
    from .driver.cli import main as full

    return full()


if __name__ == "__main__":
    raise SystemExit(main())
