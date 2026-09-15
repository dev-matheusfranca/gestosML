"""python -m gesturelab abre o desktop; subcomandos usam a CLI."""
import sys


def main():
    if len(sys.argv) == 1:
        from gesturelab.ui import run
        return run()
    from gesturelab.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
