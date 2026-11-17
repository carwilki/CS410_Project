"""Entry point.

    uv run python main.py                     # full-screen search + admin UI (user decision: TUI only)
    uv run python main.py bootstrap [--check] # build corpus + indexes from data/*.zip on a fresh clone
"""
import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["bootstrap"]:
        from cfpb_hyde.bootstrap import main
        main(sys.argv[2:])
    else:
        from cfpb_hyde.tui import run_tui
        run_tui()
