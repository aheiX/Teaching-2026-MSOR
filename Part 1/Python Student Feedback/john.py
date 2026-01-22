"""Print the current local day and time."""

from datetime import datetime


def main() -> None:
    now = datetime.now()
    # Example output: Hello! Right now it's Monday 2026-01-22 at 02:30:05 PM.
    print(f"Hello! Right now it's {now.strftime('%A %Y-%m-%d at %I:%M:%S %p')}.")


if __name__ == "__main__":
    main()
