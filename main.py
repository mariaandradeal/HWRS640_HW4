import argparse
from data import summarize_dataset


def main():
    parser = argparse.ArgumentParser(description="HWRS640 Assignment 4 CLI")

    parser.add_argument(
        "command",
        choices=["summarize-data", "train", "evaluate", "plot"],
        help="Command to run"
    )

    args = parser.parse_args()

    if args.command == "summarize-data":
        summarize_dataset()

    elif args.command == "train":
        print("Train command not implemented yet.")

    elif args.command == "evaluate":
        print("Evaluate command not implemented yet.")

    elif args.command == "plot":
        print("Plot command not implemented yet.")


if __name__ == "__main__":
    main()
