from pic.parsing import get_parameters
import argparse
from pathlib import Path
import pickle


def main():
    parser = argparse.ArgumentParser(description="PIC configuration parser")
    parser.add_argument(
        "configuration",
        metavar="CFG",
        type=str,
        help="A path to the configuration file",
    )

    parser.add_argument(
        "target", metavar="TARGET", type=str, nargs="?", default="simulation"
    )

    args = parser.parse_args()
    print(args)
    parameters = get_parameters(args.configuration)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    param_file = target / "parameters.pkl"
    print(f"Saving parameters at {param_file}")
    with open(param_file, "wb") as f:
        pickle.dump(parameters, f)


if __name__ == "__main__":
    main()
