import argparse
from typing import List

from main import start as start_main
import multiprocessing as mp
from pathlib import Path
import sys
import pickle


def start(folder):
    with open(folder / "out.txt", "w") as f:
        sys.stdout = f
        config = folder / "parameters.pkl"
        parameters = pickle.load(open(config, "rb"))
        try:
            start_main(parameters, restart=folder)
        except Exception as e:
            raise RuntimeError(f"Error in {folder}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D-3V PIC-MCC + fluid simulation, multiprocess version for inplace restarts"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="*",
        help="A path to the configuration file",
    )

    args = parser.parse_args()
    targets: List[Path] = []
    file = [Path(conf) for conf in args.path]
    while file:
        path = file.pop()
        if (path / "parameters.pkl").exists():
            targets.append(path)
        else:
            file.extend(p for p in path.iterdir() if p.is_dir())

    mp.set_start_method("fork")
    handles = []
    for folder in targets:
        print(f"Starting {folder} ")
        p = mp.Process(target=start, args=(folder,))
        p.start()
        handles.append(p)

    for p in handles:
        p.join()
    print("All done")
