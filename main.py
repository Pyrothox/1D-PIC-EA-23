## Import ##

import numpy as np
import os
from os import path
import argparse

import pickle
from pic.plasma import Plasma
from pic.MergingSplitting import DPMSA

from pic.parsing import get_parameters

from datetime import datetime
from pathlib import Path
import signal
import sys

## Parameters ##


def main():
    parser = argparse.ArgumentParser(description="1D-3V PIC-MCC")
    parser.add_argument(
        "configuration",
        metavar="CFG",
        type=str,
        nargs="?",
        help="A path to the configuration file",
        default="configurations/turner1.cfg",
    )
    parser.add_argument(
        "--wall-time",
        "-w",
        dest="wall",
        action="store",
        nargs="?",
        type=int,
        help="Maximum time before save and stop in seconds",
    )
    parser.add_argument(
        "--restart",
        "-r",
        dest="restart",
        action="store",
        nargs="?",
        type=str,
        help="Path to the folder of the run to be resumed",
    )
    parser.add_argument(
        "--name",
        "-n",
        dest="name",
        action="store",
        nargs="?",
        type=str,
        help="Name of the result folder",
    )
    parser.add_argument(
        "--profile",
        "-p",
        dest="prof",
        action="store_true",
        help="Flag to enable profiling",
    )

    args = parser.parse_args()

    print(args)

    np.seterr(all="raise", under="ignore")
    if args.restart is not None:
        restart = Path(args.restart)
        parameters = pickle.load(open(restart / "parameters.pkl", "rb"))

    else:
        parameters = get_parameters(args.configuration)
        restart = None

    start(parameters, restart, args.wall, args.name, args.prof)


def start(parameters, restart=None, wall_time=None, name=None, prof=False):
    ## Restart initialisation ##
    pla = Plasma(**parameters)
    start_time = datetime.now()
    Time = start_time.strftime("%Y-%m-%d_%Hh%M")
    if restart is not None:
        restart_file = restart / "restart.h5"
        if restart_file.exists():
            t0 = pla.load(restart_file) + 1
            print("restarting")
        else:
            t0 = 0
            pla.init_particles()
        dataFileName = restart

    else:
        pla.init_particles()
        t0 = 0
        if name is not None:
            dataFileName = "data/" + name

        else:
            try:
                os.mkdir("data")
            except FileExistsError:
                pass
            dataFileName = "data/" + Time
        os.mkdir(dataFileName)
        pickle.dump(parameters, open(path.join(dataFileName, "parameters.pkl"), "wb"))

    pla.print_init()

    ## Create files and save parameters ##

    print("~~~~ Data File Name: ", dataFileName)
    
    ## Initialize merging-splitting algorithm ##
    dpmsa = DPMSA(pla)

    ## Do loops ##

    def run():
        interrupted = False

        def signal_handler(sig, frame):
            nonlocal interrupted
            if interrupted:
                print(
                    "2nd SIGINT received, terminating simulation now.", file=sys.stderr
                )
                sys.exit(0)
            else:
                interrupted = True
                print(
                    "SIGINT received, stopping simulation after next average cycle.",
                    file=sys.stderr,
                )

        signal.signal(signal.SIGINT, signal_handler)

        # Main loop #
        # entire cycle duration Nt
        for avg_nt in range(t0, parameters["Nt"], parameters["n_average"]):
            # average cycle duration n_average
            cycle = 0
            for nt in range(avg_nt, avg_nt + parameters["n_average"]):
                pla.pusher()
                pla.boundary()
                pla.apply_mcc()
                pla.inject()
                dpmsa.execute(nt)
                pla.compute_rho()
                pla.solve_poisson(nt)
                pla.diags(nt)
                
            cycle +=1
            pla.diagnostics.average_diags(parameters["n_average"])
            if pla.wall is not None:
                pla.wall.update()
            pla.diagnostics.save_diags(nt, dataFileName)

            # pla.recombine(pla.diagnostics.average)
            total_particles = sum((part.Npart for part in pla.species.values()))
            print(
                f"loop # {int(nt) + 1:}, t = {nt * pla.dT * 1e6:2.5f} over {parameters['Nt'] * pla.dT * 1e6:2.5f} mu s. Compute time: {datetime.now() - start_time}, {total_particles:} particles",
                flush=True,
            )
            if (
                interrupted
                or wall_time is not None
                and wall_time < (datetime.now() - start_time).seconds
            ):
                pla.save(dataFileName, nt)
                interrupted = True
                print("\nrestart file created")
                break

        if not interrupted:
            print("\nend of simulation")
            pla.save(dataFileName, nt)
            print("\nrestart file created")

    if prof:
        import cProfile
        import pstats

        print("Starting profiling")
        with cProfile.Profile() as pr:
            run()

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
        stats.dump_stats(filename=path.join(dataFileName, "loops.prof"))
    else:
        run()


if __name__ == "__main__":
    main()
