import argparse
import plotext as pot
import pickle
import astropy.units as u


def main():
    parser = argparse.ArgumentParser(description="1D-3V PIC-MCC")
    parser.add_argument(
        "configuration",
        metavar="CFG",
        type=str,
        nargs="?",
        help="A path to the parameters file",
    )
    args = parser.parse_args()

    parameters: dict = pickle.load(open(args.configuration, "rb"))

    for k, v in parameters.items():
        match k:
            case "species" | "neutrals":
                print(k)
                for kk, vv in v.items():
                    match vv:
                        case (x, n, _T):
                            pot.plot(
                                x.to_value(u.cm),
                                n.to_value(u.m**-3),
                            )
                            pot.show()
                        case (x, n, _T, _V):
                            pot.plot(
                                x.to_value(u.cm),
                                n.to_value(u.m**-3),
                            )
                            pot.show()
                        case _:
                            print(kk, vv)
            case _:
                print(k, v)


if __name__ == "__main__":
    main()
