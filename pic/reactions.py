from plasmapy import particles
import astropy.units as u
import re
import pic.parsing as pars
import numpy as np

regex_e = r"([A-Z_]+) *\n(.+) *\n( ?.+) *\n((?:[A-Z1-9_]+\.? ?:.+\n)*)-+\n((?: *\d.\d+e[+-]\d+\s+\d.\d+(?:e[+-]\d+)?\n)+)-+"
regex_i = (
    r"((?:[A-Z1-9_]+\.? *.:.+\n)+)-+\n((?: *\d.\d+e[+-]\d+\s+\d.\d+(?:e[+-]\d+)?\n)+)-+"
)


class Reaction:
    def __init__(self, energy, cross_section, *params, **infos):
        assert energy.shape == cross_section.shape
        self.energy = energy
        self.cross_section = cross_section
        self.infos = infos
        if params:
            self.type = params[0]
            self.process = params[1]
            if self.type in ("EFFECTIVE", "ELASTIC"):
                self.m_ratio = float(params[2])
            else:
                self.threshold = u.Quantity(params[2], u.eV)
            if self.type in (
                "IONIZATION",
                "DISSOCIATIVE_IONIZATION",
                "DISSOCIATIVE_ATTACHMENT",
            ):
                self.product = pars.parse_molecule(
                    self.process.split("->")[1].split()[-1].strip().replace("^", " 1")
                )
        else:
            self.type = infos["PROCESS"].split(",")[1].strip().upper()
            if self.type in ("CHARGE_EXCHANGE"):
                self.product = pars.parse_molecule(
                    infos["PROCESS"]
                    .split("->")[1]
                    .split()[2]
                    .strip()
                    .replace("^", " 1")
                    .rstrip(",")
                )
                param = infos["PARAM."].split("=")
                if len(param) == 2 and param[0].strip() == "delta E":
                    self.energy_loss = u.Quantity(param[1].strip(), u.eV)

    def cross_section_interpolator(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.energy.to_value(u.J), self.cross_section.to_value(u.m**2))

    def __repr__(self):
        if self.type in (
            "IONIZATION",
            "DISSOCIATIVE_IONIZATION",
            "DISSOCIATIVE_ATTACHMENT",
            "EXCITATION",
        ):
            return f"{self.type}  with {self.threshold}"
        elif self.type in ("CHARGE_EXCHANGE"):
            return f"{self.type}  to {self.product} with {self.energy_loss} loss"
        else:
            return f"{self.type}"

    def name(self):
        if self.type in ("EXCITATION",):
            return f"{self.type}({self.threshold.to_value(u.eV):.2f}eV)"
        else:
            return f"{self.type}"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Reaction):
            return (
                self.type == __o.type
                and np.all(self.energy == __o.energy)
                and np.all(self.cross_section == __o.cross_section)
            )
        return NotImplemented


class ReactionsSet:
    def __init__(self, specie: particles.ParticleLike, neutral: particles.ParticleLike):
        self.specie = specie
        self.neutral = neutral
        self.reactions: list[Reaction] = []

    def __str__(self) -> str:
        return f"{self.specie.symbol} -> {self.neutral.symbol} with {len(self.reactions)} reactions"

    def __repr__(self) -> str:
        return f"{self.specie.symbol} -> {self.neutral.symbol} with {self.reactions}"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, ReactionsSet):
            return (
                self.specie == __o.specie
                and self.neutral == __o.neutral
                and self.reactions == __o.reactions
            )
        return NotImplemented

    def read_txt(self, file):
        with open(file, "r") as f:
            input = f.read()
            if self.specie == particles.electron:
                matches = re.finditer(regex_e, input, re.MULTILINE)
                for match in matches:
                    params = [match.group(i).strip() for i in range(1, 4)]
                    infos = {}
                    for line in match.group(4).splitlines():
                        k, v, *_ = line.split(":")
                        if k in infos:
                            infos[k] = infos[k] + v
                        else:
                            infos[k] = v
                    energy = []
                    cross_section = []
                    for line in match.group(5).splitlines():
                        e, sigma = line.split()
                        energy.append(float(e))
                        cross_section.append(float(sigma))

                    energy = u.Quantity(energy, u.eV)
                    cross_section = u.Quantity(cross_section, u.m**2)
                    self.reactions.append(
                        Reaction(energy, cross_section, *params, **infos)
                    )

            else:
                matches = re.finditer(regex_i, input, re.MULTILINE)
                for match in matches:
                    infos = {}
                    for line in match.group(1).splitlines():
                        k, v, *_ = line.split(":")
                        if k in infos:
                            infos[k] = infos[k] + v
                        else:
                            infos[k] = v
                    energy = []
                    cross_section = []
                    for line in match.group(2).splitlines():
                        e, sigma = line.split()
                        energy.append(float(e))
                        cross_section.append(float(sigma))

                    energy = u.Quantity(energy, u.eV)
                    cross_section = u.Quantity(cross_section, u.m**2)
                    self.reactions.append(Reaction(energy, cross_section, **infos))

    def total_cross_section(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        # en_min = min((r.energy[0].to_value(u.eV) for r in self.reactions))
        # en_max = max((r.energy[-1].to_value(u.eV) for r in self.reactions))
        # en_tot = np.linspace(en_min, en_max, n)

        # cs_tot = sum(
        #     (
        #         np.interp(
        #             en_tot,
        #             r.energy.to_value(u.eV),
        #             r.cross_section.to_value(u.m**2),
        #             0,
        #             0,
        #         )
        #         for r in self.reactions
        #     )
        # )
        cs_tot = sum(r.cross_section.to_value(u.m**2) for r in self.reactions)
        return (self.reactions[0].energy.to_value(u.eV), cs_tot)

    def __getitem__(self, key):
        return self.reactions[key]
