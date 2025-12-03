import pytest
import numpy as np

q = 1.6021765650e-19
eps_0 = 8.8548782e-12
qf = 3e15


@pytest.mark.skip(reason="cassé")
@pytest.mark.parametrize(
    "n,rho,Phi",
    [
        (100, lambda x: 0, lambda x: 10 * x),
        (100, lambda x: 0, lambda x: 10 - 10 * x),
        (1000, lambda x: 0, lambda x: 10 * x),
        (1000, lambda x: -6 * x * eps_0, lambda x: x**3 - 10 * x + 2),
    ],
)
def test_dirichlet(n, rho, Phi):
    from pic.poisson_solver import Dirichlet

    V0 = Phi(0)
    Vn = Phi(1)
    PS = Dirichlet(n, V0, Vn)
    PS.init_thomas()

    rho = np.array(list(map(rho, np.linspace(0, 1, n))))
    Phi = np.array(list(map(Phi, np.linspace(0, 1, n))))

    normed_rho = rho / (q * qf * n)
    normed_phi = PS.thomas_solver(normed_rho, 0, 1 / n, q, qf, eps_0)

    phi = normed_phi / eps_0 * (q * qf) / n

    for i in range(len(Phi)):
        assert np.isclose(phi[i], Phi[i], rtol=1e-1), (Phi[i], phi[i], i)
