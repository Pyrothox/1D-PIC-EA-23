# test the values of the constantes
import pytest
import numpy as np

from pic.functions import numba_interp1D_normed, particle_to_grid
from numba.typed import List

# heure + date pour les noms de dossier


def test_generate_maxw():
    from pic.functions import generate_maxw

    temperature = 1
    mass = 0.1
    v = generate_maxw(temperature, mass)

    assert type(v) is float

    sample_size = 1000
    velocities = np.empty(sample_size)
    for idx in np.arange(sample_size):
        velocities[idx] = generate_maxw(temperature, mass)

    assert np.isclose(np.mean(velocities), 0.0)
    from scipy.constants import elementary_charge as q

    assert np.isclose(np.std(velocities), np.sqrt(q * temperature / mass))


def test_velocity_maxw_flux():
    from pic.functions import velocity_maxw_flux

    temperature = 1
    mass = 0.1
    v = velocity_maxw_flux(temperature, mass)

    assert type(v) is np.float64


def test_max_vect():
    from pic.functions import max_vect

    temperature = 5
    mass = 0.1
    number = 1000

    v = max_vect(number, temperature, mass)
    assert type(v) is np.ndarray
    assert type(v[1]) is np.float64
    assert len(v) == number
    assert np.isclose(np.mean(v), 0.0)
    from scipy.constants import elementary_charge as q

    assert np.isclose(np.std(v), np.sqrt(q * temperature / mass))
    print(np.std(v))


@pytest.mark.parametrize(
    "position, result",
    [
        (0.15, [0.0, 0.5, 0.5, 0.0]),
        (0.00, [2.0, 0.0, 0.0, 0.0]),
        (0.10, [0.0, 1.0, 0.0, 0.0]),
        (0.29, [0.0, 0.0, 0.1, 1.8]),
        (0.4, [0.0, 0.0, 0.0, 2.0]),
    ],
)
def test_numba_return_part_diag(position, result):
    from pic.functions import numba_return_part_diag

    Np = 1
    partx = np.array([position])
    partv = np.array([1.0])
    tabx = np.array([0, 0.1, 0.2, 0.3])
    diag = np.zeros_like(tabx)
    dx = 0.1
    power = 0

    diag = numba_return_part_diag(Np, partx, partv, tabx, diag, dx, power)

    np.testing.assert_allclose(diag, np.array(result))


@pytest.mark.parametrize(
    "position, vitesse, result",
    [
        ([0.0, 0.5, 0.5, 0.0], [np.ones(3)] * 4, (2, 0)),
        ([-1.0, 4.0, 2.0, 1.0], [np.ones(3)] * 4, (1, 1)),
        ([1.0, 1.0, 1.0, 1.0], [np.ones(3)] * 4, (0, 0)),
        ([2.0, 3.0, 1.0, 0.9], [np.ones(3)] * 4, (0, 1)),
        ([0.0, 3.0, 4.0, -1.0], [np.ones(3)] * 4, (2, 2)),
    ],
)
def test_popout(position, vitesse, result):
    from pic.functions import popout

    x = np.array(position)
    V = np.array(vitesse)
    val = 3
    left, right = popout(x, V, val, True)
    assert (left, right) == result


@pytest.mark.parametrize(
    "position, result",
    [
        (0.15, [[0.0] * 3, [0.5] * 3, [0.5] * 3, [0.0] * 3]),
        (0.00, [[2.0] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3]),
        (0.10, [[0.0] * 3, [1.0] * 3, [0.0] * 3, [0.0] * 3]),
        (0.29, [[0.0] * 3, [0.0] * 3, [0.1] * 3, [1.8] * 3]),
        (0.4, [[0.0] * 3, [0.0] * 3, [0.0] * 3, [2.0] * 3]),
    ],
)
def test_particle_to_grid_u(position, result):
    Np = 1
    partx = np.array([position])
    partv = np.array([[1.0, 1.0, 1.0]]).T
    tabx = np.array([0, 0.1, 0.2, 0.3])
    diag = np.zeros((4, 3))
    dx = 0.1

    diag = particle_to_grid(Np, partx, partv, tabx, diag, dx)

    np.testing.assert_allclose(diag, np.array(result))


def test_tensor_2():
    from pic.functions import tensor_2

    a = np.array([np.arange(3), np.arange(3)], dtype=np.float64)
    b = tensor_2(a)
    np.testing.assert_array_equal(
        b,
        np.array(
            [[[0, 0, 0], [0, 1, 2], [0, 2, 4]], [[0, 0, 0], [0, 1, 2], [0, 2, 4]]]
        ),
    )


def test_remove():
    from pic.functions import __remove_jit

    Idxs = List([3, 2])
    x = np.arange(5, dtype="float64")
    V = np.ones((5, 3), dtype="float64")
    assert 3 == __remove_jit(Idxs, x, V, 5)
    np.testing.assert_array_equal(x, np.array([0.0, 1.0, 4.0, -1.0, -1.0]))


def test_remove_random():
    from pic.functions import remove_random

    x = np.arange(5, dtype="float64")
    V = np.ones((5, 3), dtype="float64")
    remove_random(x, V, 5, 2)
    np.testing.assert_array_equal(x[3:], np.array([-1.0, -1.0]))


def test_np_sort():
    n = 100
    n_cells = 10
    dx = 1.0 / n_cells
    x = np.random.rand(n)
    V = np.array([x, x, x]).T
    p = (x / dx).astype(int).argsort()
    V[:] = V[p]
    x[:] = x[p]
    assert np.all(np.floor(x[:-1] / dx) <= np.floor(x[1:] / dx))
    np.testing.assert_array_almost_equal(x, V[:, 0])


def test_interp():
    lx = 1
    Npart = 20
    nx = 100
    N_cells = nx + 1

    x_j = np.arange(0, N_cells, dtype="float64") * lx / (nx)
    dx = x_j[1]
    lx = x_j[-1]
    x = np.linspace(0, lx, Npart)
    field = np.cos(x_j)
    result = numba_interp1D_normed(x / dx, field)
    np.testing.assert_allclose(result, np.cos(x), rtol=1e-4)
