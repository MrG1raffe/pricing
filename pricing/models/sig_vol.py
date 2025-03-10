import numpy as np
from numpy.typing import NDArray
from numpy import float64, complex128
from numba import jit
from dataclasses import dataclass
from typing import Union, Tuple
from math import ceil

# TODO: remove second "signature" for usage
from signature.signature.tensor_sequence import TensorSequence
from signature.signature.tensor_algebra import TensorAlgebra, Alphabet
from signature.signature.shuffle_operator import ShuffleOperator
from simulation.diffusion import Diffusion
from simulation.utility import DEFAULT_SEED, to_numpy
from .characteristic_function_model import CharacteristicFunctionModel
from .monte_carlo_model import MonteCarloModel


@dataclass
class SigVol(CharacteristicFunctionModel, MonteCarloModel):
    vol_ts: TensorSequence
    ta: TensorAlgebra
    rho: float

    def characteristic_function(
        self,
        T: float,
        x: float,
        u1: complex,
        cf_timestep: float = 0.001,
        **kwargs
    ) -> Union[complex128, NDArray[complex128]]:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * X_T}]     (1)

        for the given model, where X_t = F_t if `model_type` == "normal" and
        X_t = log(F_t) if `model_type` == "log-normal".

        :param u1: X_T coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: X_0, equals to F_0 if `model_type` == "normal" and to log(F_0) if `model_type` == "log-normal".
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        u_arr = to_numpy(1j * u1)

        u_shape = u_arr.shape
        u_arr = u_arr.flatten()

        timestep = min(cf_timestep, T / 10)
        t_grid = np.linspace(0, T, ceil(T / timestep) + 1)
        res = jit_char_func(t_grid=t_grid, u_arr=u_arr, vol_ts=self.vol_ts, shuop=self.ta.shuop, rho=self.rho)
        res *= np.exp(u_arr * x)
        return np.reshape(res, u_shape)

    def get_vol_trajectory(
        self,
        t_grid: NDArray[float64],
        size: int,
        rng: np.random.Generator = None,
        B_traj: NDArray[float64] = None,
        return_sig: bool = False
    ) -> NDArray[float64]:
        """
        Simulate the variance and the factor processes on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param rng: random number generator to simulate the trajectories with.
        :param B_traj: pre-simulated trajectories of the BM B_t corresponding to the stochastic volatility.
            By default, None, and will be simulated within the function.
        :return: an array of shape (size, len(t_grid)) with the volatility trajectories.
        """
        if B_traj is None:
            # simulation of B_traj
            diffusion = Diffusion(t_grid=t_grid, dim=1, size=size, rng=rng)
            B_traj = diffusion.brownian_motion()[:, 0, :]  # shape (size, len(t_grid))
        else:
            if B_traj.shape != (size, len(t_grid)):
                raise ValueError("Inconsistent dimensions of B_traj were given.")

        path = np.zeros((t_grid.size, 2, size))
        path[:, 0, :] = np.reshape(t_grid, (-1, 1))
        path[:, 1, :] = B_traj.T
        B_Sig = self.ta.path_to_sequence(path=path, trunc=self.vol_ts.trunc)
        if return_sig:
            return np.real(B_Sig @ self.vol_ts).T, B_Sig
        return np.real(B_Sig @ self.vol_ts).T

    def get_price_trajectory(
        self,
        t_grid: NDArray[float64],
        size: int,
        F0: Union[float, NDArray[float64]],
        rng: np.random.Generator = None,
        return_vol: bool = False,
        return_bm: bool = False,
        return_sig: bool = False,
        **kwargs
    ) -> Union[NDArray[float64], Tuple[NDArray[float64], ...]]:
        """
        Simulates the underlying price trajectories on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price.
        :param rng: random number generator to simulate the trajectories with.
        :param return_vol: whether to return the volatility trajectory together with the prices.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories,
            an array `sigma_traj` of shape (size, len(t_grid)) of volatility trajectories if `return_vol` == True.
        """

        if rng is None:
            rng = np.random.default_rng(seed=DEFAULT_SEED)

        diffusion = Diffusion(t_grid=t_grid, dim=2, size=size, rng=rng)

        corr_mat = np.array([[1, self.rho],
                             [self.rho, 1]])

        brownian_motion = diffusion.brownian_motion(correlation=corr_mat)

        dt = np.diff(t_grid)
        dW_traj = np.diff(brownian_motion[:, 0, :], axis=1)  # shape (size, n_hist_factors, len(t_grid)-1)
        B_traj = brownian_motion[:, 1, :]                    # shape (size, len(t_grid))

        vol_traj, B_Sig = self.get_vol_trajectory(t_grid=t_grid, size=size, B_traj=B_traj, return_sig=True)

        log_F_traj = np.log(F0) * np.ones((size, len(t_grid)))
        log_F_traj[:, 1:] += np.cumsum(dW_traj * vol_traj[:, :-1] - 0.5 * dt * vol_traj[:, :-1]**2, axis=1)
        F_traj = np.exp(log_F_traj)

        result = (F_traj,)
        if return_vol:
            result = result + (vol_traj,)
        if return_bm:
            result = result + (brownian_motion,)
        if return_sig:
            result = result + (B_Sig,)

        return result


@jit(nopython=True)
def jit_char_func(
    t_grid: NDArray[float64],
    u_arr: NDArray[complex128],
    vol_ts: TensorSequence,
    shuop: ShuffleOperator,
    rho: float
) -> NDArray[complex128]:

    trunc = vol_ts.trunc
    dt = np.diff(t_grid)

    alphabet = Alphabet(2)
    psi = TensorSequence(alphabet, trunc, np.zeros((alphabet.number_of_elements(trunc), u_arr.size)))
    psi_pred = TensorSequence(alphabet, trunc, np.zeros((alphabet.number_of_elements(trunc), u_arr.size)))

    u_arr = np.reshape(u_arr, (1, u_arr.size, 1))

    vol_shuffle_squared = shuop.shuffle_prod(vol_ts, vol_ts) * (0.5 * (u_arr ** 2 - u_arr))
    for i in range(len(dt)):
        f_psi = jit_riccati_func(psi, vol_ts, vol_shuffle_squared, shuop, u_arr, rho)
        f_psi_pred = jit_riccati_func(psi_pred, vol_ts, vol_shuffle_squared, shuop, u_arr, rho)

        psi_pred.update(psi + f_psi * dt[i])
        psi.update(psi + (f_psi_pred + f_psi) * (dt[i] / 2))

    return np.exp(psi[""][:, 0])


@jit(nopython=True)
def jit_riccati_func(
    psi: TensorSequence,
    vol_ts: TensorSequence,
    vol_shuffle_squared: TensorSequence,
    shuop: ShuffleOperator,
    u_arr: NDArray[complex128],
    rho: float
):
    psi_proj_2 = psi.proj("2")
    return shuop.shuffle_prod_2d(psi_proj_2, psi_proj_2 / 2 + vol_ts * (u_arr * rho)) + \
           psi.proj("22") / 2 + psi.proj("1") + vol_shuffle_squared
