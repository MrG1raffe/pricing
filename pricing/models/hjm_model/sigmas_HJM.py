import numpy as np
from numpy.typing import NDArray
from numpy import float_
from typing import Callable
from scipy.interpolate import interp1d
from math import ceil

from models.model_params import HistoricalParams
from models.hjm_model.contracts import ForwardContract
from utility.utility import to_numpy


def forward_vol_from_hist_params(
    hist_params: HistoricalParams,
    contract: ForwardContract,
    T_grid: NDArray[float_],
    g_arr: NDArray[float_],
    h_func: Callable = lambda x: np.ones_like(x),
    is_kemna_vorst: bool = True,
    is_interpolate: bool = True
) -> Callable[[NDArray[float_]], NDArray[float_]]:
    """
    Calculates an HJM deterministic volatility component of the given forward contract.

    :param hist_params:
    :param contract: a forward contract the volatility of which to be calculated.
    :param T_grid: delivery date T integration grid.
    :param g_arr: the values of the function g on the integration grid T_grid.
    :param h_func: function h as a callable object.
    :param is_kemna_vorst: whether to use the Kemna-Vorst approximation or create a three-dimensional function sigmas
        with the volatilities of the instantaneous forwards with delivery T for T in T_grid.
    :param is_interpolate: whether to use the linear interpolation of sigmas instead of the function itself.
        May be useful for fine T-grids to accelerate the evaluation of sigma for pricing.
    :return: a callable object `sigmas` which is the deterministic volatility of the forward.
    """
    T_s, T_e = contract.time_to_delivery_start, contract.time_to_delivery_end
    inf_tau_idx = np.isinf(hist_params.taus)
    taus = np.ones_like(hist_params.taus)
    taus[~inf_tau_idx] = hist_params.taus[~inf_tau_idx]
    if np.isclose(T_s, T_e):
        def sigmas(t):
            t_grid = to_numpy(t)
            inst_vols = np.mean(g_arr) * get_instantaneous_vols(hist_params=hist_params, t_grid=t_grid, T_grid=np.array([T_s]))
            return inst_vols[:, :, 0]  * np.reshape((t_grid <= T_s) * h_func(t_grid), (-1, 1))
    else:
        if not is_kemna_vorst:
            def sigmas(t):
                t_grid = to_numpy(t)
                inst_vols = get_instantaneous_vols(hist_params=hist_params, t_grid=t_grid, T_grid=T_grid)
                res = np.einsum('j,kij -> kij',g_arr, inst_vols)
                res *= np.reshape((t_grid <= T_s) * h_func(t_grid), (t_grid.size, 1, 1))
                return res
        else:
            def sigmas(t):
                t_grid = to_numpy(t)
                int_vols = get_integrated_vols(hist_params=hist_params, t_grid=t_grid, T_grid=T_grid)
                res = np.einsum("j,kij -> ki",g_arr, int_vols) / (T_e - T_s)
                res *= np.reshape((t_grid <= T_s) * h_func(t_grid), (t_grid.size,  1))
                return res
            if is_interpolate:
                dt = 0.001
                t_grid = np.linspace(0, T_s, ceil(T_s / dt))
                sigmas = interp1d(x=t_grid, y=sigmas(t_grid), axis=0, bounds_error=False, fill_value=0)
    return sigmas

def get_integrated_vols(hist_params: HistoricalParams, t_grid: NDArray[float_], T_grid: NDArray[float_]):
    n_L, n_S, n_C = hist_params.lsc_factors["nber_l_factors"], hist_params.lsc_factors["nber_s_factors"], hist_params.lsc_factors["nber_c_factors"]
    t_grid = np.reshape(t_grid, (-1, 1, 1))
    T_grid_1 = np.reshape(T_grid[:-1], (1, 1, -1))
    T_grid_2 = np.reshape(T_grid[1:], (1, 1, -1))
    sigmas = np.reshape(hist_params.sigmas, (1, -1, 1))
    taus = np.reshape(hist_params.taus, (1, -1, 1))
    int_vols = np.zeros((t_grid.size, hist_params.sigmas.size, T_grid_2.size)) # kij
    int_vols[:, :n_L, :] = sigmas[:, :n_L] * (T_grid_2 - T_grid_1)
    int_vols[:, n_L:, :] = taus * sigmas[:, n_L:] * (np.exp(-(T_grid_1 - t_grid) / taus) - np.exp(-(T_grid_2 - t_grid) / taus))
    int_vols[:, n_L + n_S:, :] += taus[:, n_S:] * sigmas[:, n_L + n_S:] * ((T_grid_1 - t_grid) / taus[:, n_S:] * np.exp(-(T_grid_1 - t_grid) / taus[:, n_S:]) -
                                                                           (T_grid_2 - t_grid) / taus[:, n_S:] * np.exp(-(T_grid_2 - t_grid) / taus[:, n_S:]))
    return int_vols

def get_instantaneous_vols(hist_params: HistoricalParams, t_grid: NDArray[float_], T_grid: NDArray[float_]):
    n_L, n_S, n_C = hist_params.lsc_factors["nber_l_factors"], hist_params.lsc_factors["nber_s_factors"], hist_params.lsc_factors["nber_c_factors"]
    t_grid = np.reshape(t_grid, (-1, 1, 1))
    T_grid = np.reshape(T_grid, (1, 1, -1))
    sigmas = np.reshape(hist_params.sigmas, (1, -1, 1))
    taus = np.reshape(hist_params.taus, (1, -1, 1))
    inst_vols = np.zeros((t_grid.size, hist_params.sigmas.size, T_grid.size)) # kij
    inst_vols[:, :n_L, :] = sigmas[:, :n_L]
    inst_vols[:, n_L:, :] = sigmas[:, n_L:] * np.exp(-(T_grid - t_grid) / taus)
    inst_vols[:, n_L + n_S:, :] = sigmas[:, n_L + n_S:] * (T_grid - t_grid) / taus[:, n_S:] * np.exp(-(T_grid - t_grid) / taus[:, n_S:])
    return inst_vols

