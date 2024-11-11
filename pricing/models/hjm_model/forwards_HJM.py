import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy import float_
from typing import List, Union, Callable, Tuple
from dataclasses import asdict, dataclass
from abc import ABCMeta
from copy import deepcopy
from math import ceil

from models.model import Model
from models.lifted_heston import LiftedHeston
from models.stein_stein import SteinStein
from models.hjm_model.contracts import ForwardContract, is_strictly_included
from models.hjm_model.function_g import FunctionG
from models.hjm_model.function_h import FunctionH
from models.hjm_model.sigmas_HJM import forward_vol_from_hist_params
from models.model_params import HistoricalParams, ModelParams
from products.vanilla_option import VanillaOption
from utility.utility import from_pseudo_delta_to_strike


class ForwardsHJM:
    hist_params: HistoricalParams
    contracts: List[ForwardContract]
    contract_names: List[str]
    inclusion_dict: dict
    models: Union[List[LiftedHeston], List[SteinStein]]   # list of models corresponding to the contracts.
    observation_date: np.datetime64
    vanilla_options: List[VanillaOption]
    smiles_to_calib: List[int]
    g_function: FunctionG
    h_function: FunctionH
    g_indices: List[int]
    h_indices: List[int]
    interpolation_method: str = "pchip"
    is_kemna_vorst: bool = True
    kemna_vorst_int_grids: NDArray[float_] = None

    def __init__(
        self,
        hist_params: HistoricalParams,
        stoch_vol_params: ModelParams,
        model_constructor: ABCMeta,
        contracts: List[ForwardContract],
        vanilla_options: List[VanillaOption],
        smiles_to_calib: List[int] = None,
        h_indices: List[int] = None,
        kemna_vorst_int_grids: NDArray[float_] = None,
        interpolation_method: str = "pchip",
        fix_g_at_zero: bool = False
    ) -> None:
        """
        A class that allows to create pricing models for forward contracts  based on the historical calibration
        parameters and the parameters of a common stochastic volatility model.

        :param hist_params: parameters corresponding to the historical calibration of the HJM curve.
        :param stoch_vol_params: stochastic volatility model parameters to be given to the model constructor.
        :param model_constructor: stochastic volatility model constructor to create the model objects.
            Model class should be inherited from the class `Model`.
        :param contracts: list of forward contracts of time ForwardContract.
        :param vanilla_options: list of vanilla options corresponding to the given contracts. Used to define the
            grids for the functions g and h.
        :param smiles_to_calib: indices of smiles to be taken into account during the calibration.
        :param h_indices: indices of options to be calibrated with the function h. By default, includes all
            the smiles with CAL, SUM, and WIN contracts as underlying.
        :param kemna_vorst_int_grids: if is not None, Kemna-Vorst approximation is not used, and the grids are used
            to define 3-dimensional function sigmas in the pricing model.
        :param interpolation_method: an interpolation method used to smooth the functions g and h.
        :param fix_g_at_zero: whether to fix g equal to 1 on the interval between zero and the first delivery period.
            If False, extrapolates the value from this delivery period.
        """
        self.hist_params = hist_params
        self.contracts = contracts
        self.contract_names = [contract.name for contract in contracts]

        self.vanilla_options = vanilla_options
        if smiles_to_calib is None:
            # Calibrate all smile until the opposite is specified
            self.smiles_to_calib = np.arange(len(self.vanilla_options))
        else:
            self.smiles_to_calib = smiles_to_calib

        self.observation_date = self.contracts[0].observation_date
        for contract in contracts:
            if contract.observation_date != self.observation_date:
                raise ValueError("All contracts should have the same observation date.")

        self.kemna_vorst_int_grids = kemna_vorst_int_grids
        self.is_kemna_vorst = (self.kemna_vorst_int_grids is None)

        if h_indices is None:
            self.h_indices = []
            for i, option in enumerate(self.vanilla_options):
                underlying = option.underlying_name.upper()
                if underlying[:3] in {"CAL", "SUM", "WIN"} and i in self.smiles_to_calib:
                    self.h_indices.append(i)
        else:
            self.h_indices = h_indices
        # The smiles not calibrated with the function h are calibrated with the function g.
        self.g_indices = [i for i in self.smiles_to_calib if i not in self.h_indices]
        self.__initialize_inclusion_dict()

        # Initialize g(T) and h(t)
        self.g_function = FunctionG(
            contracts=contracts,
            # contracts=[contract for contract in contracts
            #            if contract.name in [self.vanilla_options[i].underlying_name for i in self.g_indices]],
            interpolation_method=interpolation_method,
            fix_level_at_zero=fix_g_at_zero
        )
        self.h_function = FunctionH(
            vanilla_options_with_indices=[(i, self.vanilla_options[i]) for i in self.h_indices],
            observation_date=self.observation_date,
            interpolation_method=interpolation_method
        )
        self.interpolation_method = interpolation_method

        self.models = []
        for i, contract in enumerate(self.contracts):
            dt = 1 / 365
            T_grid = np.linspace(contract.time_to_delivery_start,
                                 contract.time_to_delivery_end,
                                 ceil((contract.time_to_delivery_end - contract.time_to_delivery_start) / dt))
            g_arr = self.g_function(t=T_grid[:-1])

            if self.is_kemna_vorst:
                sigmas = forward_vol_from_hist_params(
                    hist_params=hist_params,
                    contract=contract,
                    g_arr=g_arr,
                    h_func=lambda t: self.h_function(t),
                    T_grid=T_grid,
                    is_kemna_vorst=self.is_kemna_vorst,
                )
                model = model_constructor(
                    sigmas=sigmas,
                    R=hist_params.corr_mat,
                    is_kemna_vorst=self.is_kemna_vorst,
                    **asdict(stoch_vol_params)
                )
            else:
                T_grid = self.kemna_vorst_int_grids[i][:-1]
                g_arr = self.g_function(t=T_grid)
                sigmas = forward_vol_from_hist_params(
                    hist_params=hist_params,
                    contract=contract,
                    g_arr=g_arr,
                    h_func=self.h_function,
                    T_grid=T_grid,
                    is_kemna_vorst=self.is_kemna_vorst
                )
                model = model_constructor(
                    sigmas=sigmas,
                    R=hist_params.corr_mat,
                    is_kemna_vorst=self.is_kemna_vorst,
                    kemna_vorst_grid=self.kemna_vorst_int_grids[i],
                    **asdict(stoch_vol_params)
                )
            self.models.append(model)

    def __initialize_inclusion_dict(self) -> None:
        """
        Initializes the dictionary self.inclusion_dict: its keys consist in names of the contracts corresponding to
        smiles in `self.g_indices` which delivery period covers the delivery period of another contracts corresponding
        to smiles in `self.g_indices`, and values contain the names of contracts being covered.
        """
        self.inclusion_dict = dict()
        for option_idx_1 in self.g_indices:
            dependent_options = [
                option_idx_2 for option_idx_2 in self.g_indices
                if is_strictly_included(self.vanilla_options[option_idx_2].underlying_name,
                                        self.vanilla_options[option_idx_1].underlying_name)
            ]
            if dependent_options:
                self.inclusion_dict[option_idx_1] = dependent_options

    def update_sigmas(
        self,
        is_interpolate: bool = True,
        models: List[Model] = None,
        contracts: List[ForwardContract] = None
    ) -> None:
        """
        Updates the deterministic volatility `sigmas` in all models if the functions g or h were changed.

        :param is_interpolate: whether to use the linear interpolation of sigmas instead of the function itself.
            May be useful for fine T-grids to accelerate the evaluation of sigma for pricing.
        """
        if models is None:
            models = self.models
            contracts = self.contracts
        for i, (model, contract) in enumerate(zip(models, contracts)):
            dt = 1 / 365
            T_grid = np.linspace(contract.time_to_delivery_start,
                                 contract.time_to_delivery_end,
                                 ceil((contract.time_to_delivery_end - contract.time_to_delivery_start) / dt))
            g_arr = self.g_function(t=T_grid[:-1])

            if self.is_kemna_vorst:
                sigmas = forward_vol_from_hist_params(
                    hist_params=self.hist_params,
                    contract=contract,
                    g_arr=g_arr,
                    h_func=self.h_function,
                    T_grid=T_grid,
                    is_kemna_vorst=self.is_kemna_vorst,
                    is_interpolate=is_interpolate
                )
            else:
                T_grid = self.kemna_vorst_int_grids[i][:-1]
                g_arr = self.g_function(t=T_grid)
                sigmas = forward_vol_from_hist_params(
                    hist_params=self.hist_params,
                    contract=contract,
                    g_arr=g_arr,
                    h_func=self.h_function,
                    T_grid=T_grid,
                    is_kemna_vorst=self.is_kemna_vorst,
                    is_interpolate=is_interpolate
                )
            model.update_params({"sigmas": sigmas})

    def update_stoch_vol_params(
        self,
        model_params_dict: dict
    ) -> None:
        """
        Updates stochastic volatility parameters for all models.

        :param model_params_dict: dictionary with parameters names as keys and parameters values as values.
        """
        for model in self.models:
            model.update_params(model_params_dict)

    def update_hist_params(self, taus, sigmas, R) -> None:
        """
        Updates historical volatility parameters for all models.

        :param taus: vector of taus.
        :param sigmas: vector of sigmas.
        :param R: correlation matrix.
        """
        self.hist_params.taus = taus
        self.hist_params.sigmas = sigmas
        self.hist_params.corr_mat = R

        self.update_stoch_vol_params({"R": R})
        self.update_sigmas()


    def integrated_variance_without_gh(
        self,
        contract: ForwardContract,
        T: float,
        taus: NDArray[float_] = None,
        sigmas: NDArray[float_] = None,
        R: NDArray[float_] = None
    ) -> float:
        """
        Calculates explicitly the integrated variance of the futures contract price until `T` using the explicit
        formula with h = g = 1.

        :param contract: contract which integrated variance should be calculated.
        :param T: the date (float) the integrated variance to be calculated on.
        :return: the integrated variance from 0 to T.
        :param taus: taus parameters from the historical params.
        :param sigmas: sigmas parameters from the historical params.
        :param R: correlation matrix from the historical params.
        """
        def exp_int(x):
            return (1 - np.exp(-x)) / x

        delta = contract.time_to_delivery_end - contract.time_to_delivery_start
        if taus is None:
            taus = self.hist_params.taus
        if sigmas is None:
            sigmas = self.hist_params.sigmas
        if R is None:
            R = self.hist_params.corr_mat

        y = sigmas * exp_int(delta / taus)
        inv_taus_mat = (1 / taus[:, None] + 1 / taus[None, :])
        T_s = contract.time_to_delivery_start
        A = R * (np.exp(-inv_taus_mat * (T_s - T)) - np.exp(-inv_taus_mat * T_s)) / inv_taus_mat
        return y.T @ A @ y

    def get_instantaneous_correlation(
        self,
        contract_idx_1: int,
        contract_idx_2: int,
        t_grid: NDArray[float_] = None
    ) -> Tuple[NDArray[float_], ...]:
        """
        Calculates the instantaneous correlations between two forward contracts with indices
        `contract_idx_1` and `contract_idx_2` on a given time grid `t_grid`.
        If time grid is not given, it is constructed automatically as a linspace between 0 and the
        smallest time too delivery start.

        :param contract_idx_1: index of the first contract.
        :param contract_idx_2: index of the second contract.
        :param t_grid: time grid.
        :return: the time grid and the instantaneous correlations on this time grid.
        """
        T = min(self.contracts[contract_idx_1].time_to_delivery_start,
                self.contracts[contract_idx_2].time_to_delivery_start)
        if t_grid is None:
            t_grid = np.linspace(0, T, 1000)
        sigmas1 = self.models[contract_idx_1].sigmas(t_grid)
        sigmas2 = self.models[contract_idx_2].sigmas(t_grid)
        R = self.hist_params.corr_mat
        spot_vol_1 = np.sqrt(np.sum(sigmas1.T * (R @ sigmas1.T), 0))
        spot_vol_2 = np.sqrt(np.sum(sigmas2.T * (R @ sigmas2.T), 0))
        rho = np.sum(sigmas1.T * (R @ sigmas2.T), 0) / spot_vol_1 / spot_vol_2
        return t_grid, rho

    def get_model_for_contract(
        self,
        contract: ForwardContract,
        dT: float = 1 / 365
    ) -> Model:
        """
        Creates a pricing model for a forward contract not necessarily presented in self.contracts.

        :param contract: an object of type ForwardContract
        :param dT: time-step used for the integration grid in T to calculate sigmas.
        :return: the pricing model of type Model describing the dynamics of the contract in the HJM model.
        """
        T_s, T_e = contract.time_to_delivery_start, contract.time_to_delivery_end
        if np.isclose(T_s, T_e):
            T_grid = np.array([T_s])
            g_arr = self.g_function(T_grid)
        else:
            T_grid = np.linspace(T_s, T_e, np.ceil((T_e - T_s) / dT).astype(int) + 1)
            g_arr = self.g_function(T_grid)[:-1]
        sigmas = forward_vol_from_hist_params(
            hist_params=self.hist_params,
            contract=contract,
            g_arr=g_arr,
            h_func=self.h_function,
            T_grid=T_grid,
            is_kemna_vorst=self.is_kemna_vorst,
            is_interpolate=True
        )
        model = deepcopy(self.models[0])
        model.update_params({"sigmas": sigmas})
        return model

    def get_hjm_without_g(self):
        """
        Creates a copy of the class with the function g set equal to 1.

        :return: a copy of self with g = 1.
        """
        hjm_copy = deepcopy(self)
        hjm_copy.g_function.function_values = np.ones_like(hjm_copy.g_function.function_values)
        hjm_copy.update_sigmas()
        return hjm_copy

    @staticmethod
    def get_maturity_from_contract_and_option_name(
        contract: ForwardContract,
        option_name: str
    ) -> float:
        T = contract.time_to_delivery_start - 5 / 365
        if option_name.startswith("Cal"):
            mon_exp = option_name[-3:]
            if mon_exp == "Sep":
                T -= 3 * 30 / 365
            if mon_exp == "Jun":
                T -= 6 * 30 / 365
            if mon_exp == "Mar":
                T -= 9 * 30 / 365
        return T

    def _smile_idx_to_contract_idx(self, i: int):
        """
        Transforms the vanilla option index into the index of its underlying contract.

        :param i: index if option is self.vanilla_options.
        :return: index of contract in self.contracts corresponding to the underlying of the i-th option.
        """
        return self.contract_names.index(self.vanilla_options[i].underlying_name)

    def get_implied_vols_for_options(
        self,
        option_names_list: List[str],
        pseudo_deltas: Tuple = (-0.1, -0.25, 0.5, 0.25, 0.1),
        is_over_vol: bool = False,
        atm_idx: int = 2,
        options_extrapolated: Tuple = ()
    ) -> pd.DataFrame:
        F0 = 100
        smiles_hjm = []
        for option_name in option_names_list:
            contract = ForwardContract(name=option_name[:6], observation_date=self.observation_date, F0=F0)
            model = self.get_model_for_contract(contract)
            T = self.get_maturity_from_contract_and_option_name(contract=contract, option_name=option_name)
            vol_atm = VanillaOption(T=T, K=np.array([F0])).get_price(model=model, method="lewis",
                                                                     is_vol_surface=True, F0=F0)
            option = VanillaOption(
                T=T,
                K=from_pseudo_delta_to_strike(deltas=np.array(pseudo_deltas), F0=F0, sigma=vol_atm, ttm=T),
                underlying_name=contract.name
            )
            smile_hjm = option.get_price(model=model, method="lewis", is_vol_surface=True, F0=F0)
            smiles_hjm.append(smile_hjm)

        smiles_hjm = np.array(smiles_hjm)
        if is_over_vol:
            df_hjm = pd.DataFrame(smiles_hjm - smiles_hjm[:, atm_idx].reshape((-1, 1)),
                                  columns=[delta if delta != 0.5 else "ATM" for delta in pseudo_deltas],
                                  index=option_names_list)
            df_hjm.ATM = smiles_hjm[:, atm_idx].reshape((-1, 1))
        else:
            df_hjm = pd.DataFrame(smiles_hjm, columns=[delta if delta != 0.5 else "ATM" for delta in pseudo_deltas],
                                  index=option_names_list)

        # Function rows being extrapolated
        def highlight_rows(s):
            if s.name in options_extrapolated:
                return ['background-color: cyan'] * len(s)
            else:
                return [''] * len(s)

        df_hjm = df_hjm.style.apply(highlight_rows, axis=1)
        return df_hjm


@dataclass
class HJMModelParams:
    """
    Historical and Implicit HJM calibration parameters.
    
    HistoricalParams: Historical HJM calibration params.
    g_function: FunctionG 
    h_function: FunctionH
    stoch_vol_params: Stochastic volatility parameters
    """
    hist_params: HistoricalParams
    model_constructor: ABCMeta
    stoch_vol_params: ModelParams
    g_function: Union[FunctionG, Callable]
    h_function: Union[FunctionH, Callable]
    is_kemna_vorst: bool = True
