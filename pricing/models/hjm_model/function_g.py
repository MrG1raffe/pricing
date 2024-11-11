import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import float_, int_
from typing import List

from models.hjm_model.contracts import ForwardContract
from utility.piece_wise_constant import PieceWiseConstantFunction

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class FunctionG(PieceWiseConstantFunction):
    function_shape: NDArray[float_]
    contract_indices: dict
    observation_date: np.datetime64

    def __init__(
        self,
        contracts: List[ForwardContract],
        interpolation_method: str = None,
        fix_level_at_zero: bool = False
    ):
        """
        Creates a piece-wise constant function g equal to 1 using the time delivery start / end dates of
        the contracts as changing points.

        :param contracts: list of contracts used to construct the grid of changing points.
        :param interpolation_method: interpolation method that will be used to smooth the peace-wise constant function.
        """
        if not len(contracts):
            self.changing_points = np.array([0, 1])
            self.function_values = np.ones(1)
            self.contract_indices = dict()
            self.fix_level_at_zero = fix_level_at_zero
            self.delivery_points = self.changing_points
            self.observation_date = None

            super().__init__(changing_points=self.changing_points,
                             function_values=self.function_values,
                             interpolation_method=interpolation_method,
                             extrapolation_left=None,
                             extrapolation_right=1)
            return

        observation_date = contracts[0].observation_date

        max_date = max([contract.delivery_end_date for contract in contracts]) + np.timedelta64(30, "D")
        max_changing_point = (max_date - observation_date) / np.timedelta64(365, 'D')
        changing_points = [0, max_changing_point]
        changing_points_dates = [np.datetime64(observation_date), np.datetime64(max_date)]
        for contract in contracts:
            if contract.observation_date != observation_date:
                raise ValueError("All contracts should have the same observation date.")
            if np.datetime64(contract.delivery_start_date) not in changing_points_dates:
                changing_points_dates.append(np.datetime64(contract.delivery_start_date))
                changing_points.append(contract.time_to_delivery_start)
            if np.datetime64(contract.delivery_end_date) not in changing_points_dates:
                changing_points_dates.append(np.datetime64(contract.delivery_end_date))
                changing_points.append(contract.time_to_delivery_end)

        changing_points, changing_points_dates = zip(*sorted(zip(changing_points, changing_points_dates)))
        self.delivery_points = np.array(changing_points)
        changing_points = np.array(changing_points)

        # i-th value corresponds to t between i-th and (i+1)-th changing points.
        function_values = np.ones_like(changing_points[:-1])
        super().__init__(changing_points=changing_points,
                         function_values=function_values,
                         interpolation_method=interpolation_method,
                         extrapolation_left=None,
                         extrapolation_right=1)

        self.contract_indices = dict()
        for contract in contracts:
            idx_start = changing_points_dates.index(contract.delivery_start_date)
            if idx_start == 1 and not fix_level_at_zero:
                # In this case, the function g will be extrapolated to the first interval and not set equal to 1.
                idx_start = 0
            idx_end = changing_points_dates.index(contract.delivery_end_date)
            self.contract_indices[contract.name] = np.arange(idx_start, idx_end + 1)

        self.fix_level_at_zero = fix_level_at_zero
        self.observation_date = observation_date

    def __getitem__(
        self,
        key: str
    ) -> NDArray[float_]:
        """
        Returns the indices of changing points corresponding to the contract name `key`.

        :param key: contract name.
        :return: list of indices corresponding to the intervals forming the delivery period of the contract "key"
            which was used to construct the changing points grid.
        """
        if key in self.contract_indices.keys():
            return self.function_values[self.contract_indices[key][:-1]]
        else:
            return self.function_values[-2:-1]

    def __setitem__(
        self,
        key: str,
        value: float
    ) -> None:
        """
        Set the `value` on the interval corresponding to delivery period of the contract `key`.

        :param key: contract name.
        :param value: value of function g to be set.
        :return:
        """
        self.function_values[self.contract_indices[key][:-1]] = value

    def get_contract_changing_points(
        self,
        contract_name: str
    ) -> NDArray[float_]:
        """
        Returns the changing points corresponding to the contract "contract_name".

        :param contract_name: contract name.
        :return: the changing points T1, ..., Tn, where T1 coincides with time to delivery start and Tn coincides
            with time ti delivery end for the given contract.
        """
        return self.changing_points[self.contract_indices[contract_name]]

    def get_contract_indices(
        self,
        contract_name: str
    ) -> NDArray[int_]:
        """
        Calculates the indices of the intervals forming the repartition of the contract's delivery period.

        :param contract_name: name of the contract.
        :return: list of indices of the intervals forming the delivery period of the contract.
        """
        return self.contract_indices[contract_name][:-1]

    def plot(
        self,
        ax: plt.axes = None
    ) -> None:
        """
        Plots the function on a given axis.

        :param ax: matplotlib axis.
        """
        if self.observation_date is not None:
            xticks = self.delivery_points
            xlabels = (self.observation_date + (xticks * 365).astype(int)).astype(str)
        else:
            xticks = None
            xlabels = None
        title = r"Function $g(T)$"
        self._plot_function(ax=ax, xlabels=xlabels, xticks=xticks, title=title)
