from dataclasses import dataclass, asdict
from typing import List
import numpy as np
from numpy.typing import NDArray
from numpy import float64

from ..models.model import Model
from ..market_data.market_data import MarketData
from ..market_data.market_data_vanilla import MarketDataVanilla
from .calibration_params import CalibrationParams
from ..products.vanilla_option import VanillaOption
from .minimize import minimize


@dataclass
class Calibration:
    model: Model
    market_data_array: List[MarketData]
    F0: float
    calibration_parameters: CalibrationParams

    def update_model_parameters(
            self,
            x: NDArray[float64]
    ) -> None:
        """

        """
        # Use it in all calibration losses.
        model_params_dict = self.calibration_parameters.vector_to_params(x)
        self.model.update_params(model_params_dict)

    def loss_iv(self, x):
        """

        """
        loss = 0
        self.update_model_parameters(x)
        for md in self.market_data_array:
            if isinstance(md, MarketDataVanilla) and isinstance(md.product, VanillaOption):
                K_shape = md.product.K.shape
                try:
                    iv_model = np.reshape(
                        md.product.get_price(
                            model=self.model,
                            method=self.calibration_parameters.pricing_method,
                            F0=self.F0,
                            is_vol_surface=True,
                            **asdict(self.calibration_parameters.pricing_params)
                        ),
                        K_shape
                    )[-1]
                except Exception as e:
                    return 1
                iv_mkt = np.reshape(md.implied_volatility, K_shape)[-1]
                loss += np.mean((iv_model - iv_mkt) ** 2)
        return loss

    def loss_price(self, x):
        """

        """
        loss = 0
        self.update_model_parameters(x)
        for md in self.market_data_array:
            if isinstance(md, MarketDataVanilla) and isinstance(md.product, VanillaOption):
                K_shape = md.product.K.shape
                prices_model = np.reshape(
                    md.product.get_price(
                        model=self.model,
                        method=self.calibration_parameters.pricing_method,
                        F0=self.F0,
                        **asdict(self.calibration_parameters.pricing_params)
                    ),
                    K_shape
                )[-1]
                vega = np.reshape(
                    md.product.vega(
                        sigma=md.implied_volatility,
                        F=self.F0
                    ),
                    K_shape
                )[-1]
                prices_mkt = np.reshape(md.price, K_shape)[-1]
                loss += np.mean(((prices_model - prices_mkt) / vega) ** 2)
        return loss

    def calibrate_model(
        self,
        x0: NDArray[float64] = None,
        **kwargs
    ) -> None:
        """
        Performs the calibration.

        :param x0: Initial value. By default, take the initial value given by `self.get_initial_guess`.
        """
        def loss_fun(x):
            return self.loss_iv(x=x)

        if x0 is not None and len(x0) != len(self.calibration_parameters.bounds):
            raise ValueError(f"Inconsistent dimensions of x0 ({len(x0)}) and "
                             f"given bounds ({len(self.calibration_parameters.bounds)})")

        print("Calibrating model parameters...")
        res = minimize(
            fun=loss_fun,
            x0=x0,
            method=self.calibration_parameters.optimiser,
            bounds=self.calibration_parameters.bounds,
            **kwargs
        )

        self.update_model_parameters(res.x)
        print(res)
        print('Done.')