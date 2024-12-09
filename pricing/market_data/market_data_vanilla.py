from dataclasses import dataclass
from typing import Union
from numpy.typing import NDArray
from numpy import float64

from .market_data import MarketData
from ..products.vanilla_option import VanillaOption

@dataclass
class MarketDataVanilla(MarketData):
    implied_volatility: Union[float, NDArray[float64]]

    def __post_init__(self):
        if not isinstance(self.product, VanillaOption):
            raise ValueError("The product should be of type `VanillaOption`.")

        # TODO: add automatic completion of prices / IVs.