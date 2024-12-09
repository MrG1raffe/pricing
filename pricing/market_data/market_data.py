from dataclasses import dataclass
from typing import Union
from numpy.typing import NDArray
from numpy import float64

from ..products.product import Product

@dataclass
class MarketData:
    product: Product
    price: Union[float, NDArray[float64]]