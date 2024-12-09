from dataclasses import dataclass
from ..models.model_params import  PricingParams
from typing import List, Tuple, Callable


@dataclass
class CalibrationParams:
    params_to_calibrate: Tuple[str]
    optimiser: str = "Powell"
    pricing_method: str = "lewis"
    pricing_params: PricingParams = PricingParams()
    vector_to_params: Callable = None
    bounds: List[Tuple] = None
