from math import copysign, floor
from typing import Literal, Union
import numpy as np


RoundingMode = Literal['floor', 'ceil', 'half_up', 'half_down']

def flexible_round(x: Union[float, np.ndarray], decimals: int = 0, mode: RoundingMode = 'floor') -> Union[float, np.ndarray]:
    """
    Round a number or NumPy array to a given number of decimal places with flexible rounding modes.
    This implementation uses arithmetic operations for precise rounding of individual elements.
    
    Args:
    x (float or np.ndarray): The number or array to round.
    decimals (int): The number of decimal places to round to. Default is 0.
    mode (RoundingMode): The rounding mode. Options are:
                'floor' (default): Round towards negative infinity.
                'ceil': Round towards positive infinity.
                'half_up': Round to nearest, ties away from zero.
                'half_down': Round to nearest, ties towards zero.
    
    Returns:
    float or np.ndarray: The rounded number or array.
    """
    def round_scalar(val: float) -> float:
        multiplier = 10 ** decimals
        x_scaled = val * multiplier
        
        if mode == 'floor':
            return floor(x_scaled+0.5) / multiplier
        elif mode == 'ceil':
            return (floor(x_scaled) + (1 if x_scaled > 0 else 0)) / multiplier
        elif mode in ['half_up', 'half_down']:
            x_int = floor(x_scaled)
            frac = abs(x_scaled - x_int)
            sign = copysign(1, x_scaled)
            
            if frac > 0.5 or (frac == 0.5 and mode == 'half_up'):
                return (x_int + sign) / multiplier
            else:
                return x_int / multiplier
        else:
            raise ValueError("Invalid mode. Choose 'floor', 'ceil', 'half_up', or 'half_down'.")

    if isinstance(x, np.ndarray):
        return np.vectorize(round_scalar)(x)
    else:
        return round_scalar(x)