import numpy as np
from typing import List
import math

class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)
        n = len(x)
        rms = math.sqrt(np.mean((np.power(x,2) + eps)))
        x_hat = (gamma * x)/rms
        return np.round(x_hat,4).tolist()
        