import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxz = np.max(z)
        ex = np.exp(z-maxz)
        new = ex/np.sum(ex)
        new = np.round(new,4)
        return new
