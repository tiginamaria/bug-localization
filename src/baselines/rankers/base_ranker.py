from abc import ABC

import numpy as np


class BaseRanker(ABC):

    def rank(self, file_names: np.array[str], vect_file_contents: np.array[float]) -> dict:
        pass
