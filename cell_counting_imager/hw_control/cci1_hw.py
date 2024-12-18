from collections.abc import Sequence

import numpy as np


class Cci1Hardware:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def get_lane_names(self) -> Sequence[str]:
        return list("ABCDEFGH")

    def move_to_lane(self, name: str):
        raise NotImplementedError()

    def capture_image(self) -> np.ndarray:
        raise NotImplementedError()
