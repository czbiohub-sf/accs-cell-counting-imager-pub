from abc import ABC, abstractmethod
import logging
import time
from collections.abc import Sequence

import numpy as np


logger = logging.getLogger(__name__)


class CciHardware(ABC):
    settling_time: float | None = None

    @abstractmethod
    def get_lane_names(self) -> Sequence[str]:
        ...

    @abstractmethod
    def move_to_lane(self, name: str):
        ...

    @abstractmethod
    def capture_image(self) -> np.ndarray:
        ...

    def wait_settling_time(self):
        if self.settling_time:
            logger.debug(f"Waiting {self.settling_time:.1f} s...")
            time.sleep(self.settling_time)

    def capture_image_at_lane(self, lane_name: str) -> np.ndarray:
        self.move_to_lane(lane_name)
        self.wait_settling_time()
        return self.capture_image()
