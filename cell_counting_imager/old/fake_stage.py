from typing import Iterable, Union

MOTION_RANGE = [10, 2100]


class FakeStage():
    def __init__(self, *args, **kwargs):
        self.clearIndexedPositions()
        self._allowed_motion_range = [0,0]
        self._is_motion_range_known = False
        self._current_pos = 0
        self.enable = False
        self._index_positions = {}

    def clearIndexedPositions(self):
        self._index_positions = {}

    def disable(self):
        self.enable = False
        return True

    def discoverMotionRange(self, *args, **kwargs) -> bool:
        if not self.enable:
            return False
        self._allowed_motion_range = list(MOTION_RANGE)
        self._is_motion_range_known = True
        return True

    def getAndParseMotorStatus(self):
        return {'OperationStatus': [],
                'ErrorStatus': [],
                'PositionStatus': []}

    def getCurrentPositionSteps(self):
        return self._current_pos

    def getIndexedPositions(self):
        return self._index_positions

    def getMotionRange(self):
        return self._allowed_motion_range

    def isLimitActive(self, limit) -> bool:
        return False

    def isTargetValid(self, target_pos):
        return True

    def moveAbsSteps(self, pos, **kwargs):
        if self._is_motion_range_known or open_loop_assert:
            self._current_pos = pos
            return True
        else:
            return False

    def moveRelSteps(self, offs, *args, **kwargs):
        if self._is_motion_range_known or open_loop_assert:
            self._current_pos += offs
            return True
        else:
            return False

    def moveToIndexedPosition(self, index, wait=True, **kwargs):
        if index in list(self._index_positions.keys()):
            return self.moveAbsSteps(self._index_positions[index])
        return False

    def moveToLimit(self, limit: str, **kwargs) -> bool:
        if not self.enable:
            return False
        if limit == "fwd":
            self.moveAbsSteps(MOTION_RANGE[1])
        elif limit == "rev":
            self.moveAbsSteps(MOTION_RANGE[0])
        else:
            return False

    def setCurrentPositionAsIndex(self, index) -> bool:
        pos = self.getCurrentPositionSteps()
        return self.setIndexedPositions({index: pos})

    def setIndexedPositions(self, position_map) -> bool:
        if type(position_map) != dict:
            return False
        if self._is_motion_range_known:
            vals = list(position_map.values())
            for v in vals:
                if not self.isTargetValid(v):
                    return False
            try:
                self._index_positions.update(position_map)
            except Exception as e:
                return False
        else:
            try:
                self._index_positions.update(position_map)
            except Exception as e:
                return False
        return True

    def setRotationSpeed(self, steps_per_second) -> bool:
        return True

    def print(self):
        pass

    def type(self):
        return "TicStage"

    def wait_for_motion(self, motion_tol_steps, timeout):
        pass

    @property
    def indexedPositions(self) -> dict:
        return self._index_positions


class TicStage(FakeStage):
    pass


class CciStage:
    def __init__(self, port: str):
        self._is_homed = False

    def home(self):
        self._is_homed = True

    def is_homed(self):
        return self._is_homed

    def move_to_lane(self, pos: str):
        self.move_to_pos(self.lane_positions[pos])

    def move_to_pos(self, pos: int, wiggle: bool = True):
        pass

    def set_lane_positions(self, poss: Union[Dict[str, int], Iterable[tuple[str, int]]]):
        self.lane_positions = {k: int(v) for (k,v) in dict(poss).items()}



__all__ = ("FakeStage", "TicStage", "CciStage")
