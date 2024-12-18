__docformat__ = "numpy"


from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, Type

from .preprocessing import CciImagePreprocessing, Cci1ImagePreprocessing, \
                           Cci2ImagePreprocessing
from .counting import CciCellCounting, Cci1CellCounting, Cci2CellCounting
from .common import update_info


CciCellCounterResult = namedtuple(
    "CciCellCounterResult",
    ["cells_per_ml", "cell_locations", "fg_cleaned", "feature_mask",
        "counting_info", "pp_info"],
)


class CciCellCounter(ABC):
    preprocessing_cls: Type[CciImagePreprocessing]
    counting_cls: Type[CciCellCounting]

    def __init__(self, preprocessing: CciImagePreprocessing = None,
                 counting: CciCellCounting = None):
        self.preprocessing = preprocessing
        self.counting = counting
        self._bg_pp_info: Dict[str, Any] = {}

    def _assert_inited(self):
        if self.preprocessing is None:
            raise RuntimeError(
                "Must init_preprocessing() before processing images")
        if self.counting is None:
            raise RuntimeError(
                "Must init_counting() before processing images")

    def init_preprocessing(self, **kwargs):
        self.preprocessing = self.preprocessing_cls(**kwargs)
        if self.counting is not None:
            self.preprocessing.set_counting_obj(self.counting)

    def init_counting(self, *args, **kwargs):
        self.counting = self.counting_cls(*args, **kwargs)
        if self.preprocessing is not None:
            self.preprocessing.set_counting_obj(self.counting)

    def set_bg_image(self, *args, **kwargs):
        self._assert_inited()
        self._bg_pp_info = self.preprocessing.set_bg_image(*args, **kwargs)
        return self._bg_pp_info

    def set_bg_image_from_path(self, *args, **kwargs):
        self._assert_inited()
        self._bg_pp_info = self.preprocessing.set_bg_image_from_path(
            *args, **kwargs)
        return self._bg_pp_info

    def _common_process_fg_image(self, fg_cleaned, feature_mask, pp_info,
                                 return_info=True):
        cell_locations, valid_volume, counting_info = \
            self.counting.process_fg_image(fg_cleaned, feature_mask)
        cells_per_ml = cell_locations.shape[0] / valid_volume
        update_info(pp_info, self._bg_pp_info)
        return CciCellCounterResult(
            cells_per_ml, cell_locations, fg_cleaned, feature_mask,
            counting_info if return_info else None,
            pp_info if return_info else None)

    def process_fg_image(self, fg_image, return_info=True):
        self._assert_inited()
        fg_cleaned, feature_mask, pp_info = \
            self.preprocessing.process_fg_image(fg_image)
        return self._common_process_fg_image(
            fg_cleaned, feature_mask, pp_info, return_info=return_info)

    def process_fg_image_from_path(self, fg_path, return_info=True):
        self._assert_inited()
        fg_cleaned, feature_mask, pp_info = \
            self.preprocessing.process_fg_image_from_path(fg_path)
        return self._common_process_fg_image(
            fg_cleaned, feature_mask, pp_info, return_info=return_info)


class Cci1CellCounter(CciCellCounter):
    preprocessing_cls = Cci1ImagePreprocessing
    counting_cls = Cci1CellCounting


class Cci2CellCounter(CciCellCounter):
    preprocessing_cls = Cci2ImagePreprocessing
    counting_cls = Cci2CellCounting
