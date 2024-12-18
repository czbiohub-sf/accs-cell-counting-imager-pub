import inspect
import itertools
import logging
import os
import random

import skimage.io  # type: ignore[import]


DATA_DIR_PATH = "fake_cam_data"
DEFAULT_LANE_NO = 0
FAILURE_RATE = 0.0


logger = logging.getLogger(__name__)


def get_subdir_names(base_path: str):
    return sorted([
        x for x in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, x))])


def get_dataset_names(data_dir_path: str = DATA_DIR_PATH):
    return get_subdir_names(data_dir_path)


def get_run_names(ds_name: str, data_dir_path: str = DATA_DIR_PATH):
    return get_subdir_names(os.path.join(data_dir_path, ds_name))


def get_run_dir_path(ds_name: str, run_name: str,
                     data_dir_path: str = DATA_DIR_PATH):
    return os.path.join(data_dir_path, ds_name, run_name)


def _get_image_paths_by_prefix(ds_name: str, run_name: str, prefix: str,
                               data_dir_path: str = DATA_DIR_PATH):
    run_dir_path = get_run_dir_path(ds_name, run_name,
                                    data_dir_path=data_dir_path)
    return [os.path.join(run_dir_path, f"{prefix}-{run_name}-{lane_no}.tif")
            for lane_no in range(8)]


def get_image_paths(ds_name: str, run_name: str,
                    data_dir_path: str = DATA_DIR_PATH):
    for fg_prefix in ("image", "image-input"):
        fg_paths = _get_image_paths_by_prefix(
            ds_name, run_name, fg_prefix, data_dir_path)
        if os.path.isfile(fg_paths[0]):
            break
    bg_paths = _get_image_paths_by_prefix(
        ds_name, run_name, "background", data_dir_path)
    return fg_paths, bg_paths


def get_datasets(data_dir_path: str = DATA_DIR_PATH,
                 assume_common_bg: bool = True):
    ds_names = get_dataset_names(data_dir_path)
    run_names = {x: get_run_names(x, data_dir_path=data_dir_path)
                 for x in ds_names}
    fg_paths = {x: {} for x in ds_names}
    bg_paths = {x: {} for x in ds_names}
    for ds_name in ds_names:
        for run_name in run_names[ds_name]:
            run_fg_paths, run_bg_paths = \
                get_image_paths(ds_name, run_name, data_dir_path=data_dir_path)
            if assume_common_bg:
                first_run_name = run_names[ds_name][0]
                if first_run_name in bg_paths[ds_name] \
                        and bg_paths[ds_name][first_run_name]:
                    run_bg_paths = bg_paths[ds_name][first_run_name]
            for paths in (run_fg_paths, run_bg_paths):
                for path in paths:
                    if not os.path.isfile(path):
                        raise RuntimeError(
                            f"Missing expected image file: {path}")
            fg_paths[ds_name][run_name], bg_paths[ds_name][run_name] = \
                run_fg_paths, run_bg_paths
    return ds_names, run_names, bg_paths, fg_paths


class NonsenseError(RuntimeError):
    pass


class FakeCciCamera:
    def __init__(self):
        self.random = random.Random()
        self._init_paths()

    def _randomly_fail(self):
        if not FAILURE_RATE:
            return
        caller_name = inspect.currentframe().f_back.f_code.co_name
        if self.random.random() < FAILURE_RATE:
            raise NonsenseError(f"{caller_name} randomly failed")

    def activateCamera(self):
        self._randomly_fail()
        return True

    def deactivateCamera(self):
        self._randomly_fail()
        return True

    def preview(self):
        return

    def print(self):
        pass

    def configTrig(self, triggerType=None):
        self._randomly_fail()
        return True

    def startAcquisition(self):
        self._randomly_fail()
        return True

    def stopAcquisition(self):
        self._randomly_fail()
        return True

    def snapImage(self):
        self._randomly_fail()
        caller_lcl = inspect.stack()[1].frame.f_locals
        # They say I did something bad
        if "i" in caller_lcl:
            lane_no = caller_lcl["i"]
            logger.debug(f"Selecting lane {lane_no}")
        else:
            lane_no = DEFAULT_LANE_NO
            logger.debug(f"Didn't detect lane number (default {lane_no})")
        if "isBackground" in caller_lcl and caller_lcl["isBackground"]:
            return self._get_next_bg(lane_no)
        return self._get_next_fg(lane_no)
        # Then why's it feel so good

    def bindTo(self, *args, **kwargs):
        pass

    @property
    def frameDims(self):
        return (self.Height, self.Width)

    @property
    def Width(self):
        return self._img_shape[1]

    @property
    def Height(self):
        return self._img_shape[0]

    def _init_paths(self):
        ds_names, run_names, bg_paths, fg_paths = get_datasets()
        self._ds_names = itertools.cycle(
            random.sample(ds_names, len(ds_names)))
        self._run_names = {
            k: itertools.cycle(random.sample(v, len(v)))
            for (k, v) in run_names.items()}
        self._all_bg_paths = bg_paths
        self._all_fg_paths = fg_paths
        self._bg_lanes_used = set()
        self._fg_lanes_used = set()

        self._img_shape=(1,1)
        try:
            sample_path = bg_paths[ds_names[0]][run_names[ds_names[0]][0]][0]
        except (IndexError, KeyError):
            sample_path = None
        if sample_path:
            self._img_shape = skimage.io.imread(sample_path).shape

        self._cycle_ds()

    def _cycle_ds(self):
        self._ds_name = next(self._ds_names)
        logger.debug(f"Cycling to new dataset: {self._ds_name}")
        self._bg_lanes_used = set()
        self._cycle_fg_set()

    def _cycle_fg_set(self):
        self._run_name = next(self._run_names[self._ds_name])
        logger.debug(f"Cycling to new run: {self._ds_name}:{self._run_name}")
        self._fg_lanes_used = set()

    def _get_next_bg(self, lane_no):
        if lane_no in self._bg_lanes_used:
            self._cycle_ds()
        logger.debug('"Capturing" BG image for'
                      f'{self._ds_name}:{self._run_name}:{lane_no}')
        img = skimage.io.imread(
            self._all_bg_paths[self._ds_name][self._run_name][lane_no])
        self._bg_lanes_used.add(lane_no)
        return img

    def _get_next_fg(self, lane_no):
        if lane_no in self._fg_lanes_used:
            self._cycle_fg_set()
        logger.debug('"Capturing" FG image for'
                      f'{self._ds_name}:{self._run_name}:{lane_no}')
        img = skimage.io.imread(
            self._all_fg_paths[self._ds_name][self._run_name][lane_no])
        self._fg_lanes_used.add(lane_no)
        return img


class PyFLIRCamera(FakeCciCamera):
    pass


class CciCameraCaptureError(IOError):
    pass


class CciCamera():
    def __init__(self):
        self._cam = FakeCciCamera()

    def set_exposure_time(self, t: float):
        pass

    def set_exposure_auto(self):
        pass

    def initialize_settings(self):
        pass

    def set_binning_factor(self, v: int):
        pass

    def get_frame_dims(self):
        return self._cam.frameDims

    def capture_image(self):
        caller_lcl = inspect.stack()[1].frame.f_locals
        if 'i' in caller_lcl:
            i = caller_lcl['i']
        if 'isBackground' in caller_lcl:
            isBackground = caller_lcl['isBackground']
        return self._cam.snapImage() / 65535.


__all__ = ("FakeCciCamera", "PyFLIRCamera", "CciCamera")
