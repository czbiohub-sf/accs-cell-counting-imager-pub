from collections.abc import Iterable, Sequence
from enum import IntFlag
import inspect
import logging
from pathlib import Path
from typing import Union

import numpy as np
from pypylon import pylon
import serial
import skimage.io

from .cci_hw import CciHardware
from . import _tic_driver
from .errors import CciCameraNotFound, CciCameraConfigError


logger = logging.getLogger(__name__)


class Cci25Stage:
    HOMING_TIMEOUT = 30.
    POS_LIMIT_L = 0
    POS_LIMIT_R = 9250
    wiggle_n_cycles = 3
    wiggle_dist = 24
    lane_positions: dict[str, int]
    driver_wrapper_cls = _tic_driver.CciTicSerial

    def __init__(self, port: str, homing_timeout: float | None = None):
        self.homing_timeout = homing_timeout or self.HOMING_TIMEOUT
        self.tic = self.driver_wrapper_cls(port)
        self.lane_positions = {}

    def get_travel_range(self) -> tuple[int, int]:
        return (self.POS_LIMIT_L, self.POS_LIMIT_R)

    def set_lane_positions(self, poss: Union[dict[str, int],
                           Iterable[tuple[str, int]]]):
        self.lane_positions = {k: int(v) for (k, v) in dict(poss).items()}
        logger.debug(f"New lane positions: {self.lane_positions!r}")

    def home(self, timeout: float | None = None):
        timeout = timeout or self.homing_timeout
        logger.debug("Homing...")
        self.tic.go_home(_tic_driver.TicHomingDir.REV, timeout=timeout)
        self.tic.move_abs(64)
        logger.debug("Homing complete.")

    def is_homed(self):
        return not self.tic.get_position_uncertain()

    def move_to_lane(self, name: str):
        self.move_to_pos(self.lane_positions[name])

    def move_to_pos(self, pos: int, rapid: bool = False):
        logger.debug(f"Moving to position {pos}...")
        self.tic.move_abs(pos)
        if not rapid:
            self._wiggle()
        logger.debug("Move complete.")

    def get_current_pos(self):
        return self.tic.get_current_pos()

    def _wiggle(self):
        start_pos = self.tic.get_current_pos()
        for dist in (self.wiggle_dist, round(self.wiggle_dist/2.)):
            logger.debug(f"Wiggle cycle: {self.wiggle_n_cycles} cycles, "
                         f"+/- {dist} steps from center")
            for _ in range(self.wiggle_n_cycles):
                self.tic.move_abs(start_pos + dist)
                self.tic.move_abs(start_pos - dist)
        self.tic.move_abs(start_pos)


class Cci25SimStage(Cci25Stage):
    driver_wrapper_cls = _tic_driver.CciTicSerialSim


class Cci2CameraCaptureError(IOError):
    pass


# TODO: exception translation wrapper
class Cci2Camera:
    GRAB_TIMEOUT = 8000
    N_RETRIES = 10
    RETRY_DELAY = 0.25

    def __init__(self):
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice())
        except pylon.RuntimeException as e:
            raise CciCameraNotFound() from e
        self.camera.Open()
        try:
            self.initialize_settings()
        except pylon.InvalidArgumentException as e:
            raise CciCameraConfigError(
                "Invalid camera config value:", repr(e)
                ) from e

    def set_exposure_time(self, t: float):
        exposure_us = int(t * 1e6)
        self.camera.ExposureAuto.SetValue("Off")
        self.camera.ExposureTimeAbs.SetValue(exposure_us)
        actual = self.camera.ExposureTimeAbs.GetValue()
        logger.debug(f"Exposure set to {exposure_us} us "
                     f"(read back as {actual})")
        return actual / 1e6

    def set_exposure_auto(self):
        self.camera.ExposureAuto.SetValue("On")

    def initialize_settings(self):
        self.camera.PixelFormat.SetValue("Mono12")
        self.camera.GainAuto.SetValue("Off")
        self.camera.GainRaw.SetValue(0)
        self.camera.GammaEnable.SetValue(False)
        self.camera.BinningHorizontalMode.SetValue("Sum")
        self.camera.BinningVerticalMode.SetValue("Sum")
        self.set_binning_factor(1)

    def set_binning_factor(self, v: int):
        self.camera.BinningHorizontal.SetValue(v)
        self.camera.BinningVertical.SetValue(v)
        self.camera.OffsetX.SetValue(0)
        self.camera.OffsetY.SetValue(0)
        self.camera.Width.SetValue(self.camera.Width.GetMax())
        self.camera.Height.SetValue(self.camera.Height.GetMax())
        actual_h = self.camera.BinningHorizontal.GetValue()
        actual_v = self.camera.BinningVertical.GetValue()
        logger.debug(f"Binning set to {v}x{v} "
                     f"(read back as {actual_h}x{actual_v})")

    def get_frame_dims(self):
        return (self.camera.Height.GetValue(), self.camera.Width.GetValue())

    def capture_image(self):
        logger.debug("Grabbing image...")
        fail_count = 0
        while fail_count <= self.N_RETRIES:
            grab_result = self.camera.GrabOne(
                self.GRAB_TIMEOUT, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                break
            err_code = grab_result.GetErrorCode()
            err_desc = grab_result.GetErrorDescription()
            logger.debug("GrabOne() resulted in error code "
                         f"0x{err_code:X} ({err_desc})")
            fail_count += 1
        else:
            tries_msg = f" after {fail_count} tries" if fail_count else ""
            raise Cci2CameraCaptureError(
                f"Grab failed{tries_msg} -- Error 0x{err_code:X} ({err_desc})")
        if fail_count:
            logger.warning(
                f"Grab retried {fail_count} time(s) due to errors")
        img_raw = grab_result.GetArray() / 4095.
        grab_result.Release()
        logger.debug("Grabbed image successfully.")
        return img_raw


class Cci2SimCamera:
    IMG_W = 2748
    IMG_H = 1836
    # TODO: Consider checking an image and using values from that instead

    def __init__(self):
        cci_hw_instance = (
            inspect.currentframe().f_back.f_locals.get('self', None)
            )  # lol
        img_dir_path = getattr(cci_hw_instance, "_fake_image_dir_path", None)
        if img_dir_path is None:
            img_dir_path = Path.cwd().joinpath("sim_images")
            logger.warning(f"sim_image_dir not specified, "
                           f"using default: {str(img_dir_path)!r}")
        self._bg_paths = list(img_dir_path.glob("background-*.tif"))
        if not self._bg_paths:
            logger.warning("No BG images found, will send gray frames")
        self._fg_paths = list(img_dir_path.glob("image-input-*.tif"))
        if not self._fg_paths:
            logger.warning("No FG images found, will send gray frames")
        self._img_idx = 0
        logger.info(f"{self.__class__.__name__} found {len(self._bg_paths)} "
                    f"BG images and {len(self._fg_paths)} FG images in "
                    f"{str(img_dir_path)!r}")

    def set_exposure_time(self, t: float):
        return t

    def set_exposure_auto(self):
        pass

    def initialize_settings(self):
        pass

    def set_binning_factor(self, v: int):
        pass

    def get_frame_dims(self):
        return (self.IMG_H, self.IMG_W)

    def _blank_image(self):
        return np.full((self.IMG_H, self.IMG_W), 0.1, dtype=np.float64)

    def capture_image(self):
        if self._img_idx >= len(self._bg_paths) + len(self._fg_paths):
            self._img_idx = 0
        if self._img_idx < len(self._bg_paths):
            img_path = (
                self._bg_paths[self._img_idx] if self._bg_paths else None
                )
        else:
            img_path = (
                self._fg_paths[self._img_idx - len(self._bg_paths)]
                if self._fg_paths
                else None
                )
        if img_path is None:
            logger.info("Returning a blank image")
            return self._blank_image()
        logger.info(f"{self.__class__.__name__} reading image "
                    f"from {str(img_path)!r} ")
        img = skimage.io.imread(img_path).astype(np.float64) / 65535.
        self._img_idx += 1
        return img


class Cci25Hardware(CciHardware):
    stage_cls = Cci25Stage
    camera_cls = Cci2Camera

    def __init__(self, serial_port: str,
                 lane_positions: Sequence[tuple[str, int]],
                 exposure_ms: float, binning_factor: int,
                 settling_time_s: float,
                 homing_timeout_s: float | None = None):
        self.settling_time_s = settling_time_s
        self._init_stage(serial_port, lane_positions, homing_timeout_s)
        self._init_camera(exposure_ms, binning_factor)

    def _init_stage(self, serial_port: str,
                    lane_positions: Sequence[tuple[str, int]],
                    homing_timeout_s: float | None = None):
        logger.info("Initializing stage...")
        self.stage = self.stage_cls(
            serial_port, homing_timeout=homing_timeout_s)
        self.stage.set_lane_positions(lane_positions)
        self.stage.home()
        logger.info("Stage initialization successful.")

    def _init_camera(self, exposure_ms: float, binning_factor: int):
        logger.info("Initializing camera...")
        self.camera = self.camera_cls()
        self.camera.set_exposure_time(exposure_ms * 1e-3)
        self.camera.set_binning_factor(binning_factor)
        logger.info("Camera initialization successful.")

    def get_lane_names(self) -> list[str]:
        return list(self.stage.lane_positions.keys())

    def move_to_lane(self, name: str):
        self.stage.move_to_lane(name)

    def capture_image(self) -> np.ndarray:
        return self.camera.capture_image()


class Cci2SimHardware(Cci25Hardware):
    stage_cls = Cci25SimStage
    camera_cls = Cci2SimCamera

    def __init__(self, serial_port: str,
                 lane_positions: Sequence[tuple[str, int]],
                 exposure_ms: float, binning_factor: int,
                 settling_time_s: float,
                 homing_timeout_s: float | None = None,
                 sim_image_dir: str | None = None):
        self._fake_image_dir_path = (
            Path(sim_image_dir) if sim_image_dir is not None else None)
        args = {
            k: v for (k, v) in locals().items()
            if k in inspect.signature(super().__init__).parameters
            }
        super().__init__(**args)

    def wait_settling_time(self):
        if self.settling_time_s:
            logger.debug(f"Skipping {self.settling_time_s:.1f} s wait.")
