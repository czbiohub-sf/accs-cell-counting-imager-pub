import argparse
import logging
from pathlib import Path
import sys
import statistics
import timeit
import traceback

import serial.tools.list_ports
import skimage.io

from .. import global_config
from .. import paths as cci_paths
from ..configurator import CciConfigurator
from ..hw_control import (
    CciHardware, CciHwError, CciStageCommsError, CciStageHomingTimeout,
    CciStageLimitSwitchHit, CciCameraNotFound)


USBSERIAL_VIDPID = (0x0403, 0x6001)  # FT232R default VID/PID


logger = logging.getLogger(__name__)


def list_usbserial_ports():
    return [
        port_info.device
        for port_info in serial.tools.list_ports.comports()
        if (port_info.vid, port_info.pid) == USBSERIAL_VIDPID
        ]


def get_arg_parser(argv):
    parser = argparse.ArgumentParser(prog=Path(argv[0]).name)
    parser.add_argument(
        "--config",
        metavar="NAME",
        help=f"specify the base hardware configuration to use "
             f"(default: {global_config.DEFAULT_BASE_CONFIG})",
        choices=cci_paths.get_system_config_names(),
        default=global_config.DEFAULT_BASE_CONFIG
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print debug messages",
        default=False
        )
    stage_group = parser.add_argument_group(
        title="Stage test",
        description=(
            "Select one of these options to run a test routine to verify the "
            "CCI stage is functioning properly."
            ),
        ).add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stage",
        metavar="PORTNAME",
        help="Connect to the serial port identified by PORTNAME "
             "(e.g. COM3, /dev/ttyUSB0, etc.)",
        default=None
        )
    stage_group.add_argument(
        "--stage-auto",
        action="store_true",
        help="Attempt to automatically discover which port the CCI is "
             "connected to baed on the USB VID/PID of the serial adapter "
             "cable. In case of ambiguity the program will print the names "
             "of all candidates.",
        default=False
        )
    cam_group = parser.add_argument_group(
        title="Camera",
        description=(
            "Select this option to verify that the software can communicate "
            "with the CCI camera."
            ),
        )
    cam_group.add_argument(
        "--camera",
        action="store_true",
        help="Attempt to connect to the camera and grab an image.",
        default=False
        )

    return parser


def _test_stage(cmdline_args: argparse.Namespace,
                cci_hw_cls: type[CciHardware]):
    stage_cls = cci_hw_cls.stage_cls
    stage_port = None
    if cmdline_args.stage_auto:
        search_desc = (
            "(looked for USB serial adapters with "
            f"VID=0x{USBSERIAL_VIDPID[0]:04X}, "
            f"PID=0x{USBSERIAL_VIDPID[1]:04X})"
            )
        port_names = list_usbserial_ports()
        if not port_names:
            print(f"Coudn't find any matching devices {search_desc}.")
            return 1
        if len(port_names) > 1:
            print(
                f"Found more than one candidate {search_desc}.\n"
                f"Specify one of these port names with the --stage option:"
                )
            for name in port_names:
                print(name)
            return 1
        stage_port = port_names[0]

    stage_port = cmdline_args.stage or stage_port
    print(f"Connecting to stage on port {stage_port!r}.")
    try:
        stage = stage_cls(stage_port)
    except CciStageCommsError as e:
        print(f"Couldn't connect to the stage!\n    {e}")
        return 1

    limit_l, limit_r = stage.get_travel_range()
    near_home = limit_l + (limit_r - limit_l) // 10

    print("\nAttempting to home. The stage should slowly move toward the "
          "side with the limit switch...")
    try:
        stage.home()
    except CciStageHomingTimeout:
        print("FAILED -- homing timed out.")
        print("\nMake sure the motor is moving in the correct direction "
              "and the limit switch is wired and adjusted properly.")
        return 1
    print("Homing complete.")

    def print_fail_advice():
        print("\nPlease check for any signs of binding or obstructions "
              "affecting the movement of the stage.")
    print("\nNow the stage will perform some test movements to check for "
          "full unobstructed travel.")
    ref_times = []
    test_times = []
    try:
        for _ in range(3):
            stage.move_to_pos(near_home, rapid=True)
            ref_times.append(
                timeit.timeit("stage.home()", number=1, globals=locals()))
            stage.move_to_pos(limit_r, rapid=True)
            stage.move_to_pos(near_home, rapid=True)
            test_times.append(
                timeit.timeit("stage.home()", number=1, globals=locals()))
    except CciStageLimitSwitchHit:
        print("FAILED -- the limit switch was activated.")
        print_fail_advice()
        return 1

    if abs(statistics.mean(test_times) - statistics.mean(ref_times)) > 0.2:
        print("FAILED -- timing measurements suggest loss of position.")
        print_fail_advice()
        return 1

    print("Success!")
    return 0


def _test_camera(cci_hw_cls: type[CciHardware]):
    print("Connecting to camera...")
    camera = cci_hw_cls.camera_cls()
    print("Capturing an image...")
    img = camera.capture_image()
    fname = "out.jpg"
    print(f"Saving to {fname!r}...")
    skimage.io.imsave(fname, skimage.img_as_ubyte(img), check_contrast=False)
    print("Success!")
    return 0


def _print_death_message(exc: BaseException, msg: str):
    print("\n", file=sys.stderr)
    traceback.print_exception(exc)
    print(
        f"\n\n********** :(\n{msg}\n"
        "See error details above.",
        file=sys.stderr
        )


def main(argv: list[str] = sys.argv) -> int:
    parser = get_arg_parser(argv)
    cmdline_args = parser.parse_args(argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if cmdline_args.debug else logging.WARNING
        )

    configurator = CciConfigurator()
    configurator.load_system_config(cmdline_args.config)
    cci_hw_cls = configurator.get_hw_class()

    try:
        if any((cmdline_args.stage, cmdline_args.stage,
                cmdline_args.stage_auto)):
            return _test_stage(
                cmdline_args=cmdline_args, cci_hw_cls=cci_hw_cls)
        if cmdline_args.camera:
            return _test_camera(cci_hw_cls=cci_hw_cls)
        parser.error("What would you like to do?")
    except CciCameraNotFound as e:
        _print_death_message(
            e.__context__, "The camera doesn't seem to be connected.")
    except CciHwError as e:
        _print_death_message(
            e, "There was a hardware-related problem during the test.")
    except RuntimeError as e:
        _print_death_message(
            e, "An unexpected error occurred :(")
        return 1

    return 0
