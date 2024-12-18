# TODO: Graceful, deliberate shutdown on CTRL-C (or other signal) including releasing handles on hardware. Doesn't matter when just running the script from the command line but would be nice to do

import argparse
import code
import inspect
import json
import logging
from pathlib import Path
import os
import sys
import time
from typing import Iterable, Union

from .. import paths as cci_paths
from ..core import CellCounterCore
from ..configurator import (
    CciConfigurator, CciConfiguration, CciConfigurationError)
from .. import recovery
from .. import _version as version_module


logger = logging.getLogger(__name__)


def _os_exit_code(ex_code_name: str):
    default = 0 if ex_code_name.upper() == "OK" else 1
    attr_name = f"EX_{ex_code_name}"
    return getattr(os, attr_name, default)


def get_arg_parser(argv):
    parser = argparse.ArgumentParser(prog=Path(argv[0]).name)
    parser.add_argument(
        "--config",
        help="load builtin config named CONFIG_NAME (stackable)",
        choices=cci_paths.get_system_config_names(),
        action="append",
        default=[]
        )
    parser.add_argument(
        "--local-config",
        help="load config from file at PATH instead of the standard location "
             "(stackable)",
        metavar="PATH",
        action="append",
        default=[]
        )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help="save outputs to directory at PATH",
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print debug messages",
        default=False
        )
    parser.add_argument(
        "--console",
        action="store_true",
        help="open an interactive Python console "
             "instead of running the server",
        default=False
        )
    parser.add_argument(
        "--load-bg",
        metavar="PATH_LIST",
        type=json.loads,
        help="pre-load a set of background images, "
             "where PATH_LIST is a JSON array of paths"
        )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="attempt to resume using the most recent CSV file and "
             "background images",
        default=False
        )
    return parser


def get_configurator(
        sys_config_names: Iterable[str],
        local_config_paths: Iterable[Union[Path, str]]
        ):
    configurator = CciConfigurator()
    for config_name in sys_config_names:
        configurator.load_system_config(config_name)
        logger.info(f"Loaded system config {config_name!r}")
    for local_config_path in (
            local_config_paths or [cci_paths.get_local_config_path()]):
        config = CciConfiguration.load_from_file(
            local_config_path, load_parent=False)
        parens_text = ""
        if (
            (config.parent_config_name is not None)
            and (config.parent_config_name not in sys_config_names)
                ):
            parens_text = (
                f" (over system config {config.parent_config_name!r})")
            config = config.reload_over_parent()
        configurator.load_config(config)
        logger.info(
            f"Loaded local config{parens_text}: {str(local_config_path)!r}")
    configurator.validate()
    return configurator


def main(argv: list[str] = sys.argv) -> int:
    output_ts_str = time.strftime("%Y%m%d-%H%M%S")
    parser = get_arg_parser(argv)
    cmdline_args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if cmdline_args.debug else logging.INFO)
    for stfu in ("PIL", "matplotlib"):
        # suppress unnecessary noise from 3rd party libs
        logging.getLogger(stfu).setLevel(logging.WARNING)

    if (
            not cci_paths.local_config_exists()
            and not cmdline_args.config
            and not cmdline_args.local_config
            ):
        parser.error(
            "No local config found. Run cci_config to create it, or specify "
            "--config and/or --local-config"
            )

    if cmdline_args.output_dir is not None:
        data_dir = Path(cmdline_args.output_dir)
        logs_dir = data_dir
        images_dir = data_dir
        cci_paths.mkdirp(data_dir)
    else:
        data_dir = cci_paths.get_cci_data_dir()
        images_dir = cci_paths.get_cci_images_dir()
        logs_dir = cci_paths.get_cci_logs_dir()

    log_file_path = logs_dir.joinpath(f"{output_ts_str}-cci_log.txt")
    log_file_handler = logging.FileHandler(log_file_path)
    # TODO: Handle IO errors for above
    log_file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(log_file_handler)
    logger.info(f"Writing log output to {str(log_file_path)!r}.")
    package_path = Path(version_module.__file__).parent.resolve()
    logger.info(f"Running package version {version_module.__version__} "
                f"from location: {str(package_path)!r}")
    config_out_path = logs_dir.joinpath(f"{output_ts_str}-cci_config.json")
    try:
        return _main_inner(cmdline_args, config_out_path=config_out_path,
                           images_dir=images_dir, data_dir=data_dir)
    except CciConfigurationError as e:
        logging.error("Configuration error: ", e)
        return 1 # TODO value
    except RuntimeError as e:
        last_frame = inspect.trace()[-1].frame.f_code.co_qualname
        logging.exception(f"{e.__class__.__name__} in {last_frame}")
        logging.critical("Aborted due to urecoverable error")
        return 1# TODO value


def _main_inner(cmdline_args: argparse.Namespace, config_out_path: Path,
                data_dir: Path, images_dir: Path) -> int:
    try:
        configurator = get_configurator(
            sys_config_names=cmdline_args.config,
            local_config_paths=cmdline_args.local_config
            )
    except CciConfigurationError as e:
        logger.error(f"Aborting due to configuration error: {e}")
        return _os_exit_code("DATAERR")
    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        logger.error(
            f"Failed to open config file {str(e.filename)!r} -- {e.strerror}")
        return _os_exit_code("IOERR")
    try:
        configurator.save_config(config_out_path)
    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        logger.error(f"Couldn't save config file {str(config_out_path)!r} "
                     f"-- {e.strerror}")
        return _os_exit_code("IOERR")
    logger.info(f"Saved configuration to {str(config_out_path)!r}")

    if cmdline_args.console:
        logger.info("Initializing hardware...")
        # pylint: disable=possibly-unused-variable
        hardware = configurator.get_hw()
        logger.info("Entering interactive console.")
        local = {
            k: v
            for (k, v) in locals().items()
            if k in ('configurator', 'hardware')}
        print("\n\n")
        banner = (f"local vars: {', '.join(local.keys())}\n"
                  "Type quit() to exit\n")
        code.interact(local=local, banner=banner)
        return 0

    logger.info("Initializing CellCounterCore...")
    core = CellCounterCore(
        configurator=configurator,
        data_dir=data_dir,
        images_dir=images_dir
        )

    bg_paths = None
    if cmdline_args.recover:
        bg_paths = recovery.find_last_valid_bg_paths(
            cmdline_args.output_dir, core.get_lane_names())
        if bg_paths is None:
            logger.error("Couldn't locate a set of background images to use "
                         "for recovery")
            return _os_exit_code("DATAERR")
    if cmdline_args.load_bg:
        bg_paths = cmdline_args.load_bg
    if bg_paths:
        logger.info("Pre-loading background images...")
        core.load_bgs_from_paths(bg_paths)

    logger.info("Starting server...")
    try:
        core.create_web_server()
    except Exception as e:
        logger.exception("Server aborted due to error (traceback follows)")
        return _os_exit_code("SOFTWARE")

    return 0
