import argparse
import logging
from pathlib import Path
import sys

from .. import global_config
from .. import paths as cci_paths


logger = logging.getLogger(__name__)


def get_arg_parser(argv):
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name,
        description="initialize the local configuration file for the CCI"
        )
    parser.add_argument(
        "--config",
        help=f"which base instrument config to use "
             f"(default: {global_config.DEFAULT_BASE_CONFIG})",
        choices=cci_paths.get_local_config_template_names(),
        default=global_config.DEFAULT_BASE_CONFIG
        )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite local config file if it exists",
        default=False
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print debug messages",
        default=False
        )
    return parser


def _print_local_config_loc():
    path = cci_paths.get_local_config_path().resolve()
    print(f"Your local configuration file is at:\n{str(path)}")


def main(argv: list[str] = sys.argv) -> int:
    parser = get_arg_parser(argv)
    cmdline_args = parser.parse_args(argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if cmdline_args.debug else logging.WARNING
        )

    if cci_paths.local_config_exists() and not cmdline_args.force:
        print(
            "Local config file already exists. "
            "Use --force if you want to overwrite it.",
            file=sys.stderr
            )
        _print_local_config_loc()
        return 1
    template_text = cci_paths.read_local_config_template(cmdline_args.config)
    dest_path = cci_paths.get_local_config_path().resolve()
    with dest_path.open("w", encoding="utf-8") as f:
        f.write(template_text)
    logger.info(
        f"Copied local config template for base config {cmdline_args.config!r} "
        f"to {str(dest_path)!r}"
        )
    _print_local_config_loc()
    return 0
