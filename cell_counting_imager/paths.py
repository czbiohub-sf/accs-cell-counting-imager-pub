from contextlib import contextmanager
import fnmatch
import importlib.resources
import logging
from pathlib import Path
from typing import Union

from . import global_config


logger = logging.getLogger(__name__)
_system_configs_dir = importlib.resources.files(
    __package__).joinpath("configs")


def mkdirp(dir_path: Union[Path, str]):
    dir_path = Path(dir_path).resolve()
    if not dir_path.is_dir():
        logger.info(f"Creating output directory: {str(dir_path)!r}")
        dir_path.mkdir(parents=True, exist_ok=True)


def get_accs_home_dir():
    path = Path.home().joinpath("accs_data")
    #mkdirp(path)
    return path


def _get_home_subdir(subdir_name: str):
    path = get_accs_home_dir().joinpath(subdir_name)
    mkdirp(path)
    return path


def get_cci_logs_dir():
    return _get_home_subdir("cci_logs")


def get_cci_data_dir():
    return _get_home_subdir("cci_data")


def get_cci_images_dir():
    return _get_home_subdir("cci_images")


def get_cci_local_config_dir():
    return _get_home_subdir("cci_config")


def _get_config_path(config_name: str):
    return _system_configs_dir.joinpath(f"{config_name}.json")


def get_local_config_path(config_name: str = global_config.DEFAULT_LOCAL_CONFIG):
    return get_cci_local_config_dir().joinpath(f"{config_name}.json")


def local_config_exists():
    return get_local_config_path().is_file()


@contextmanager
def open_system_config_file(config_name: str):
    with importlib.resources.as_file(_get_config_path(config_name)) as path:
        with path.open("r", encoding="utf-8") as f:
            yield f


def read_local_config_template(config_name: str):
    return _get_config_path(f"{config_name}_local.template").read_text()


def _get_config_names():
    return [
        p.name.rsplit(".", 1)[0] for p in _system_configs_dir.iterdir()
        if fnmatch.fnmatch(p.name, "*.json") and p.is_file()
        ]


def get_system_config_names():
    return [
        x for x in _get_config_names()
        if x != "common" and not x.endswith("template")
        ]


def get_local_config_template_names():
    return [
        x.rsplit("_", 1)[0] for x in _get_config_names()
        if x.endswith("_local.template")
        ]
