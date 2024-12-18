import dataclasses
import inspect
import json
import logging
from pathlib import Path
from collections.abc import Sequence
from typing import Any, TextIO

from .hw_control.cci_hw import CciHardware
from .hw_control import cci1_hw
from .hw_control import cci2_hw
from .image_processing.cell_counter import CciCellCounter
from . import paths as cci_paths


logger = logging.getLogger(__name__)


def get_hw_classes():
    # TODO: Implement Cci1Hardware? Or don't bother at this point?
    return [
        cci1_hw.Cci1Hardware,
        #cci1_hw.Cci1SimHardware,
        cci2_hw.Cci25Hardware,
        cci2_hw.Cci2SimHardware,
        ]


# pylint: disable=import-outside-toplevel
def get_counter_classes():
    from .image_processing import Cci1CellCounter, Cci2CellCounter
    return [Cci1CellCounter, Cci2CellCounter]


@dataclasses.dataclass
class CciServerParams:
    api_mode: str
    bind_addr: str
    bind_port: int


class CciConfigurationError(RuntimeError):
    pass


class CciConfiguration:
    PARENT_CONFIG_KEY = 'system_config'
    HW_CLS_NAME_KEY = 'hardware_class'
    COUNTER_CLS_NAME_KEY = 'cell_counter_class'
    SERVER_PARAMS_KEY = 'server_params'
    HW_PARAMS_KEY = 'hardware_params'
    PP_PARAMS_KEY = 'preprocessing_params'
    COUNTING_PARAMS_KEY = 'counting_params'

    hw_cls_name: str | None
    counter_cls_name: str | None
    server_params: CciServerParams | None
    hw_params: dict[str, Any]
    pp_params: dict[str, Any]
    counting_params: dict[str, Any]
    parent_config_name: str | None

    # pylint: disable=unused-argument
    def __init__(self, *,
                 hw_cls_name: str | None = None,
                 counter_cls_name: str | None = None,
                 server_params: CciServerParams | None = None,
                 hw_params: dict[str, Any] | None = None,
                 pp_params: dict[str, Any] | None = None,
                 counting_params: dict[str, Any] | None = None,
                 parent_config_name: str | None = None):
        for k, v in locals().items():
            if k == 'self':
                continue
            if k.endswith('params') and v is None:
                v = {}
            setattr(self, k, v)

    @staticmethod
    def _get_init_args(cls_):
        allowed = set()
        required = set()
        sig = inspect.signature(cls_)
        for name, param in sig.parameters.items():
            allowed.add(name)
            if param.default == param.empty:
                required.add(name)
        return allowed, required

    @classmethod
    def load_system_config(cls, config_name: str):
        with cci_paths.open_system_config_file(config_name) as f:
            return cls.load_from_file(f)

    @classmethod
    def load_from_file(cls, file_or_path: str | Path | TextIO,
                       load_parent: bool = True):
        if isinstance(file_or_path, (str, Path)):
            with Path(file_or_path).open("r", encoding="utf-8") as f:
                return cls.load_from_file(f, load_parent=load_parent)
        file = file_or_path
        path_str = getattr(file, 'name', None)
        logger.debug(f"Loading config from: {path_str!r}")
        path_desc = f" {path_str!r}" if path_str is not None else ""
        try:
            info = json.load(file)
        except json.decoder.JSONDecodeError as e:
            raise CciConfigurationError(
                f"Error while parsing config file{path_desc} -- "
                f"{e.msg} at line {e.lineno}") from e
        if not isinstance(info, dict):
            raise CciConfigurationError(
                f"Config file{path_desc} is invalid -- "
                f"top level must be a mapping ")
        parent_config_name = info.get(cls.PARENT_CONFIG_KEY, None)
        server_params = info.get(cls.SERVER_PARAMS_KEY, {})
        hw_cls_name = info.get(cls.HW_CLS_NAME_KEY, None)
        hw_params = info.get(cls.HW_PARAMS_KEY, {})
        counter_cls_name = info.get(cls.COUNTER_CLS_NAME_KEY, None)
        pp_params = info.get(cls.PP_PARAMS_KEY, {})
        counting_params = info.get(cls.COUNTING_PARAMS_KEY, {})

        config = cls(
            hw_cls_name=hw_cls_name, counter_cls_name=counter_cls_name,
            server_params=server_params, hw_params=hw_params,
            pp_params=pp_params, counting_params=counting_params,
            parent_config_name=parent_config_name)
        if load_parent and (parent_config_name is not None):
            return config.reload_over_parent()
        return config

    def reload_over_parent(self):
        parent_config = self.load_system_config(self.parent_config_name)
        parent_config.update_from_config(self)
        return parent_config

    def write_to_file(self, file_or_path: str | Path | TextIO):
        if isinstance(file_or_path, (str, Path)):
            with Path(file_or_path).open("w", encoding="utf-8") as f:
                self.write_to_file(f)
                return
        json.dump(self.as_dict(), file_or_path, indent=4)

    def as_dict(self):
        return {
            self.HW_CLS_NAME_KEY: self.hw_cls_name,
            self.COUNTER_CLS_NAME_KEY: self.counter_cls_name,
            self.SERVER_PARAMS_KEY: self.server_params.copy(),
            self.HW_PARAMS_KEY: self.hw_params.copy(),
            self.PP_PARAMS_KEY: self.pp_params.copy(),
            self.COUNTING_PARAMS_KEY: self.counting_params.copy()
            }

    def as_json(self):
        return json.dumps(self.as_dict())

    def update_from_config(self, config: 'CciConfiguration'):
        for name in ('hw_cls_name', 'counter_cls_name'):
            new_v = getattr(config, name)
            if new_v:
                setattr(self, name, new_v)
        for name in (
                'server_params', 'hw_params', 'pp_params', 'counting_params'):
            getattr(self, name).update(getattr(config, name))

    def validate(self, hw_cls: CciHardware, counter_cls: CciCellCounter,
                 require_server_params: bool = False):
        # TODO: Actually type-check values...?
        # (OR just have a convention that init methods should check or attempt
        # to convert everything?)
        to_check = [
            (self.HW_PARAMS_KEY, self.hw_params, hw_cls),
            (self.PP_PARAMS_KEY, self.pp_params,
                counter_cls.preprocessing_cls),
            (self.COUNTING_PARAMS_KEY, self.counting_params,
                counter_cls.counting_cls)
            ]
        if require_server_params:
            to_check.append((
                self.SERVER_PARAMS_KEY, self.server_params, CciServerParams))

        for parent_key, params, cls_ in to_check:
            allowed, required = self._get_init_args(cls_)
            for param in params:
                if param not in allowed:
                    raise CciConfigurationError(
                        f"unrecognized parameter name {param!r} "
                        f"in section {parent_key!r}")
            for param in required:
                if param not in params:
                    raise CciConfigurationError(
                        f"missing required parameter {param!r} "
                        f"in section {parent_key!r}")
            if hasattr(cls_.__init__, "check_arg_values"):
                try:
                    cls_.__init__.check_arg_values(**params)
                except ValueError as e:
                    raise CciConfigurationError(
                        "Illegal parameter value(s) in section "
                        f"{parent_key!r}: {e}"
                        ) from e


class CciConfigurator:
    def __init__(self, populate_classes: bool = True):
        self.hw_classes: dict[str, type] = {}
        self.counter_classes: dict[str, type] = {}
        self.config = CciConfiguration()
        self._validated = False
        self.hw_cls = None
        self.counter_cls = None

        if populate_classes:
            self.add_hw_classes(get_hw_classes())
            self.add_counter_classes(get_counter_classes())

    def _add_class(self, dest, x: type, override_name: str | None = None):
        self._validated = False
        cls_name = override_name
        if cls_name is None:
            cls_name = x.__name__
        dest[cls_name] = x

    def add_hw_class(self, x: type, override_name: str | None = None):
        self._add_class(self.hw_classes, x, override_name)

    def add_hw_classes(self, x: Sequence[type]):
        for c in x:
            self.add_hw_class(c)

    def add_counter_class(self, x: type, override_name: str | None = None):
        self._add_class(self.counter_classes, x, override_name)

    def add_counter_classes(self, x: Sequence[type]):
        for c in x:
            self.add_counter_class(c)

    def load_config(self, config: CciConfiguration):
        # TODO: Consistent naming scheme for these methods --
        # "load" is used confusingly
        self._validated = False
        self.config.update_from_config(config)

    def load_config_from_file(self, path_or_file: str | Path | TextIO):
        config = CciConfiguration.load_from_file(path_or_file)
        self.load_config(config)

    def load_system_config(self, config_name: str):
        config = CciConfiguration.load_system_config(config_name)
        self.load_config(config)

    def save_config(self, path_or_file: str | Path | TextIO):
        self.config.write_to_file(path_or_file)

    def get_hw_class(self):
        if self.config.hw_cls_name is None:
            raise CciConfigurationError(
                "missing required config item "
                + repr(self.config.HW_CLS_NAME_KEY))
        if self.config.hw_cls_name not in self.hw_classes:
            raise CciConfigurationError(
                "Can't find specified CCI hardware class: "
                + repr(self.config.hw_cls_name)
                )
        return self.hw_classes[self.config.hw_cls_name]

    def get_counter_class(self):
        if self.config.counter_cls_name is None:
            raise CciConfigurationError(
                "missing required config item "
                + repr(self.config.COUNTER_CLS_NAME_KEY))
        if self.config.counter_cls_name not in self.counter_classes:
            raise CciConfigurationError(
                "Can't find specified CCI cell counter class: "
                + repr(self.config.counter_cls_name)
                )
        return self.counter_classes[self.config.counter_cls_name]

    def validate(self):
        if self._validated:
            return
        hw_cls = self.get_hw_class()
        counter_cls = self.get_counter_class()
        self.config.validate(hw_cls, counter_cls)
        self._hw_cls = hw_cls
        self._counter_cls = counter_cls
        self._validated = True

    def get_server_params(self):
        self.validate()
        return CciServerParams(**self.config.server_params)

    def get_hw(self):
        self.validate()
        return self._hw_cls(**self.config.hw_params)

    def get_counter(self):
        self.validate()
        counter = self._counter_cls()
        counter.init_preprocessing(**self.config.pp_params)
        counter.init_counting(**self.config.counting_params)
        return counter
