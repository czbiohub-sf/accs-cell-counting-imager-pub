from enum import IntEnum, IntFlag
import functools
import time

import serial
import ticlib

from .errors import (
    CciHwError, CciStageCommsError, CciStageHomingTimeout,
    CciStageLimitSwitchHit, CciStageRemoteError)


class TicHomingDir(IntEnum):
    REV = 0
    FWD = 1


class TicMiscFlags(IntFlag):
    ENERGIZED = 1
    POSITION_UNCERTAIN = 2
    FWD_LIMIT_ACTIVE = 4
    REV_LIMIT_ACTIVE = 8
    HOMING_ACTIVE = 16


class TicErrorStatus(IntFlag):
    INTENTIONALLY_DEENERGIZED = 1 << 0
    MOTOR_DRIVER_ERROR = 1 << 1
    LOW_VIN = 1 << 2
    KILL_SWITCH_ACTIVE = 1 << 3
    REQUIRED_INPUT_INVALID = 1 << 4
    SERIAL_ERROR = 1 << 5
    COMMAND_TIMEOUT = 1 << 6
    SAFE_START_VIOLATION = 1 << 7
    ERR_LINE_HIGH = 1 << 8
    SERIAL_FRAMING = 1 << 16
    SERIAL_RX_OVERRUN = 1 << 17
    SERIAL_FORMAT = 1 << 18
    SERIAL_CRC = 1 << 19
    ENCODER_SKIP = 1 << 20


class _TicErrorChecking:
    IGNORE_FLAGS = {
        'exit_safe_start': (
            TicErrorStatus.SAFE_START_VIOLATION
            | TicErrorStatus.COMMAND_TIMEOUT
            ),
        'reset_command_timeout': TicErrorStatus.COMMAND_TIMEOUT,
        }

    def _wrap(self, meth, meth_name: str, check_for_error: bool = True):
        @functools.wraps(meth)
        def outer(*args, **kwargs):
            try:
                result = meth(*args, **kwargs)
            except RuntimeError as e:
                raise CciStageCommsError(f"ticlib error: {e}") from e
            except serial.SerialException as e:
                raise CciStageCommsError(f"pyserial error: {e}") from e
            if check_for_error:
                self._check_error_occurred(
                    ignore=TicErrorStatus(
                        self.IGNORE_FLAGS.get(meth_name, 0)
                        )
                    )
            return result
        return outer

    def __init__(self, tic: ticlib.ticlib.TicBase):
        meth_names = [cmd for (cmd, _, _) in ticlib.ticlib.COMMANDS]
        meth_names += [
            f"get_{var}" for (var, _, _, _) in ticlib.ticlib.VARIABLES
            ]
        for meth_name in meth_names:
            check_for_error = True
            meth = getattr(tic, meth_name, None)
            if meth is None:
                continue
            if meth_name == "get_error_occured":  # (sic)
                if isinstance(meth, functools.partial):
                    # use read command 0xA2 so it clears the error on read
                    meth = functools.partial(
                        meth.func, 0xA2, *meth.args[1:], **meth.keywords)
                check_for_error = False
            setattr(
                self,
                meth_name,
                self._wrap(
                    meth,
                    meth_name=meth_name,
                    check_for_error=check_for_error
                    )
                )

    # this is just to shut pylint up
    def __getattr__(self, name):
        raise AttributeError()

    def _check_error_occurred(self, ignore: TicErrorStatus = 0):
        error_val = ticlib.ticlib.unsigned_int(self.get_error_occured())
                                                    # (sic)
        error_flag = TicErrorStatus(error_val) & ~TicErrorStatus(ignore)
        if error_flag:
            raise CciStageRemoteError(
                "Tic error flags set: "
                + ", ".join(x.name for x in TicErrorStatus if x in error_flag)
                )


class CciTic:
    POLL_INTERVAL = 0.02

    def __init__(self, tic):
        self._tic = _TicErrorChecking(tic)
        self.exit_safe_start()
        self.reset_command_timeout()
        self.clear_driver_error()

    def exit_safe_start(self):
        self._tic.exit_safe_start()

    def reset_command_timeout(self):
        self._tic.reset_command_timeout()

    def energize(self):
        self._tic.energize()

    def deenergize(self):
        self._tic.deenergize()

    def clear_driver_error(self):
        self._tic.clear_driver_error()

    def get_misc_flags(self):
        return TicMiscFlags(self._tic.get_misc_flags()[0])

    def get_homing_active(self):
        return self.match_misc_flags(TicMiscFlags.HOMING_ACTIVE)

    def get_position_uncertain(self):
        return self.match_misc_flags(TicMiscFlags.POSITION_UNCERTAIN)

    def get_limit_switches_active(self):
        return self.match_misc_flags(
            TicMiscFlags.FWD_LIMIT_ACTIVE | TicMiscFlags.REV_LIMIT_ACTIVE)

    def match_misc_flags(self, v: TicMiscFlags):
        return self.get_misc_flags() & v

    def get_current_pos(self):
        return self._tic.get_current_position()

    def go_home(self, dir_: TicHomingDir, wait: bool = True,
                timeout: float | None = None):
        self.exit_safe_start()
        self.reset_command_timeout()
        self._tic.go_home(int(dir_))
        if wait:
            start_time = time.monotonic()
            while self.get_homing_active():
                self.reset_command_timeout()
                time.sleep(self.POLL_INTERVAL)
                if (timeout is not None
                        and time.monotonic() - start_time > timeout):
                    raise CciStageHomingTimeout(
                        f"Homing timed out after {timeout:.1f} s.")

    def move_abs(self, pos: int, wait: bool = True):
        self.exit_safe_start()
        if self.get_position_uncertain():
            raise CciHwError("Not homed")
        self._tic.set_target_position(pos)
        if wait:
            while (self.get_current_pos()
                    != self._tic.get_target_position()):
                self._tic.set_target_position(pos)
                switches = self.get_limit_switches_active()
                if switches:
                    self._tic.set_target_position(self.get_current_pos())
                    raise CciStageLimitSwitchHit(
                        "Limit switch(es) activated during motion: "
                        + ", ".join(
                            ["fwd"] if switches
                                & TicMiscFlags.FWD_LIMIT_ACTIVE else []
                            + ["rev"] if switches
                                & TicMiscFlags.REV_LIMIT_ACTIVE else []
                            )
                        )
                time.sleep(self.POLL_INTERVAL)


class CciTicUsb(CciTic):
    driver_cls = ticlib.TicUSB

    def __init__(self, *args, **kwargs):
        tic = self.driver_cls(*args, **kwargs)
        super().__init__(tic)


class CciTicSerial(CciTic):
    serial_cls = serial.Serial
    driver_cls = ticlib.TicSerial

    def __init__(self, port: str, baud_rate: int = 9600, **kwargs):
        try:
            self._ser_port = self.serial_cls(
                port, baudrate=baud_rate, timeout=0.5, write_timeout=0.25)
        except serial.SerialException as e:
            raise CciStageCommsError(
                f"Failed to open serial port: {port!r}") from e
        try:
            tic = self.driver_cls(self._ser_port, **kwargs)
            # pylint: disable=no-member
            tic.get_misc_flags()  # ping
        except RuntimeError as e:
            raise CciStageCommsError("Motor controller not connected?") from e
        super().__init__(tic)

    def close(self):
        self._ser_port.close()


class CciTicSerialSim(CciTicSerial):
    class DummySerial:
        def __init__(self, *args, **kwargs):
            pass

    class DummyTic:
        # pylint: disable=unused-argument
        def __init__(self, *args, **kwargs):
            self._pos_uncertain = True
            self._energized = False
            self._cur_pos = 0
            self._tgt_pos = 0

        def exit_safe_start(self):
            pass

        def energize(self):
            self._energized = True

        # pylint: disable=unused-argument
        def go_home(self, dir_: int):
            self._pos_uncertain = False
            self._cur_pos = 0

        def set_target_position(self, pos: int):
            self._tgt_pos = pos
            self._cur_pos = pos

        def get_current_position(self) -> int:
            return self._cur_pos

        def get_target_position(self) -> int:
            return self._tgt_pos

        def get_error_occured(self):
            return bytes(b"\x00\x00\x00\x00")

        def reset_command_timeout(self):
            pass

        def clear_driver_error(self):
            pass

        def get_misc_flags(self) -> bytes:
            byte0 = (
                int(self._energized) * TicMiscFlags.ENERGIZED.value
                | int(self._pos_uncertain)
                * TicMiscFlags.POSITION_UNCERTAIN.value
                )
            return bytes([byte0])

    serial_cls = DummySerial
    driver_cls = DummyTic
