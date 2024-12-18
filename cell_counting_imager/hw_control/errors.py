class CciHwError(RuntimeError):
    pass


class CciStageError(CciHwError):
    pass


class CciStageRemoteError(CciHwError):
    pass


class CciMechanicalFault(CciStageError):
    pass


class CciStageHomingTimeout(CciMechanicalFault):
    pass


class CciStageLimitSwitchHit(CciMechanicalFault):
    pass


class CciStageCommsError(CciStageError):
    pass


class CciCameraError(CciHwError):
    pass


class CciCameraNotFound(CciCameraError):
    pass


class CciCameraConfigError(CciCameraError):
    pass
