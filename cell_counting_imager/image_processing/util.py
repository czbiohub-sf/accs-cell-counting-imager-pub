from .cell_counter import CciCellCounter


# XXXX TODO rewrite/deprecate
def get_cell_counters(n_channels: int, cell_counter_cls: CciCellCounter,
                      pp_params: dict | None = None,
                      counting_params: dict | None = None):
    pp_params = {} if pp_params is None else pp_params
    counting_params = {} if counting_params is None else counting_params
    counters = [cell_counter_cls() for i in range(n_channels)]
    for counter in counters:
        counter.init_preprocessing(**pp_params)
        counter.init_counting(**counting_params)
    return counters
