import fnmatch
from pathlib import Path
from typing import Sequence

# XXXX TODO finish rewrite
# Should this be renamed something like data_paths.py and be used to also generate output paths?


def find_last_valid_bg_paths(data_dir: str, lane_names: Sequence[str]):
    name_pat = "????????-??????"
    dir_path = Path(data_dir)
    ds_dir_matches = (dir_path.joinpath(x)
                      for x in fnmatch.filter(
                        [y.name for y in dir_path.iterdir()], name_pat))
    ds_dir_matches = sorted(x for x in ds_dir_matches if x.is_dir())
    while ds_dir_matches:
        ds_dir_path = ds_dir_matches.pop()
        run_dir_matches = (ds_dir_path.joinpath(x)
                           for x in fnmatch.filter(
                                [y.name for y in ds_dir_path.iterdir()],
                                name_pat))
        run_dir_matches = sorted(x for x in run_dir_matches if x.is_dir())
        while run_dir_matches:
            run_dir_path = run_dir_matches.pop()
            file_paths = {
                lane_name: run_dir_path.joinpath(
                    f"background-{run_dir_path.name}-{lane_name}.tif")
                for lane_name in lane_names}
            found_paths = {k: v for (k,v) in file_paths.items() if v.is_file()}
            if len(found_paths) < len(lane_names):
                continue
            return found_paths
    return None
