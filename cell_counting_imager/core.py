import asyncio
import csv
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Union

from aiohttp import web
import matplotlib  # type: ignore[import]
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import fourier_shift  # type: ignore[import]
import skimage.io  # type: ignore[import]
from skimage.util import img_as_uint  # type: ignore[import]

from .configurator import CciConfigurator


_IMG_PAUSE_TIME = 0.5
_DRAW_PAUSE_TIME = 0.001
BG_DIRTY_THRESH = 0.20
STAGE_SETTLING_DELAY = 3.0


logger = logging.getLogger(__name__)


class _BgRejected(Exception):
    pass


class CellCounterCoreError(Exception):
    pass


class CellCounterCore():

    def __init__(self, configurator: 'CciConfigurator',
                 data_dir: Union[str, Path],
                 images_dir: Union[str, Path]
                 ):
        self._camera_is_loaded = False
        self._stage_is_loaded = False
        self.server_running = False
        self.csv_filename = None
        self.img_subdir = images_dir

        self.configurator = configurator
        self.server_params = configurator.get_server_params()
        self.hw = configurator.get_hw()
        self.data_dir = data_dir
        self.images_dir = images_dir
        self._init_cell_counters()
        self.backgroundImages = [None] * len(self.hw.get_lane_names())

    async def close_web_app(self):
        if self.server_running:
            # self._web_app.close()
            await self._web_app['/cci'].wait_closed()
            self.server_running = False

    def _init_cell_counters(self):
        self._cell_counter_channels = [
            (name, self.configurator.get_counter())
            for name in self.hw.get_lane_names()]

    def get_lane_names(self):
        return self.hw.get_lane_names()

    def set_bg_image(self, lane_no, bg_image, check_dirty=False):
        name, counter = self._cell_counter_channels[lane_no]
        info = counter.set_bg_image(bg_image)
        if check_dirty:
            bright = info['bg_feature_mask']
            bright_frac = 1. - bright.astype('bool').sum() / bright.size
            logger.info(f"Lane {name} background obstruction: "
                        f"{bright_frac*100.:.1f}%")
            if bright_frac > BG_DIRTY_THRESH:
                raise _BgRejected(
                    f"Lane {name} BG image: excessive bright features --"
                    " possible bubble, contamination, bad alignment?")
            # TODO: This is a quick fix, consider reimplementing more nicely

    def load_bgs_from_paths(self, paths):
        # XXXX TODO finish rewrite
        n_channels = len(self._cell_counter_channels)
        n_paths = len(paths)
        if n_paths != n_channels:
            raise ValueError(f"Wrong number of paths supplied (got {n_paths}, "
                             f"expected {n_channels})")
        for i in range(n_channels):
            lane_name = self._cell_counter_channels[i][0]
            path = paths[lane_name]
            logger.info(f"Setting BG channel {lane_name} from '{path}'...")
            img = skimage.io.imread(path) / 65535.
            self.backgroundImages[i] = img
            self.set_bg_image(i, img, check_dirty=False)

    def count_cells(self, lane_no, fg_image):
        # TODO: Change to lane name instead of index
        return self._cell_counter_channels[lane_no][1].process_fg_image(fg_image)

    async def handle_http_request(self, request):
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Error occurred processing the http request: {e}")
            return web.json_response(status=400,  # Bad Request
                                     data={'error': 'bad-request'})
        if 'Action' not in body:
            msg = "Request missing 'Action' key"
            logger.error(msg)
            return web.json_response(status=400, data={'error': msg})

        action = body['Action']

        if action == 'Start Background':
            logger.info('Received command to run BG scan')
            try:
                self.run_scan(False, True, True, False)
            except CellCounterCoreError as e:
                msg = f"BG scan failed: {e}"
                logger.exception(msg)
                return web.json_response(
                    status=500,
                    data={"error": msg})
            return web.json_response(
                status=200,
                data={
                    'Status': 'Acquired background',
                    'img_dir': self.img_subdir,
                    # TODO change behavior so we return the per-scan image dir
                    })

        if action == 'Start OT2 Scan':
            logger.info('Received command to run scan from OT2')
            try:
                densities, valid_areas, warnings = \
                    self.run_scan(True, True, False, True)
            except CellCounterCoreError as e:
                msg = f"Counting scan failed: {e}"
                logger.exception(msg)
                return web.json_response(
                    status=500,
                    data={"error": msg})
            return web.json_response(
                status=200,
                data={
                    # We're stuck with this structure for the moment for
                    # forward/backward compatibility reasons. Will likely
                    # rework eventually.
                    'Status': {
                        'Scan Results': densities,
                        'warnings': warnings,
                        'valid_areas': valid_areas,
                        'csv_path': self.csv_filename,
                        'img_dir': self.img_subdir,
                        # TODO change behavior so we return the
                        # per-scan image dir
                    }
                }
            )

        if action == 'Open CSV':
            self.start_new_record()
            resp_data = {
                'Status': 'Opened CSV',
                'csv_path': self.csv_filename,
                'img_dir': self.img_subdir
                }
            return web.json_response(status=200, data=resp_data)

        elif action == 'Close Server':
            await self.close_web_app()
            return web.json_response(status=410, data={
                'Status': 'Server Closed'})

        else:
            return web.json_response(status=400, data={
                'error': "Action not recognized"})

    def start_new_record(self, create=True):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestr}-cell_density.csv"
        self.csv_filename = os.path.join(self.data_dir, filename)
        self.img_subdir = os.path.join(self.images_dir, timestr)
        logger.info('Set CSV output path: ' + self.csv_filename)
        if create:
            with open(self.csv_filename, 'a') as f:
                pass
        return self.csv_filename

    def create_web_server(self, hostIP=None, portID=None):
        if hostIP is None:
            hostIP = self.server_params.bind_addr
        if portID is None:
            portID = self.server_params.bind_port
        try:
            # Create and run the actual server application
            self._web_app = web.Application()

            # Install the update function to serve the /update endpoint for
            # POST
            self._web_app.router.add_post('/cci', self.handle_http_request)

            # Initialize the CSV path in case client doesn't do it before
            # starting counting
            if self.csv_filename is None:
                self.start_new_record(create=False)

            # Configure the close/cleanup function
            # self._web_app.on_cleanup.append(self.close_web_app)

            # Run the application
            self.server_running = True
            web.run_app(self._web_app, host=hostIP, port=portID)

        except Exception as e:
            raise CellCounterCoreError(
                f"Could not create web application: {e}") from e

    def imshow(self, img, **kwargs):
        plt.imshow(img, **kwargs)
        plt.axis('off')
        plt.gca().set_aspect('equal')
        plt.tight_layout()

    def run_scan(self, runTracking=False, saveImages=True, isBackground=False,
                ot2Scan=False):
        """
        Runs a full scan: ie. moves to each position sequentially, snaps a
        camera image, counts cells in the image, then returns a dictionary
        containing the number of cells in each lane.

        """
        densities = None
        valid_areas = None
        warnings = None

        # Get the number of scan positions
        # XXXX TODO: Get rid of this numeric index stuff, rewrite to just use lane names
        n_images = len(self.hw.get_lane_names())

        logger.info("Starting scan...")

        try:
            images = [None] * n_images 

            plt.ion()

            scan_ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.img_subdir, scan_ts)
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Images will be saved to: {run_dir!r}")

            densities = []
            valid_areas = []
            warnings = []
            for i in range(n_images):
                warnings.append([])
                lane_name = self.hw.get_lane_names()[i]

                if (runTracking and not isBackground
                        and self.backgroundImages[i] is None):
                    raise CellCounterCoreError(
                        "Can't count cells without background image "
                        f"for lane {lane_name}")

                logger.info(f"Capturing image for lane {lane_name}...")
                images[i] = self.hw.capture_image_at_lane(lane_name)

                if np.mean(images[i]) < 0.05:
                    raise Exception("Got very dark image from camera")
                # TODO: Implement this more nicely -- should it go in the
                # image processing library?

                # If it's a background image, store it
                if isBackground:
                    self.backgroundImages[i] = images[i]
                    try:
                        self.set_bg_image(i, images[i], check_dirty=True)
                    except _BgRejected:
                        self.save_image(run_dir, scan_ts, 'bg-rejected-',
                                        lane_name, images[i])
                        raise

                # Show the raw image either way
                plt.cla()
                plt.title(f"Raw image, Lane {lane_name}", fontsize=8)
                self.imshow(images[i], cmap='gray', vmin=0., vmax=1.)
                plt.draw()
                plt.pause(_IMG_PAUSE_TIME)

                if saveImages and not isBackground:
                    self.save_image(run_dir, scan_ts, 'image-input-',
                                    lane_name, images[i])

                if runTracking and not isBackground:
                    logger.info("Counting cells...")

                    result = self.count_cells(i, images[i])
                    if "warnings" in result.pp_info:
                        warnings[-1] += result.pp_info["warnings"]
                    if "warnings" in result.counting_info:
                        warnings[-1] += result.counting_info["warnings"]
                    for warning in warnings[-1]:
                        logger.warning(
                            f"Warning (lane {lane_name}): {warning}")

                    if saveImages:
                        self.save_image(run_dir, scan_ts, 'image-input-',
                                        lane_name, images[i])
                        self.save_image(run_dir, scan_ts, 'image-pp-',
                                        lane_name, result.fg_cleaned*0.5 + 0.5)

                    # show diagnostic plot
                    plt.cla()
                    self.imshow(
                        result.fg_cleaned, vmin=0., vmax=0.05, cmap="gray")
                    if len(result.cell_locations):
                        plt.gca().scatter(
                            result.cell_locations[:, 1],
                            result.cell_locations[:, 0],
                            edgecolor='red',
                            s=30.,
                            facecolor="none")
                    plt.title(f'Processed image, lane {lane_name} '
                              f'({len(result.cell_locations)} total '
                              f'-> {result.cells_per_ml:.1f}/mL)',
                              fontsize=8)
                    plt.draw()
                    plt.pause(_IMG_PAUSE_TIME)

                    valid_area = (result.feature_mask.astype("bool").sum()
                                  / result.feature_mask.size)
                    valid_area = min(valid_area / 0.649104, 1.) # XXXX
                    logger.info(
                        f"Lane {lane_name}: "
                        f"cell count = {len(result.cell_locations)}, "
                        f"valid area = {valid_area*100.:.0f}%, "
                        f"shift = {result.pp_info['align_shift']}")
                    densities.append(result.cells_per_ml * 1e-3)
                    # ^ We respond in cells/μL
                    valid_areas.append(valid_area)

                if saveImages:
                    self.save_image(run_dir, scan_ts, 'background-', lane_name,
                                    self.backgroundImages[i])
                    if not isBackground:
                        self.save_image(
                            run_dir, scan_ts, 'plot-', lane_name, is_plt=True)
                        plt.cla()
                        self.imshow(  # XXXX TODO rewrite all this mess, no more pyplot
                            result.counting_info['count_mask'],
                            vmin=0., vmax=1.0)
                        if len(result.cell_locations):
                            plt.gca().scatter(
                                result.cell_locations[:, 1],
                                result.cell_locations[:, 0],
                                edgecolor='red',
                                s=50.,
                                facecolor="none")
                        plt.title(f'Count mask, lane {i} '
                                  f'({len(result.cell_locations)} total '
                                  f'-> {result.cells_per_ml:.1f}/mL)',
                                  fontsize=8)
                        plt.draw()
                        plt.pause(_IMG_PAUSE_TIME)
                        self.save_image(run_dir, scan_ts, 'mask-',
                                        lane_name, is_plt=True)

                try:
                    plt.close()
                except Exception:
                    logger.exception('cannot close fig')

            # TODO: Write out cell positions as well
            if ot2Scan:
                if self.csv_filename is None:
                    logger.warning(
                        "CSV filename not set; not saving count values")
                else:
                    logger.info('Saving counts to ' + self.csv_filename)
                    with open(self.csv_filename, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(densities)

        except Exception as e:
            raise CellCounterCoreError(
                f"Error during scan attempt: {e}") from e

        logger.info("Scan complete.")

        return densities, valid_areas, warnings

    def save_image(self, run_dir, run_ts, header, channel, img=None, is_plt=False):
        filename = header + run_ts + '-' + str(channel)
        fullFilename = os.path.join(run_dir, filename)
        if is_plt:
            plt.savefig(fullFilename)
        else:
            path = fullFilename + '.tif'
            logger.debug(f"Saving image to {path!r}")
            skimage.io.imsave(
                path,
                (img*65535.).astype('uint16'),
                check_contrast=False
                )

    def __del__(self):
        # TODO
        pass
