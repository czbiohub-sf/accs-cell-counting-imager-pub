from abc import ABC, abstractmethod
import logging
from typing import Any

import numpy as np
import scipy.signal  # type: ignore[import]
import skimage.feature  # type: ignore[import]
import skimage.morphology  # type: ignore[import]


logger = logging.getLogger(__name__)


class CciCellCounting(ABC):
    @abstractmethod
    def process_fg_image(self, fg_image: np.ndarray,
                         feature_mask: np.ndarray) \
            -> tuple[np.ndarray, float, dict[str, Any]]:
        pass


class Cci1CellCounting(CciCellCounting):
    roi_volume_ml: float
    margin: float
    corner_zone_radius: float
    kernel_disk_radius: int
    kernel_padding: int
    kernel_downsample_factor: int
    blobdog_threshold: float
    blobdog_overlap: float
    blobdog_sigma: float
    blobdog_sigma_ratio: float

    # pylint: disable=unused-argument
    def __init__(self,
                 roi_volume_ml: float,
                 margin: float,
                 corner_zone_radius: float,
                 kernel_disk_radius: int,
                 kernel_padding: int,
                 kernel_downsample_factor: int,
                 blobdog_threshold: float,
                 blobdog_overlap: float,
                 blobdog_sigma: float,
                 blobdog_sigma_ratio: float):
        for key, value in locals().items():
            if key not in ("self",):
                setattr(self, key, value)

    def process_fg_image(
            self, fg_image: np.ndarray, feature_mask: np.ndarray):
        count_mask = feature_mask.copy()

        margin_px = round(count_mask.shape[0] * self.margin)
        margin_px = min(margin_px, min(feature_mask.shape))
        count_mask[:margin_px + 1, ...] = 0
        count_mask[count_mask.shape[0] - margin_px - 1:, ...] = 0
        count_mask[..., :margin_px + 1] = 0
        count_mask[..., count_mask.shape[1] - margin_px - 1:] = 0
        # TODO: make a method for this in CciImageProcessingCommon
        # (similar op in Cci1ImagePreprocessing.make_align_mask, ...)

        if self.corner_zone_radius:
            corner_zone_r_pix = round(
                count_mask.shape[0] * self.corner_zone_radius)
            corner_mask = np.zeros(count_mask.shape, dtype=bool)
            center = (round(count_mask.shape[0] / 2),
                      round(count_mask.shape[1] / 2))
            corner_mask[skimage.draw.disk(center, corner_zone_r_pix,
                                          shape=count_mask.shape)] = 1
            count_mask[corner_mask == 0] = 0
        # TODO: make a method for this in CciImageProcessingCommon

        kernel = skimage.morphology.disk(self.kernel_disk_radius)
        kernel = np.pad(kernel, self.kernel_padding)
        kernel = skimage.transform.downscale_local_mean(
            kernel, (self.kernel_downsample_factor,)*2)

        cc_img = scipy.signal.correlate2d(
            fg_image, kernel, mode="same", boundary="symm")
        cc_img[cc_img < 0.] = 0.

        blobs = skimage.feature.blob_dog(
            cc_img,
            min_sigma=self.blobdog_sigma,
            max_sigma=self.blobdog_sigma,
            sigma_ratio=self.blobdog_sigma_ratio,
            threshold=self.blobdog_threshold,
            overlap=self.blobdog_overlap)
        cell_locations = blobs[..., :2].astype(int)
        cell_locations = np.array(
            [(y, x) for (y, x) in cell_locations if count_mask[y][x]])

        valid_volume = (self.roi_volume_ml * count_mask.astype("bool").sum()
                        / count_mask.size)

        info = {
            "count_mask": count_mask,
        }

        return cell_locations, valid_volume, info


class Cci2CellCounting(CciCellCounting):
    volume_per_px: float
    kernel_disk_radius: int
    kernel_padding_ratio: float
    kernel_downsample_factor: int
    blob_area_max: int
    blob_flood_tol: float
    peak_thresh: float
    peak_min_dist: int

    # pylint: disable=unused-argument
    def __init__(self, *,
                 volume_per_px: float,
                 kernel_disk_radius: int,
                 kernel_padding_ratio: float,
                 kernel_downsample_factor: int,
                 blob_area_max: int,
                 blob_flood_tol: float,
                 peak_thresh: float,
                 peak_min_dist: int):
        for key, value in iter(locals().items()):
            if key not in ("self",):
                setattr(self, key, value)

    def process_fg_image(
            self, fg_image: np.ndarray, feature_mask: np.ndarray):
        count_mask = feature_mask.copy()

        kernel = skimage.morphology.disk(self.kernel_disk_radius)
        kernel = np.pad(
            kernel, round(self.kernel_padding_ratio*self.kernel_disk_radius)
            ).astype('double')
        # XXXX FIXME: generate with zeros() and draw.disk() instead?
        kernel = skimage.transform.downscale_local_mean(
            kernel, (self.kernel_downsample_factor,)*2, cval=kernel.min())
        kernel -= kernel.mean()
        cc_img = scipy.signal.correlate2d(
            fg_image, kernel, mode="same", boundary="symm")
        all_peaks = skimage.feature.corner_peaks(
            cc_img, threshold_abs=self.peak_thresh,
            min_distance=self.peak_min_dist)
        cell_locations = []
        #cells_mask = np.zeros(cc_img.shape)
        n_blobs_removed = 0
        for (i, j) in all_peaks:
            zone = skimage.segmentation.flood(
                cc_img, (i, j), tolerance=self.blob_flood_tol*cc_img[i, j])
            if zone.sum() > self.blob_area_max:
                n_blobs_removed += 1
                continue
            #cells_mask[zone] = 1
            cell_locations.append((i, j))
        logger.debug(f"Removed {n_blobs_removed} blobs")

        cell_locations_valid = np.array(
            [(y, x) for (y, x) in cell_locations if count_mask[y][x]])

        valid_volume = self.volume_per_px * count_mask.astype("bool").sum()

        info = {
            "count_mask": count_mask,
        }

        return cell_locations_valid, valid_volume, info
