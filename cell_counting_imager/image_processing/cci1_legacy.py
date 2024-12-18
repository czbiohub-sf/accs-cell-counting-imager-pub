# pylint: disable=invalid-name

import numpy as np
import scipy.ndimage
import skimage.filters
import skimage.measure
import skimage.registration
import skimage.transform

from .preprocessing import CciImagePreprocessing
from .counting import CciCellCounting
from .cell_counter import CciCellCounter


class _ConfigData:
    _stateMachine = {
        "_MIN_CELL_AREA": 10, 
        "_MAX_CELL_AREA": 500,
        "_HIGHPASS_PIXELS": 20,
        "_LOWPASS_PIXELS": 2,
        "_IMAGE_DOWNSAMPLE": 2,
        "_IMAGE_XCROP": 500,
        "_IMAGE_YCROP": 300,
        "_IMG_PERCENTILE_HIGH": 99.9,
        "_IMG_PERCENTILE_LOW": 0.1,
        "_MIN_CELL_SPACING": 3,
        "_IMAGING_VOLUME": 1.043
    }


class Cci1LegacyImagePreprocessing(CciImagePreprocessing, _ConfigData):
    def process_fg_image(self, fg_image: np.ndarray
                         ) -> tuple[np.ndarray, float, dict]:
        if self.bg_image is None:
            raise RuntimeError(
                "Must call set_bg_image() before process_fg_image()")
        ds = self._stateMachine['_IMAGE_DOWNSAMPLE']
        bg = self.bg_image_ds

        img = skimage.transform.downscale_local_mean(fg_image, (ds, ds))
        img = img.astype(float)

        # imAlign
        shift, error, diffphase = \
            skimage.registration.phase_cross_correlation(bg, img)
        alignedImage = scipy.ndimage.fourier_shift(
            np.fft.fftn(img), shift)
        alignedImage = np.fft.ifftn(alignedImage)

        img = alignedImage.real
        img = img - bg
        img[img < 0] = 0

        xcrop = np.round(self._stateMachine['_IMAGE_XCROP']/ds).astype(int)
        ycrop = np.round(self._stateMachine['_IMAGE_YCROP']/ds).astype(int)
        img = img[ycrop:-ycrop, xcrop:-xcrop]

        lp = self._stateMachine['_LOWPASS_PIXELS']/ds
        hp = self._stateMachine['_HIGHPASS_PIXELS']/ds

        #img = self.filterImage(img, lp, hp)
        img = skimage.filters.gaussian(
            img - skimage.filters.gaussian(img, sigma=hp), sigma=lp)

        return img, np.ones(img.shape), {}

    def set_bg_image(self, bg_image: np.ndarray) -> dict:
        ds = self._stateMachine['_IMAGE_DOWNSAMPLE']

        bg = skimage.transform.downscale_local_mean(bg_image, (ds, ds))
        bg = bg.astype(float)

        self.bg_image_ds = bg
        self.bg_image = bg_image

        info = {
            "bg_scaled": bg
        }
        return info


class Cci1LegacyCellCounting(CciCellCounting, _ConfigData):
    BINNING_FACTOR = 2

    def process_fg_image(self, fg_image: np.ndarray, feature_mask: np.ndarray
                         ) -> tuple[np.ndarray, float, dict]:
        img = fg_image

        # img = self.convertTo_uint16(img)
        minVal = img.min().astype('float')
        maxVal = img.max().astype('float')
        img = np.uint16(65535*(img.astype('float') - minVal)/(maxVal - minVal))

        threshold = skimage.filters.threshold_otsu(img)

        mask = img > threshold

        max_region_area = self._stateMachine['_MAX_CELL_AREA']
        min_region_area = self._stateMachine['_MIN_CELL_AREA']

        mask_label = skimage.measure.label(mask)
        props = skimage.measure.regionprops(mask_label)

        for prop in props:
            if prop.area > max_region_area or prop.area < min_region_area:
                mask[mask_label == prop.label] = 0

        img_masked = np.multiply(img, mask)

        min_cell_spacing = self._stateMachine['_MIN_CELL_SPACING']
        positions = skimage.feature.peak_local_max(
            img_masked, indices=True, min_distance=min_cell_spacing,
            threshold_abs=threshold)

        return positions, self._stateMachine['_IMAGING_VOLUME']*1e-3, {}


class Cci1LegacyCellCounter(CciCellCounter):
    preprocessing_cls = Cci1LegacyImagePreprocessing
    counting_cls = Cci1LegacyCellCounting
