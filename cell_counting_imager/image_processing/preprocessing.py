__docformat__ = "numpy"


from abc import ABC, abstractmethod
import logging
import math
from typing import Any

import scipy.ndimage  # type: ignore[import]
import skimage.draw  # type: ignore[import]
import skimage.filters  # type: ignore[import]
import skimage.io  # type: ignore[import]
import skimage.measure  # type: ignore[import]
import skimage.registration  # type: ignore[import]
import skimage.segmentation  # type: ignore[import]
import skimage.transform  # type: ignore[import]
import numpy as np

from .common import CciImageProcessingCommon, update_info
from .counting import CciCellCounting
from .._param_rules import require_b_not_none_if_a


logger = logging.getLogger(__name__)


class CciImagePreprocessing(ABC, CciImageProcessingCommon):
    @abstractmethod
    def process_fg_image(self, fg_image: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        pass

    @abstractmethod
    def set_bg_image(self, bg_image: np.ndarray) -> dict[str, Any]:
        pass

    def process_fg_image_from_path(self, path: str) \
            -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        fg_image = self.read_image(path)
        fg_cleaned, feature_mask, info = self.process_fg_image(fg_image)
        info.update({"fg_image": fg_image})
        return fg_cleaned, feature_mask, info

    def set_bg_image_from_path(self, path: str) -> dict[str, Any]:
        bg_image = self.read_image(path)
        info = self.set_bg_image(bg_image)
        info.update({"bg_image": bg_image})
        return info


class Cci1ImagePreprocessing(CciImagePreprocessing):
    input_rescale_factor: float
    input_gaussian_sigma: float
    align_bg_pixel_thresh: float
    align_bg_row_thresh: float
    align_wall_extent_max: float
    align_mask_margin: float
    align_cutout_height: float
    align_mpcc_overlap_ratio: float
    align_oversub_factor: float
    crop_roi_size: tuple[int, int]
    cleanup_wall_margin: float
    cleanup_bg_brt_thresh: float
    cleanup_bg_grow_factor: float
    cleanup_fg_brt_thresh: float
    cleanup_fg_grow_factor: float
    cleanup_fg_grow_cycles: int
    cleanup_fg_size_thresh: float
    cleanup_gaussian_sigma: float
    bg_image: np.ndarray | None

    # pylint: disable=unused-argument
    def __init__(
            self,
            input_rescale_factor: float,
            input_gaussian_sigma: float,
            align_bg_pixel_thresh: float,
            align_bg_row_thresh: float,
            align_wall_extent_max: float,
            align_mask_margin: float,
            align_cutout_height: float,
            align_mpcc_overlap_ratio: float,
            align_oversub_factor: float,
            crop_roi_size: tuple[int, int],
            cleanup_wall_margin: float,
            cleanup_bg_brt_thresh: float,
            cleanup_bg_grow_factor: float,
            cleanup_fg_brt_thresh: float,
            cleanup_fg_grow_factor: float,
            cleanup_fg_grow_cycles: int,
            cleanup_fg_size_thresh: float,
            cleanup_gaussian_sigma: float,
            bg_image: np.ndarray | None = None
            ):
        self._bg_cropped = None
        for key, value in locals().items():
            if key not in ("self", "bg_image"):
                setattr(self, key, value)
        if bg_image is not None:
            self.set_bg_image(bg_image)

    @staticmethod
    def make_align_mask(bg_image: np.ndarray, pixel_thresh: float,
                        row_thresh: float, mask_margin: float,
                        wall_margin: float,
                        wall_extent_max: float) \
            -> tuple[np.ndarray, tuple[int, int]]:
        # Make initial mask based on bright regions
        raw_mask = bg_image > pixel_thresh
        rows_hot = \
            raw_mask.sum(axis=1) / raw_mask.shape[1] >= row_thresh

        # Exclude the middle region
        wall_extent_max_px = round(bg_image.shape[0] * wall_extent_max)
        rows_hot[wall_extent_max_px:-wall_extent_max_px] = False

        # Fill in gaps in walls
        hot_rows = np.asarray(rows_hot.nonzero(), dtype=int)
        mid_row_idx = int(raw_mask.shape[0] / 2)
        last_hot_top = hot_rows[hot_rows <= mid_row_idx].max(initial=0)
        first_hot_btm = hot_rows[hot_rows > mid_row_idx].min(
            initial=raw_mask.shape[0]-1)
        rows_hot[:last_hot_top+1] = True
        rows_hot[first_hot_btm:] = True

        # Generate the mask
        align_mask = np.zeros(bg_image.shape)
        align_mask[rows_hot, ...] = True

        # Clear the outer margin
        mask_margin_px = round(bg_image.shape[0] * mask_margin)
        align_mask[:mask_margin_px+1] = 0
        align_mask[align_mask.shape[0] - mask_margin_px:] = 0
        align_mask[:, :mask_margin_px+1] = 0
        align_mask[:, align_mask.shape[1] - mask_margin_px:] = 0

        # Calculate crop bounds
        wall_margin_px = math.ceil(wall_margin * align_mask.shape[0])
        crop_bounds = (0, bg_image.shape[0])
        # FIXME um...
        return align_mask, crop_bounds

    @classmethod
    def prep_align_bg(cls, bg_image: np.ndarray, align_mask: np.ndarray,
                      cutout_height: float) -> tuple[np.ndarray, np.ndarray]:
        hroi_height_px = round(0.5 * bg_image.shape[0] * (1. - cutout_height))
        ref_image = np.concatenate((
            bg_image[:hroi_height_px+1],
            bg_image[-hroi_height_px:]
        ))
        ref_mask = np.concatenate((
            align_mask[:hroi_height_px+1],
            align_mask[-hroi_height_px:]
        ))
        return ref_image, ref_mask

    @classmethod
    def prep_align_fg(cls, fg_image: np.ndarray, cutout_height: float) \
            -> np.ndarray:
        hroi_height_px = round(0.5 * fg_image.shape[0] * (1. - cutout_height))
        moving_image = np.concatenate((
            fg_image[:hroi_height_px+1],
            fg_image[-hroi_height_px:]
        ))
        return moving_image

    @classmethod
    def prep_cleanup_bg(cls, bg_image: np.ndarray,
                        crop_bounds: tuple[int, int] | None,
                        bright_thresh: float,
                        grow_factor: float) -> np.ndarray:
        feature_mask = bg_image > bright_thresh
        grow_px = bg_image.shape[0] * grow_factor
        feature_mask = skimage.segmentation.expand_labels(
            feature_mask, grow_px)
        if crop_bounds is not None:
            feature_mask[:crop_bounds[0]] = 1
            feature_mask[crop_bounds[1]+1:] = 1
        return ~feature_mask

    @classmethod
    def prep_cleanup_fg(cls, fg_subbed: np.ndarray, bright_thresh: float,
                        size_thresh: float, grow_factor: float,
                        grow_cycles: int) -> np.ndarray:
        # Select bright features
        feature_mask = fg_subbed > bright_thresh
        labeled = skimage.measure.label(feature_mask)

        # Exclude small regions
        for props in skimage.measure.regionprops(labeled):
            if props.area < labeled.size * size_thresh:
                feature_mask[labeled == props.label] = 0

        # Grow
        grow_px = fg_subbed.shape[0] * grow_factor
        for _ in range(grow_cycles):
            feature_mask = skimage.segmentation.expand_labels(feature_mask,
                                                              grow_px)

        # Fill in hollow regions
        labeled = skimage.measure.label(~feature_mask)
        propses = skimage.measure.regionprops(labeled)
        areas = [(props.area, props.label) for props in propses]
        areas.sort()
        if areas:
            areas.pop()
        for _, label in areas:
            feature_mask[labeled == label] = 1

        # Also mask out undefined areas due to image alignment
        feature_mask[np.isnan(fg_subbed)] = 1

        return ~feature_mask

    @classmethod
    def align_and_subtract(cls, bg_image: np.ndarray, fg_image: np.ndarray,
                           mpcc_overlap_ratio: float,
                           ref_image: np.ndarray | None = None,
                           ref_mask: np.ndarray | None = None,
                           moving_image: np.ndarray | None = None,
                           align_mask: np.ndarray | None = None,
                           cutout_height: int | None = None,
                           oversub_factor: float = 1.0
                           ) -> tuple[np.ndarray, tuple[float, float],
                                      dict[str, Any]]:
        # Generate moving image if not supplied
        if moving_image is None:
            if cutout_height is None:
                raise ValueError(
                    "either moving_image or cutout_height must be supplied")
            moving_image = cls.prep_align_fg(fg_image, cutout_height)

        # Generate reference image if not supplied
        if any(x is None for x in (ref_image, ref_mask)):
            if any(x is None for x in (align_mask, cutout_height)):
                raise ValueError(
                    "either align_mask & cutout_height"
                    + "or ref_image & ref_mask must be supplied"
                )
            assert align_mask is not None
            assert cutout_height is not None
            ref_image, ref_mask = cls.prep_align_bg(
                bg_image, align_mask, cutout_height)
        assert ref_image is not None
        assert ref_mask is not None

        # Perform alignment, subtract, clip to 0
        fg_shifted, shift = cls.align_images(
            ref_image, ref_mask, moving_image, fg_image, mpcc_overlap_ratio)
        fg_subbed = fg_shifted - oversub_factor * bg_image
        fg_subbed[fg_subbed < 0.] = 0.

        info = {
            "moving_image": moving_image,
            "ref_image": ref_image,
            "ref_mask": ref_mask,
        }
        return fg_subbed, shift, info

    @classmethod
    def cleanup_features(cls, fg_subbed: np.ndarray, gaussian_sigma: float,
                         bg_feature_mask: np.ndarray | None = None,
                         bg_image: np.ndarray | None = None,
                         crop_bounds: tuple[int, int] | None = None,
                         bg_bright_thresh: float | None = None,
                         bg_grow_factor: float | None = None,
                         fg_feature_mask: np.ndarray | None = None,
                         fg_bright_thresh: float | None = None,
                         fg_size_thresh: float | None = None,
                         fg_grow_factor: float | None = None,
                         fg_grow_cycles: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
        info: dict[str, Any] = {}

        # Generate background feature mask if not supplied
        if bg_feature_mask is None:
            if any(x is None for x in
                   (bg_image, bg_bright_thresh, bg_grow_factor)):
                raise ValueError(
                    "either bg_feature_mask "
                    + "or bg_image & crop_bounds & bg_bright_thresh "
                    + "& bg_grow_factor must be supplied")
            assert bg_image is not None
            assert crop_bounds is not None
            assert bg_bright_thresh is not None
            assert bg_grow_factor is not None
            bg_feature_mask, bg_info = cls.prep_cleanup_bg(
                bg_image, crop_bounds, bg_bright_thresh, bg_grow_factor)
            update_info(info, bg_info)

        # Generate foreground feature mask if not supplied
        if fg_feature_mask is None:
            if None in (fg_bright_thresh, fg_size_thresh, fg_grow_factor,
                        fg_grow_cycles):
                raise ValueError(
                    "either fg_feature_mask or fg_bright_thresh"
                    + "& fg_size_thresh & fg_grow_factor & fg_grow_cycles"
                    + "must be supplied")
            assert fg_bright_thresh is not None
            assert fg_size_thresh is not None
            assert fg_grow_factor is not None
            assert fg_grow_cycles is not None
            fg_feature_mask = cls.prep_cleanup_fg(fg_subbed, fg_bright_thresh,
                                                  fg_size_thresh,
                                                  fg_grow_factor,
                                                  fg_grow_cycles)

        # Generate composite mask and apply to foreground image
        feature_mask = fg_feature_mask & bg_feature_mask
        if gaussian_sigma is not None:
            feature_mask_soft = skimage.filters.gaussian(
                feature_mask, sigma=gaussian_sigma)
        else:
            feature_mask_soft = feature_mask
        fg_masked = fg_subbed.copy()
        fg_masked *= feature_mask_soft

        info.update({
            "fg_feature_mask": fg_feature_mask,
            "bg_feature_mask": bg_feature_mask,
            "feature_mask_soft": feature_mask_soft,
        })
        return fg_masked, feature_mask, info

    def set_bg_image(self, bg_image: np.ndarray) -> dict[str, Any]:
        # Rescale and gaussian filter
        bg_scaled = self.rescale_and_blur(
            bg_image,
            self.input_rescale_factor,
            self.input_gaussian_sigma
        )

        # Generate reference image and mask for alignment step
        align_mask, self._crop_bounds = self.make_align_mask(
            bg_scaled,
            self.align_bg_pixel_thresh,
            self.align_bg_row_thresh,
            self.align_mask_margin,
            self.cleanup_wall_margin,
            self.align_wall_extent_max,
        )
        self._ref_image, self._ref_mask = self.prep_align_bg(
            bg_scaled, align_mask, self.align_cutout_height)

        # Adjust crop_bounds to account for crop
        # TODO: Rename crop_bounds to something else
        # TODO: This is ugly, hide this elsewhere or do it more elegantly
        if self.crop_roi_size is not None:
            y_top = ((bg_scaled.shape[0] - self.crop_roi_size[1]) // 2)
            crop_bounds_adj = (
                max(self._crop_bounds[0] - y_top, 0),
                min(self._crop_bounds[1] - y_top, self.crop_roi_size[1] - 1)
            )
            bg_cropped = self.crop_rect_centered(bg_scaled, *self.crop_roi_size)
        else:
            crop_bounds_adj = self._crop_bounds
            bg_cropped = bg_scaled

        # Generate feature mask for cleanup step
        self._bg_feature_mask = self.prep_cleanup_bg(
            bg_cropped,
            crop_bounds_adj,
            self.cleanup_bg_brt_thresh,
            self.cleanup_bg_grow_factor
            )

        self._bg_cropped = bg_cropped

        info = {
            "bg_scaled": bg_scaled,
            "bg_cropped": bg_cropped,
            "align_mask": align_mask,
            "crop_bounds": self._crop_bounds,
            "ref_image": self._ref_image,
            "ref_mask": self._ref_mask,
            "bg_feature_mask": self._bg_feature_mask
        }
        return info

    def process_fg_image(self, fg_image: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self._bg_cropped is None:
            raise RuntimeError(
                "Must call set_bg_image() before process_fg_image()")

        fg_scaled = self.rescale_and_blur(
            fg_image,
            self.input_rescale_factor,
            self.input_gaussian_sigma
        )

        moving_image = self.prep_align_fg(
            fg_scaled, cutout_height=self.align_cutout_height)
        if self.crop_roi_size is not None:
            fg_cropped = self.crop_rect_centered(
                fg_scaled, *self.crop_roi_size)
        else:
            fg_cropped = fg_scaled

        fg_subbed, shift, sub_info = self.align_and_subtract(
            self._bg_cropped,
            fg_cropped,
            self.align_mpcc_overlap_ratio,
            moving_image=moving_image,
            ref_image=self._ref_image,
            ref_mask=self._ref_mask,
            oversub_factor=self.align_oversub_factor
            )

        fg_cleaned, feature_mask, clean_info = self.cleanup_features(
            fg_subbed,
            bg_feature_mask=self._bg_feature_mask,
            fg_bright_thresh=self.cleanup_fg_brt_thresh,
            fg_size_thresh=self.cleanup_fg_size_thresh,
            fg_grow_factor=self.cleanup_fg_grow_factor,
            fg_grow_cycles=self.cleanup_fg_grow_cycles,
            gaussian_sigma=self.cleanup_gaussian_sigma
            )

        info = {
            "fg_scaled": fg_scaled,
            "fg_cropped": fg_cropped,
            "fg_subbed": fg_subbed,
            "shift": shift
        }
        update_info(info, sub_info)
        update_info(info, clean_info)
        return fg_cleaned, feature_mask, info


class Cci2ImagePreprocessing(CciImagePreprocessing):
    input_scale_factor: float
    fid_roi_width: float
    fid_roi_height: float
    fid_roi_ctr_x: float
    fid_roi_ctr_y: float
    fid_brt_thresh: float
    align_upsample_factor: float
    align_overlap_ratio: float
    margin: float
    corner_zone_radius: float
    cleanup_bg_brt_thresh: float
    cleanup_bg_grow_factor: float
    cleanup_use_anticounting: bool
    cleanup_anticounting_y_offs: float | None
    cleanup_anticounting_excl_w: int | None
    cleanup_anticounting_excl_h: int | None
    cleanup_fg_brt_thresh: float
    cleanup_fg_size_thresh: float
    cleanup_fg_grow_factor: float
    cleanup_fg_grow_cycles: int
    cleanup_gaussian_sigma: float

    # pylint: disable=unused-argument
    @require_b_not_none_if_a(
        'cleanup_use_anticounting',
        [
            'cleanup_anticounting_y_offs',
            'cleanup_anticounting_excl_w',
            'cleanup_anticounting_excl_h'
            ]
        )
    def __init__(
            self, *,
            input_scale_factor: float,
            fid_roi_width: float,
            fid_roi_height: float,
            fid_roi_ctr_x: float,
            fid_roi_ctr_y: float,
            fid_brt_thresh: float,
            align_upsample_factor: float,
            align_overlap_ratio: float,
            margin: float,
            corner_zone_radius: float,
            cleanup_bg_brt_thresh: float,
            cleanup_bg_grow_factor: float,
            cleanup_use_anticounting: bool,
            cleanup_anticounting_y_offs: float | None,
            cleanup_anticounting_excl_w: int | None,
            cleanup_anticounting_excl_h: int | None,
            cleanup_fg_brt_thresh: float,
            cleanup_fg_size_thresh: float,
            cleanup_fg_grow_factor: float,
            cleanup_fg_grow_cycles: int,
            cleanup_gaussian_sigma: float,
            ):
        for key, value in locals().items():
            if key not in ("self",):
                setattr(self, key, value)
        self._bg_img: np.ndarray | None = None
        self._counting_obj: CciCellCounting | None = None

    def set_counting_obj(self, counting_obj: CciCellCounting):
        self._counting_obj = counting_obj

    @staticmethod
    def _subtract_imgs(img_a: np.ndarray, img_b: np.ndarray,
                       oversub_factor: float = 1.,
                       clip_zero: bool = False):
        # TODO move this to common?
        subbed_img = img_a - oversub_factor * img_b
        if clip_zero:
            subbed_img[subbed_img < 0.] = 0.
        return subbed_img

    @classmethod
    def prep_cleanup_fg(
            cls, *,
            margin: float,
            corner_zone_radius: float,
            fg_subbed: np.ndarray,
            bright_thresh: float,
            size_thresh: float,
            grow_factor: float,
            grow_cycles: int) -> np.ndarray:
        # FIXME: De-duplicate the CCI1 and CCI2 implementations

        # Build the vignette mask
        # XXXX FIXME TODO: dedicated method for building vignette mask
        #   since we need it in a couple places
        vignette_mask = np.ones(fg_subbed.shape, dtype='bool')
        cls.fill_margin(vignette_mask, margin, 0)
        cls.fill_outside_circle(vignette_mask, corner_zone_radius, 0)

        # Select bright features
        feature_mask = fg_subbed > bright_thresh
        labeled = skimage.measure.label(feature_mask)

        # Exclude small regions
        for props in skimage.measure.regionprops(labeled):
            if props.area < labeled.size * size_thresh:
                feature_mask[labeled == props.label] = 0

        # Grow
        grow_px = fg_subbed.shape[0] * grow_factor
        for _ in range(grow_cycles):
            feature_mask = skimage.segmentation.expand_labels(feature_mask,
                                                              grow_px)

        # Merge with vignette mask to close bubbles that extend out of view
        feature_mask |= ~vignette_mask

        # Fill in hollow regions
        labeled = skimage.measure.label(~feature_mask)
        propses = skimage.measure.regionprops(labeled)
        areas = [(props.area, props.label) for props in propses]
        areas.sort()
        if areas:
            areas.pop()
        for _, label in areas:
            feature_mask[labeled == label] = 1

        # Also mask out any undefined areas created by shifting if applicable
        feature_mask[np.isnan(fg_subbed)] = 1

        return ~feature_mask

    @require_b_not_none_if_a(
        'use_anticounting',
        ['anticounting_y_offs', 'anticounting_excl_w', 'anticounting_excl_h'])
    @classmethod
    def prep_cleanup_bg(cls, bg_image: np.ndarray,
                        bright_thresh: float,
                        grow_factor: float,
                        use_anticounting: bool,
                        anticounting_y_offs: float | None = None,
                        anticounting_excl_w: int | None = None,
                        anticounting_excl_h: int | None = None,
                        counting_obj: CciCellCounting | None = None
                        ) -> np.ndarray:
        if use_anticounting and counting_obj is None:
            raise ValueError(
                "counting_obj must not be None if use_anticounting is True")

        feature_mask = bg_image > bright_thresh
        grow_px = bg_image.shape[0] * grow_factor
        feature_mask = skimage.segmentation.expand_labels(
            feature_mask, grow_px)

        if use_anticounting:
            assert counting_obj is not None  # checked above
            for i in range(1): # TODO: Make no. of cycles a config param?
                anticounting_cell_locs: set[tuple[int, int]] = set()
                #for x_offs in (-anticounting_y_offs, anticounting_y_offs):
                for x_offs in (0,):
                    for y_offs in (-anticounting_y_offs, anticounting_y_offs):
                        if y_offs is None:
                            fg_subbed = cls._subtract_imgs(
                                bg_image,
                                skimage.filters.gaussian(bg_image, sigma=2.0)
                                )
                        else:
                            shifted_img = scipy.ndimage.shift(
                                bg_image, (y_offs, x_offs), mode='reflect')
                            fg_subbed = cls._subtract_imgs(shifted_img, bg_image)
                            fg_subbed *= 1.25 # FIXME TODO
                        cell_locs = counting_obj.process_fg_image(
                            fg_subbed, ~feature_mask)[0]
                        anticounting_cell_locs.update((y, x) for (y, x) in cell_locs)
                for ctr_i, ctr_j in anticounting_cell_locs:
                    # XXXX FIXME: bounds checks
                    feature_mask[
                        ctr_i-anticounting_excl_h:ctr_i+anticounting_excl_h+1,
                        ctr_j-anticounting_excl_w:ctr_j+anticounting_excl_w+1
                        ] = True
                logger.debug(f"Anticounting phase {i} masked "
                             f"{len(anticounting_cell_locs)} locations")

        return ~feature_mask

    def _prep_alignment_img(self, image: np.ndarray):
        out_img = self.crop_rect_propo(
            image,
            self.fid_roi_width,
            self.fid_roi_height,
            self.fid_roi_ctr_x,
            self.fid_roi_ctr_y)
        out_img[out_img < self.fid_brt_thresh] = self.fid_brt_thresh
        return out_img

    def _rescale_input_img(self, image: np.ndarray):
        return skimage.transform.rescale(
            image, self.input_scale_factor, anti_aliasing=True)

    def set_bg_image(self, bg_image: np.ndarray) -> dict[str, Any]:
        bg_image = self._rescale_input_img(bg_image)
        self._bg_img = bg_image
        self._align_ref_img = self._prep_alignment_img(bg_image)
        anticounting_kwargs = {}
        if self.cleanup_use_anticounting:
            anticounting_kwargs.update({
                'use_anticounting': True,
                'anticounting_y_offs': self.cleanup_anticounting_y_offs,
                'anticounting_excl_w': self.cleanup_anticounting_excl_w,
                'anticounting_excl_h': self.cleanup_anticounting_excl_h,
                'counting_obj': self._counting_obj
                })
        self._bg_feature_mask = self.prep_cleanup_bg(
            bg_image,
            self.cleanup_bg_brt_thresh,
            self.cleanup_bg_grow_factor,
            **anticounting_kwargs
            )
        info: dict[str, Any] = {
            'bg_feature_mask': self._bg_feature_mask
        }
        return info

    def process_fg_image(self, fg_image: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self._bg_img is None:
            raise RuntimeError(
                "Must call set_bg_image() before process_fg_image()")
        fg_image = self._rescale_input_img(fg_image)
        mvg_img = self._prep_alignment_img(fg_image)

        fg_shifted, fg_shift, align_info = self.align_images(
            ref_image=self._align_ref_img,
            moving_image=mvg_img,
            fg_image=fg_image,
            upsample_factor=self.align_upsample_factor,
            overlap_ratio=self.align_overlap_ratio,
            )

        # TODO: Formal implementation of this
        # (and also rewrite the alignment routine)
        for v in fg_shift: # XXXX
            if abs(v) > 30.: # XXXX
                fg_shifted = fg_image # XXXX
                logger.warning(f"Alignment aborted (shift was {fg_shift})") # XXXX
                break # XXXX

        fg_subbed = self._subtract_imgs(fg_shifted, self._bg_img)

        feature_mask = self.prep_cleanup_fg(
            fg_subbed=fg_subbed,
            margin=self.margin,
            corner_zone_radius=self.corner_zone_radius,
            bright_thresh=self.cleanup_fg_brt_thresh,
            size_thresh=self.cleanup_fg_size_thresh,
            grow_factor=self.cleanup_fg_grow_factor,
            grow_cycles=self.cleanup_fg_grow_cycles)
        feature_mask *= self._bg_feature_mask
        feature_mask_soft = skimage.filters.gaussian(
            feature_mask, sigma=self.cleanup_gaussian_sigma)
        fg_cleaned = fg_subbed.copy()
        fg_cleaned *= feature_mask_soft
        # TODO: Instead of blacking out areas,
        # fill with local average value?

        info: dict[str, Any] = {
            'align_mvg_img': mvg_img,
            'align_ref_img': self._align_ref_img,
            'align_shift': fg_shift,
            }
        update_info(info, align_info)

        return fg_cleaned, feature_mask, info
