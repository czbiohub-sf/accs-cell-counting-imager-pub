from typing import Any

import numpy as np
import scipy.ndimage
import skimage.io


def update_info(dest, src):
    dest.update({
        k: v for (k, v) in src.items()
        if k not in ("warnings",)})
    if "warnings" in dest and dest["warnings"]:
        warnings = dest["warnings"]
    else:
        warnings = []
    if "warnings" in src and src["warnings"]:
        warnings += src["warnings"]
    dest["warnings"] = warnings


class CciImageProcessingCommon:
    @staticmethod
    def read_image(path: str) -> np.ndarray:
        """
        Read an image file from a given path and return the image as an array.

        The current implementation assumes a 16-bit single-channel image.
        Handling of other formats may be added in future versions.

        Parameters
        ----------
        path : str
            Path to the image file.

        Returns
        -------
        image : ndarray
            The image as a 2D float array.
        """
        return skimage.io.imread(path).astype(np.float64) / 65535.

    @staticmethod
    def rescale_and_blur(image: np.ndarray,
                         scale_factor: float | None = None,
                         sigma: float | None = None) -> np.ndarray:
        """
        Rescale an image and apply a gaussian filter.

        Parameters
        ----------
        image : ndarray
            Input image as a 2D float array
        scale_factor : float or None, default=None
            The image will be rescaled by this factor, e.g. scale_factor=0.25
            would correspond to reducing the linear dimensions by 75%.
            If None, the rescale step is skipped.
        sigma : float or None, default=None
            Sigma value for gaussian filter applied after rescaling.
            If None, the blur step is skipped.

        Returns
        -------
        image : ndarray
            Resulting image as a 2D float array
        """
        if scale_factor is not None:
            image = skimage.transform.rescale(
                image,
                scale_factor,
                anti_aliasing=True
            )
        if sigma is not None:
            image = skimage.filters.gaussian(image, sigma)
        return image

    @classmethod
    def align_images(cls, ref_image: np.ndarray,
                     moving_image: np.ndarray, fg_image: np.ndarray,
                     ref_mask: np.ndarray | None = None,
                     upsample_factor: float | None = 1.,
                     overlap_ratio: float = 0.3) \
            -> tuple[np.ndarray, tuple[float, float], dict[str, Any]]:
        info = {}
        result = skimage.registration.phase_cross_correlation(
            ref_image,
            moving_image,
            reference_mask=ref_mask,
            upsample_factor=upsample_factor,
            overlap_ratio=overlap_ratio,
        )
        if ref_mask is not None:
            shift = result
        else:
            shift, info['pcc_error'], info['pcc_phasediff'] = result
        shift = tuple(map(float, shift))
        shifted_image = scipy.ndimage.shift(
            fg_image, shift, mode='reflect')
        return shifted_image, shift, info

    @staticmethod
    def crop_margin(image: np.ndarray, margin_x: int,
                    margin_y: int) -> np.ndarray:
        return image[margin_y:-margin_y, margin_x:-margin_x].copy()

    @staticmethod
    def crop_rect_centered(image: np.ndarray, rect_width: int,
                           rect_height: int):
        # TODO verify correct behavior
        img_h, img_w = image.shape
        x_left = (img_w - rect_width) // 2
        y_top = (img_h - rect_height) // 2
        return image[y_top:y_top+rect_height, x_left:x_left+rect_width].copy()

    @staticmethod
    def crop_rect_propo(image: np.ndarray, width: float, height: float,
                        ctr_x: float, ctr_y: float):
        width_px = round(image.shape[1] * width)
        height_px = round(image.shape[0] * height)
        x_left = int(image.shape[1] * (ctr_x - width / 2.))
        y_top = int(image.shape[0] * (ctr_y - height / 2.))
        return image[y_top:y_top+height_px, x_left:x_left+width_px].copy()

    @staticmethod
    def fill_margin(image: np.ndarray, margin: float, val: Any = True):
        margin_px = min(round(image.shape[0] * margin), min(image.shape)//2)
        image[:margin_px + 1, ...] = val
        image[image.shape[0] - margin_px - 1:, ...] = val
        image[..., :margin_px + 1] = val
        image[..., image.shape[1] - margin_px - 1:] = val

    @staticmethod
    def fill_outside_circle(image: np.ndarray, radius: float, val: Any = True):
        radius_px = round(image.shape[0] * radius)
        corner_mask = np.zeros(image.shape, dtype=bool)
        center = (round(image.shape[0] / 2), round(image.shape[1] / 2))
        disk = skimage.draw.disk(center, radius_px, shape=image.shape)
        corner_mask[disk] = 1
        image[corner_mask == 0] = val
