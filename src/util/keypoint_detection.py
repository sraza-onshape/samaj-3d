from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .gaussian_base import BaseGaussianFilter
from .ops import (
    convolution as convolution_op,
    non_max_suppression_2D as suppression_op,
    Filter2D,
)


class AbstractKeypointDetector:
    """This class is intentionally left blank."""

    # TODO: utilize the `abc.ABC` module here, to learn cool new Python skills :)
    pass


class HessianDetector(AbstractKeypointDetector):
    DEFAULT_PERCENTILE_FOR_DETERMINANT = 75  # totally arbitrary

    def __init__(self, threshold: float = None):
        self.threshold = threshold  # should be an exact value, expected to be in the range of the image determinant

    def _set_threshold(self, values: np.array, percentile: float) -> None:
        """Sets the determinant based on a percentile of an n-dimensional array (representing the range of some function)."""
        self.threshold = np.percentile(values, percentile)

    def _get_threshold(self) -> float:
        return self.threshold

    def find_keypoints(
        self, image: np.array, percentile: float = DEFAULT_PERCENTILE_FOR_DETERMINANT
    ) -> np.array:
        """Utilizes the determinant of the Hessian matrix to find keypoints."""
        ### DRIVER
        # compute the second order partial derivatives
        (
            second_order_derivator_x,
            second_order_derivator_y,
            second_order_derivator_xy,
        ) = (
            convolution_op(
                Filter2D.HORIZONTAL_SOBEL_FILTER.value,
                Filter2D.HORIZONTAL_SOBEL_FILTER.value,
                padding_type="zero",
            ),
            convolution_op(
                Filter2D.VERTICAL_SOBEL_FILTER.value,
                Filter2D.VERTICAL_SOBEL_FILTER.value,
                padding_type="zero",
            ),
            convolution_op(
                Filter2D.HORIZONTAL_SOBEL_FILTER.value,
                Filter2D.VERTICAL_SOBEL_FILTER.value,
                padding_type="zero",
            ),
        )

        # apply a Gaussian smoothening
        image_list = image.tolist()
        smoother = BaseGaussianFilter()
        image_list = smoother.smooth(image_list)

        # formulate the Hessian matrix
        hessian_xx = np.array(
            convolution_op(image_list, second_order_derivator_x, padding_type="zero")
        )
        hessian_yy = np.array(
            convolution_op(image_list, second_order_derivator_y, padding_type="zero")
        )
        hessian_xy = np.array(
            convolution_op(image_list, second_order_derivator_xy, padding_type="zero")
        )

        # find the determinant
        determinant_hessian = hessian_xx * hessian_yy - (hessian_xy**2)

        # (if needed) set the threshold (should be an actual value, in the range of determinant)
        lower_threshold = self.threshold
        if lower_threshold is None and (percentile is not None):
            self._set_threshold(determinant_hessian, percentile)
            lower_threshold = self._get_threshold()

        # zero out non-keypoints - via thresholding
        keypoints = np.where(
            determinant_hessian > lower_threshold, determinant_hessian, 0
        )

        # zero out any non-keypoints - via non max suppression
        keypoints_suppressed = suppression_op(keypoints)

        keypoint_locations = [[], []]
        for y in range(keypoints_suppressed.shape[0]):
            for x in range(keypoints_suppressed.shape[1]):
                if keypoints_suppressed[y][x] > 0:
                    keypoint_locations[0].append(y)
                    keypoint_locations[1].append(x)
        return np.array(keypoint_locations)

    @classmethod
    def find_keypoints_and_visualize(
        cls: "HessianDetector",
        image: np.array,
        image_name: str,
        percentile: float = DEFAULT_PERCENTILE_FOR_DETERMINANT,
    ) -> None:
        """Convenience wrapper for both execution + visualization."""
        # run the algorithm
        keypoint_detector = cls(threshold=None)
        keypoint_locations = keypoint_detector.find_keypoints(image, percentile)
        # show the results -
        plt.imshow(image, cmap="gray")
        plt.scatter(y=keypoint_locations[0], x=keypoint_locations[1], color="red")
        plt.title(f'Keypoints Detected for Image: "{image_name}"')
        plt.show()


if __name__ == "__main__":
    print("script interprets without errors :)")
