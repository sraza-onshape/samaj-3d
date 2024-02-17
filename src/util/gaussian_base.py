import matplotlib.pyplot as plt
import numpy as np
from typing import List

from .ops import convolution as convolution_op


class BaseGaussianFilter:
    """A 2D Gaussian filter to use for smoothening images."""

    def __init__(self, sigma: int = 1) -> None:
        self._set_parameters(sigma)

    def _set_parameters(self, sigma: int):
        self.sigma = sigma
        self.filter_width = 6 * sigma + 1

    def _get_element_in_filter(self, x: int, y: int) -> float:
        """Samples from a 2D Gaussian to determine what value goes in a given element."""
        exponent = -1 * (((x**2) + (y**2)) / (2 * (self.sigma**2)))
        power = np.exp(exponent)  # Euler's number (e) is the base
        return power / (2 * np.pi * (self.sigma**2))

    def create_gaussian_filter(self, sigma: int = None) -> List[List[float]]:
        """create 2D array for for the filter"""
        if sigma:
            self._set_parameters(sigma)
        center_pos = 3 * self.sigma  # in both axes
        filter = list()

        for x_index in range(self.filter_width):
            filter_row = list()
            for y_index in range(self.filter_width):
                # coordinates are "centered" at the center element in the matrix
                x_coord, y_coord = abs(x_index - center_pos), abs(y_index - center_pos)
                filter_row.append(self._get_element_in_filter(x_coord, y_coord))
            filter.append(filter_row)

        return filter

    def smooth(
        self,
        image: List[List[float]],
        filter: List[List[float]] = None,
        padding_type="zero",
    ) -> List[List[float]]:
        """Applies this Gaussian filter to an input image."""
        if not filter:
            filter = self.create_gaussian_filter()
        return convolution_op(image, filter, padding_type=padding_type)

    @classmethod
    def smooth_image_and_visualize(
        cls: "BaseGaussianFilter",
        image: List[List[int]],
        image_name: str,
        sigma: int = 1,
        padding_type: str = "zero",
    ) -> np.array:
        """Convenience wrapper + uses Matplotlib to plot the smoothed image.

        Returns: np.array: the output image. Will have same dimensions as the input.
        """
        smoother = cls(sigma=sigma)
        filtered_image = smoother.smooth(image, padding_type=padding_type)
        plt.imshow(filtered_image, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{image_name} after Filtering (w/ Gaussian Kernel), sigma={sigma}")
        plt.show()
        return filtered_image


if __name__ == "__main__":
    # test out the functions, ensure the coefs sum to approx. 1
    sigma = 1
    filter = BaseGaussianFilter()
    matrix = filter.create_gaussian_filter(sigma)
    print(f"Sum of values: {sum([sum(row) for row in matrix])}")
    print(
        f"The filter itself: {np.array(matrix)}"
    )  # using NumPy soley for printability
