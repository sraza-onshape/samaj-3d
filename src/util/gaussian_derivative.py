from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .ops import convolution as convolution_op
from .ops import Filter2D

from .gaussian_base import BaseGaussianFilter


class GaussianDerivativeFilter(BaseGaussianFilter):
    def __init__(self, sigma: int = 1) -> None:
        super().__init__(sigma)
        # create the 2D Gaussian filter
        self.filter_matrix = self.create_gaussian_filter(self.sigma)

    def _compute_derivatives(
        self, image: List[List[float]], padding_type: str = "zero"
    ) -> Tuple[List[List[float]]]:
        """Setter function for the partial derivatives of the iage"""
        # 2) "separate into x and y"
        # for x: convolve filter with the horizontal Sobel
        gaussian_derivative_x_filter = convolution_op(
            self.filter_matrix,
            Filter2D.HORIZONTAL_SOBEL_FILTER.value,
            padding_type=padding_type,
        )
        # then convolve the image with the convolved filter
        partial_derivative_x = convolution_op(
            image, gaussian_derivative_x_filter, padding_type=padding_type
        )
        # do the same for y
        gaussian_derivative_y_filter = convolution_op(
            self.filter_matrix,
            Filter2D.VERTICAL_SOBEL_FILTER.value,
            padding_type=padding_type,
        )
        partial_derivative_y = convolution_op(
            image, gaussian_derivative_y_filter, padding_type=padding_type
        )
        return partial_derivative_x, partial_derivative_y

    def _compute_magnitude(self, partial_derivative_x, partial_derivative_y):
        # convert to numpy array (only for the purpose of making element-wise computations easier)
        partial_derivative_x = np.array(partial_derivative_x)
        partial_derivative_y = np.array(partial_derivative_y)
        magnitude = np.sqrt(
            (partial_derivative_x**2) + (partial_derivative_y**2)
        )
        return magnitude

    def detect_edges(
        self,
        image: List[List[int]],
        threshold: float = float("-inf"),
        padding_type: str = "zero",
    ) -> np.array:
        """Do edge detection: use the convolved images in the magnitude formula --> visualize it"""
        ### DRIVER
        partial_derivative_x, partial_derivative_y = self._compute_derivatives(
            image, padding_type=padding_type
        )
        edges = self._compute_magnitude(partial_derivative_x, partial_derivative_y)
        # apply the threshold to zero out extraneous magnitudes
        edges = np.where(edges > threshold, edges, 0)  # default: no change

        return edges

    def suppress_edge_pixels(
        self,
        partial_derivative_x: List[List[float]],
        partial_derivative_y: List[List[float]],
        edge_image: np.array,
    ) -> np.array:
        """Implement non-maximum suppression."""

        ### HELPERS
        def _find_neighbors(x: int, y: int, theta: float) -> List[Tuple[int, int]]:
            """
            Returns the coordinates of the neighboring pixels, or None if out of bounds.
            The fractions used in the `if` conditions here represent angle measures around the unit circle.
            """
            lower_px_indices, higher_px_indices = None, None
            # edge is vertical, get left-right neighbors
            if ((3 / 8) <= theta < (5 / 8)) or ((11 / 8) <= theta < (13 / 8)):
                lower_px_indices = (x - 1, y)
                higher_px_indices = (x + 1, y)
            # edge is horizontal, get up-down neighbors
            elif (
                (theta >= (15 / 8)) or (theta < (1 / 8)) or ((7 / 8) <= theta < (9 / 8))
            ):
                lower_px_indices = (x, y - 1)
                higher_px_indices = (x, y + 1)
            # edge is right-left diagonal, get left-right diagonal neighbors
            elif ((1 / 8) <= theta < (3 / 8)) or ((9 / 8) < theta < (11 / 8)):
                lower_px_indices = (x - 1, y - 1)
                higher_px_indices = (x + 1, y + 1)
            # edge is left-right diagonal, get right-left diagonal neighors
            else:
                lower_px_indices = (x + 1, y + 1)
                higher_px_indices = (x - 1, y - 1)

            # last thing: if neighbor indices out of bounds, return None instead
            final_output = list()
            for coords in [lower_px_indices, higher_px_indices]:
                if (
                    -1 < coords[0] < edge_image.shape[0]
                    and -1 < coords[1] < edge_image.shape[1]
                ):
                    final_output.append(coords)
                else:
                    final_output.append(None)
            return final_output

        ### DRIVER
        # compute orientation of gradient
        orientation_image = (
            np.arctan2(partial_derivative_y, partial_derivative_x) + (2 * np.pi) / np.pi
        )
        orientation_image = np.arctan2(partial_derivative_y, partial_derivative_x)
        # constrain the values to the 0-2pi range
        orientation_image = np.where(
            orientation_image < 0, orientation_image + (2 * np.pi), orientation_image
        )
        # reduce the angle measures to coefficients of pi
        orientation_image /= np.pi
        # C: formulate the suppressed version of the edge image
        suppressed = edge_image.copy()
        for index_row in range(edge_image.shape[0]):
            for index_col in range(edge_image.shape[1]):
                edge_px = edge_image[index_row][index_col]
                if edge_px > 0:
                    # get neighboring pixels
                    gradient_angle = orientation_image[index_row][index_col]
                    lower_px_indices, higher_px_indices = _find_neighbors(
                        index_row, index_col, gradient_angle
                    )
                    lower_px = 0
                    if lower_px_indices is not None:
                        lower_px = edge_image[lower_px_indices[0]][lower_px_indices[1]]
                    higher_px = 0
                    if higher_px_indices is not None:
                        higher_px = edge_image[higher_px_indices[0]][
                            higher_px_indices[1]
                        ]
                    # eliminate the neighboring pixels?
                    if edge_px > lower_px and edge_px > higher_px:
                        if lower_px_indices is not None:
                            suppressed[lower_px_indices[0]][lower_px_indices[1]] = 0
                        if higher_px_indices is not None:
                            suppressed[higher_px_indices[0]][higher_px_indices[1]] = 0
                    # eliminate the pixel itself?
                    else:
                        suppressed[index_row][index_col] = 0
        return suppressed

    @classmethod
    def detect_edges_and_visualize(
        cls: "GaussianDerivativeFilter",
        image: List[List[int]],
        image_name: str,
        sigma: int = 1,
        threshold: float = float("-inf"),
        use_non_max_suppression: bool = False,
        padding_type: str = "zero",
    ) -> np.array:
        """Convenience wrapper + uses Matplotlib to plot the edges found.

        Returns: np.array: the edges detected. If non-max supression is set to True,
                 the return matrix will have utilized non-max suppression.
        """
        edge_detector = cls(sigma=sigma)
        return_image = detected_edges = edge_detector.detect_edges(
            image, threshold, padding_type=padding_type
        )

        if not use_non_max_suppression:
            plt.imshow(detected_edges, cmap="gray", vmin=0, vmax=255)
            plt.title(f"{image_name} Edges, sigma={sigma}, threshold={threshold}")
        else:
            (
                partial_derivative_x,
                partial_derivative_y,
            ) = edge_detector._compute_derivatives(image, padding_type=padding_type)
            return_image = suppressed_edges = edge_detector.suppress_edge_pixels(
                partial_derivative_x,
                partial_derivative_y,
                detected_edges,
            )
            plt.imshow(suppressed_edges, cmap="gray", vmin=0, vmax=255)
            plt.title(
                f"{image_name} Edges (w/ Non-max Suppression), sigma={sigma}, threshold={threshold}"
            )
        plt.show()
        return return_image


if __name__ == "__main__":
    # TODO: add test cases
    print("code interprets without errors")
