import abc, heapq
from abc import ABCMeta
from typing import List, Tuple

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np

from .gaussian_base import BaseGaussianFilter
from . import ops
from .ops import (
    Filter2D,
    SimilarityMeasure,
)


class BaseCornerDetector(metaclass=ABCMeta):
    @classmethod
    @abc.abstractmethod
    def execute_and_visualize(cls):
        pass


class HarrisCornerDetector(BaseCornerDetector):
    CORNER_RESPONSE_CONSTANT = 0.05
    TOP_MANY_FEATURES_TO_DETECT = 1000  # as outlined in the hw 3 description
    TOP_MANY_SIMILARITIES_TO_SELECT = 20  # as outlined in the hw 3 description

    def detect_features(
        self,
        image: np.ndarray,
        use_non_max_suppression: bool = False,
        derivative_operator_x: list[list[int]] = None,
        derivative_operator_y: list[list[int]] = None,
        gaussian_window: np.ndarray = None,
    ) -> np.ndarray:
        """
        Find potential corner points in an image.

        Parameters:
            image(np.ndarray): assumes a single grayscale image of shape NxM
            use_non_max_suppression(bool): whether or not to use non-max suppression
            derivative_operator_x(array-like): filter we'll use in computing
                the Harris corner response. Defaults to using the Horizontal Sobel.
            derivative_operator_y(array-like): filter we'll use in computing
                the Harris corner response. Defaults to using the Vertical Sobel.
            gaussian_window(array-like): optional Gaussian filter to smooth the image.
                Defaults to a 3x3 Gaussian filter that uses sigma = 1.

        Returns: np.ndarray: NxM matrix that represents the Harris corner response
        """

        ### HELPERS
        def _compute_derivatives_in_gaussian_window(
            image: np.ndarray,
            gaussian_window: List[List[float]],
        ) -> np.ndarray:
            # compute second order gradient
            (
                second_order_derivator_x,
                second_order_derivator_y,
                second_order_derivator_xy,
            ) = (
                ops.convolution(
                    derivative_operator_x,
                    derivative_operator_x,
                    padding_type="zero",
                ),
                ops.convolution(
                    derivative_operator_y,
                    derivative_operator_y,
                    padding_type="zero",
                ),
                ops.convolution(
                    derivative_operator_x,
                    derivative_operator_y,
                    padding_type="zero",
                ),
            )
            # image_list = image.tolist()
            (hessian_xx, hessian_yy, hessian_xy) = (
                np.array(
                    ops.convolution(
                        image, second_order_derivator_x, padding_type="zero"
                    )
                ),
                np.array(
                    ops.convolution(
                        image, second_order_derivator_y, padding_type="zero"
                    )
                ),
                np.array(
                    ops.convolution(
                        image, second_order_derivator_xy, padding_type="zero"
                    )
                ),
            )

            # compute the second moment matrix in a Gaussian window around each pixel
            (convolved_hessian_xx, convolved_hessian_yy, convolved_hessian_xy) = (
                np.array(
                    ops.convolution(hessian_xx, gaussian_window, padding_type="zero")
                ),
                np.array(
                    ops.convolution(hessian_yy, gaussian_window, padding_type="zero")
                ),
                np.array(
                    ops.convolution(hessian_xy, gaussian_window, padding_type="zero")
                ),
            )

            return (convolved_hessian_xx, convolved_hessian_yy, convolved_hessian_xy)

        def _compute_corner_response(
            kernel: List[List[float]],
            # these 3 have the same dims as the input image
            convolved_hessian_xx: np.ndarray,
            convolved_hessian_yy: np.ndarray,
            convolved_hessian_xy: np.ndarray,
            stride: int = 1,
        ) -> np.array:
            """Evaluate the Harris corner response function, to get points of interest."""
            # ensure the corner response matrix has the same dims as the input image
            convolved_hessian_xx, _, _ = ops.pad(
                convolved_hessian_xx, kernel, stride, "zero"
            )
            convolved_hessian_yy, _, _ = ops.pad(
                convolved_hessian_yy, kernel, stride, "zero"
            )
            convolved_hessian_xy, _, _ = ops.pad(
                convolved_hessian_xy, kernel, stride, "zero"
            )

            # computation begins below - TODO[Zain]: try to make this more DRY
            corner_response = list()
            kernel_h, kernel_w = len(kernel), len(kernel[0])
            # iterate over the rows and columns
            starting_row_ndx = 0
            while starting_row_ndx <= len(convolved_hessian_xy) - kernel_h:
                # convolve the next row of this response
                response_row = list()
                starting_col_ndx = 0
                while starting_col_ndx <= len(convolved_hessian_xy[0]) - kernel_w:
                    # compute the response for this window
                    col_index = starting_col_ndx
                    block_of_derivative_xx = convolved_hessian_xx[
                        starting_row_ndx : (kernel_h + starting_row_ndx),
                        col_index : (kernel_w + col_index),
                    ]
                    block_of_derivative_xy = convolved_hessian_xy[
                        starting_row_ndx : (kernel_h + starting_row_ndx),
                        col_index : (kernel_w + col_index),
                    ]
                    block_of_derivative_yy = convolved_hessian_yy[
                        starting_row_ndx : (kernel_h + starting_row_ndx),
                        col_index : (kernel_w + col_index),
                    ]
                    second_moment_matrix_elements = [
                        np.sum(block_of_derivative_xx),
                        np.sum(block_of_derivative_xy),
                        np.sum(block_of_derivative_yy),
                    ]
                    # add it to the output
                    determinant = (
                        second_moment_matrix_elements[0]
                        * second_moment_matrix_elements[2]
                    ) - (second_moment_matrix_elements[1] ** 2)
                    trace = (
                        second_moment_matrix_elements[0]
                        + second_moment_matrix_elements[2]
                    )
                    corner_response_element = determinant - (
                        self.CORNER_RESPONSE_CONSTANT * (trace**2)
                    )
                    response_row.append(corner_response_element)
                    # move on to the next starting column, using the stride
                    starting_col_ndx += stride
                # now, add the new row to the list
                corner_response.append(response_row)
                # move to the next starting row for the corner response calculation
                starting_row_ndx += stride
            return np.array(corner_response)

        ### DRIVER
        if derivative_operator_x is None:
            derivative_operator_x = Filter2D.HORIZONTAL_SOBEL_FILTER.value
        if derivative_operator_y is None:
            derivative_operator_y = Filter2D.VERTICAL_SOBEL_FILTER
        if gaussian_window is None:
            gaussian_window = BaseGaussianFilter().create_gaussian_filter()
        (
            convolved_hessian_xx,
            convolved_hessian_yy,
            convolved_hessian_xy,
        ) = _compute_derivatives_in_gaussian_window(
            image, gaussian_window=gaussian_window
        )
        corner_response = _compute_corner_response(
            gaussian_window,
            convolved_hessian_xx,
            convolved_hessian_yy,
            convolved_hessian_xy,
        )

        if use_non_max_suppression is True:
            corner_response = ops.non_max_suppression_2D(corner_response)

        return corner_response

    def pick_top_features(
        self,
        corner_response: np.ndarray,
        top_many_features: int = TOP_MANY_FEATURES_TO_DETECT,
    ):
        """
        Choose the largest k points computed as a response to the Harris corner detector.

        Parameters:
            corner_response(nd.array): a nxm matrix
            top_many_features(int): this is k

        Returns: np.ndarray: a kx3 matrix representing the top Harris corners as:
            - their y-coordinate
            - their x-coordinate
            - the value of their Harris corner response
        """
        height, width = corner_response.shape
        coordinate_value_pairs = np.zeros((height * width, 3))
        pixel_indices_flattened = np.arange(height * width)

        # first, populate all the YX coordinate locations
        coordinate_value_pairs[:, :2] = ops.convert_1d_indices_to_2d(
            corner_response,
            pixel_indices_flattened
        )

        # add the corner response at each locations
        for val_index in range(coordinate_value_pairs.shape[0]):
            row_index, col_index = coordinate_value_pairs[val_index, :2].astype(int)
            coordinate_value_pairs[val_index, 2] = corner_response[row_index, col_index]

        # filter out the locations with the largest corner responses
        return np.array(
            heapq.nlargest(
                top_many_features,
                coordinate_value_pairs,
                key=lambda coords_followed_by_val: coords_followed_by_val[2],
            )
        )

    @classmethod
    def execute_and_visualize(
        cls: "HarrisCornerDetector",
        image: np.ndarray,
        image_name: str,
        top_many_features: int = TOP_MANY_FEATURES_TO_DETECT,
        use_non_max_suppression: bool = False,
        derivative_operator_x: list[list[int]] = None,
        derivative_operator_y: list[list[int]] = None,
        gaussian_window: np.ndarray = None,
    ) -> None:
        # detect_features
        detector = cls()
        corner_response = detector.detect_features(
            image,
            use_non_max_suppression,
            derivative_operator_x=derivative_operator_x,
            derivative_operator_y=derivative_operator_y,
            gaussian_window=gaussian_window,
        )
        # pick top features
        top_k_points = detector.pick_top_features(corner_response, top_many_features)
        # plotting
        plt.imshow(image, cmap="gray")
        plt.scatter(y=top_k_points[:, 0], x=top_k_points[:, 1], color="red")
        plt.title(f'Corner Points Detected for Image: "{image_name}"')
        plt.show()

        return super().execute_and_visualize()

    @staticmethod
    def compute_feature_correspondences(
        descriptors: List[Tuple[int, int, np.ndarray]],
        desired_num_similarities: int = TOP_MANY_SIMILARITIES_TO_SELECT,
        similarity_metric: SimilarityMeasure = SimilarityMeasure.COS,
    ) -> Tuple[int, int, float]:
        """
        Computes a 2D list of feature correspondences.

        The `descriptors` parameter is assumed to only contain 2 sets of descriptors:
        one for each of the two images we have in HW 3 (CS 558).

        Each nested list in the out follows the format of: (
            int: index of a feature from image 1,
            int: index of a feature from image 2,
            float: the similarity between the two, if known
        )
        """
        # compute the similarities between every possible pair of descriptors, and then grab the highest ones
        similarities = list()
        # prevent duplicate pairs of points from being plotted
        descriptors1, descriptors2 = descriptors
        existing_correspondence_pairs = set()
        rng = np.random.default_rng()
        extract_similarity = lambda indicies_and_similarity: indicies_and_similarity[4]
        for index1 in range(len(descriptors1)):
            left_y, left_x, descriptor1 = descriptors1[index1]

            similarities_for_one_point_in_one_image = list()
            for index2 in range(len(descriptors2)):
                right_y, right_x, descriptor2 = descriptors2[index2]

                next_entry = ()
                if similarity_metric is not SimilarityMeasure.NULL:
                    next_entry = (
                        left_y,
                        left_x,
                        right_y,
                        right_x,
                        ops.compute_similarity(
                            similarity_metric, descriptor1, descriptor2
                        ),
                    )
                else:
                    next_entry = (left_y, left_x, right_y, right_x, np.random.random())
                similarities_for_one_point_in_one_image.append(next_entry)

            # choose the strongest correspondence in the 2nd image, to this single point
            best_correspondences_for_one_point = sorted(
                similarities_for_one_point_in_one_image,
                key=extract_similarity,
                reverse=True,
            )
            best_correspondence = None
            for y1, x1, y2, x2, similarity_val in best_correspondences_for_one_point:
                # break early once the best similarity found
                if (y1, x1, y2, x2) not in existing_correspondence_pairs:
                    best_correspondence = [y1, x1, y2, x2, similarity_val]
                    existing_correspondence_pairs.add((y1, x1, y2, x2))
                    break
            similarities.append(best_correspondence)

        # choose the strongest correspondences overall, across all the points
        top_similarities = []
        if similarity_metric == SimilarityMeasure.SSD:
            # the lower the measure, the better the correspondence
            top_similarities = np.array(
                heapq.nsmallest(
                    desired_num_similarities,
                    similarities,
                    key=extract_similarity,
                )
            )
        elif (
            similarity_metric == SimilarityMeasure.COS
            or similarity_metric == SimilarityMeasure.NCC
        ):
            # for these types of measures, bigger is better
            top_similarities = np.array(
                heapq.nlargest(
                    desired_num_similarities,
                    similarities,
                    key=extract_similarity,
                )
            )
        elif similarity_metric == SimilarityMeasure.NULL:
            top_similarities = rng.choice(
                similarities,
                size=desired_num_similarities,
                replace=False,
                axis=0,
            )

        assert top_similarities.shape == (
            desired_num_similarities,
            5,
        ), f"Expected to pick up ({desired_num_similarities}, 5) similarities, actually have: {top_similarities.shape}"
        return top_similarities

    @staticmethod
    def compute_feature_descriptors(
        image: np.ndarray,
        corner_points: np.ndarray,
        window_side_length: int,
    ) -> List[Tuple[int, int, np.ndarray]]:
        """
        Return is a 2D array.
        Each row of said array represents (
            y_coordinate of corner point,
            x_coordinate of corner point,
            feature_descriptor
        )

        For the sake of simplicity - our feature descriptor is
        just a 1D array of the patch, normalized to a
        Standard Normal Gaussian.
        """
        # get pixel patches
        mock_filter = np.eye(window_side_length, window_side_length)
        padded_img, _, _ = ops.pad(image, mock_filter, 1, "zero")
        descriptors = []

        for corner in corner_points:
            y, x, _ = corner.ravel()
            y = int(y)
            x = int(x)

            # Ensure the patch is within the image boundaries
            patch = padded_img[
                y - 1 : (y - 1) + window_side_length,
                x - 1 : (x - 1) + window_side_length,
            ]

            # Flatten the patch values to create the descriptor
            descriptor = patch.flatten()

            # Normalize to have zero mean and unit variance
            descriptor = (
                (descriptor - np.mean(descriptor)) / np.std(descriptor)
                if np.std(descriptor) != 0
                else descriptor
            )
            descriptors.append((y, x, descriptor))

        return descriptors

    @staticmethod
    def visualize_correspondences(
        left_img: np.ndarray,
        right_img: np.ndarray,
        correspondences: np.ndarray,
        plot_title: str = "",
    ):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image
        ax[0].imshow(left_img, cmap="gray")
        ax[0].set_title("Left Image")

        # Plot the second image
        ax[1].imshow(right_img, cmap="gray")
        ax[1].set_title("Right Image")

        # Loop through the size list and draw lines between connected points
        for left_img_y, left_img_x, right_img_y, right_img_x, _ in correspondences:
            # Plot red dots on the images
            ax[0].plot(left_img_x, left_img_y, "ro")
            ax[1].plot(right_img_x, right_img_y, "ro")

            # Draw a line connecting the points
            connector = ConnectionPatch(
                xyA=(left_img_x, left_img_y),
                coordsA=ax[0].transData,
                xyB=(right_img_x, right_img_y),
                coordsB=ax[1].transData,
                color="green",
            )

            fig.add_artist(connector)

        # Set axis limits to include the entire images
        ax[0].set_xlim([0, left_img.shape[1]])
        ax[0].set_ylim([0, left_img.shape[0]])
        ax[1].set_xlim([0, left_img.shape[1]])
        ax[1].set_ylim([0, right_img.shape[0]])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        plt.title(plot_title)
        plt.show()

    @classmethod
    def find_and_visualize_correspondences(
        cls: "HarrisCornerDetector",
        left_img: np.ndarray,
        right_img: np.ndarray,
        plot_title: str,
        top_many_features: int = TOP_MANY_FEATURES_TO_DETECT,
        desired_num_similarities: int = TOP_MANY_SIMILARITIES_TO_SELECT,
        use_non_max_suppression: bool = False,
        similarity_metric: SimilarityMeasure = SimilarityMeasure.COS,
        window_side_length=3,  # for the patch we want to define around each corner point
    ):
        ### DRIVER
        # detect_features
        detector = cls()
        corner_response1 = detector.detect_features(left_img, use_non_max_suppression)
        corner_response2 = detector.detect_features(right_img, use_non_max_suppression)
        # pick top features
        top_k_points1 = detector.pick_top_features(corner_response1, top_many_features)
        top_k_points2 = detector.pick_top_features(corner_response2, top_many_features)
        # compute feature descriptors, for top points
        descriptors1 = cls.compute_feature_descriptors(
            left_img, top_k_points1, window_side_length
        )
        descriptors2 = cls.compute_feature_descriptors(
            right_img, top_k_points2, window_side_length
        )

        # Compute & plot the feature correspondences
        top_similarities = cls.compute_feature_correspondences(
            descriptors=[descriptors1, descriptors2],
            desired_num_similarities=desired_num_similarities,
            similarity_metric=similarity_metric,
        )

        cls.visualize_correspondences(
            left_img, right_img, top_similarities, plot_title=plot_title
        )

        return super().execute_and_visualize()


if __name__ == "__main__":
    HarrisCornerDetector.execute_and_visualize()
