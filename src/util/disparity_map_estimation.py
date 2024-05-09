from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Union

import matplotlib.pyplot as plt
import numpy as np

from util import ops
from util.ops import SimilarityMeasure
from util.rank_transform import RankTransform2D


@dataclass
class SimpleStereoDisparityMap:
    """
    Computes a disparity map between two images based on the
    assumptions that they are:
        1) of the same shape,

        2) grayscale,

        3) and already rectified.
    """

    left_image: np.ndarray
    right_image: np.ndarray
    stride: int = 1
    padding_type: Union[Literal["zero"], Literal["repeat"]] = "zero"

    def compute(
        self: SimpleStereoDisparityMap,
        similarity_measure: SimilarityMeasure,
        rank_transform_filter_side_length: int,
        window_size: int,
        do_logging: bool = False,
        max_disparity_level: int | float = float("inf"),
    ) -> np.ndarray:
        """
        Compute pixel-wise disparity by searching in the second image
        the most similar patch along the same row in the first image.

        Parameters:
            similarity_measure(SimilarityMeasure): determines how the hypotheses for the correspondence are determined.
                Currently, only Sum of Absolute Difference (aka `SimilarityMeasure.SAD`) is supported.
            rank_transform_filter_side_length(int): the side length of the kernel used for rank filtering.
            window_size(int): the side length of the window you want to use to compare local patches in the left/right views.
            do_logging(bool): enables certain debugging statements in the code to be reached during execution. Defaults to False.
            max_disparity_level(int | float): if known a priori, we use this as an upper threshold for disparity levels.
                It means that we think that if a disparity > max_disparity_level, there must not actually be any correspondence in
                the right image, for some pixel in the left.

        Returns: np.ndarray: 2D disparity map
        """

        ### HELPER(S)
        def _compute_patch_sum(
            starting_col: np.ndarray,
            starting_row_ndx: int,
            image: np.ndarray,
            kernel_dims: tuple[int, int],  # h, w
        ) -> int:
            """
            Returns the sum of cell values within a small window of an image.

            Parameters:
                starting_col(np.ndarray): shape of (1,), has an integer dtype
                starting_row_ndx(int): index for the top row of the window
                image(nd.ndarray): a grayscale image
                kernel_dims(int, int): height and width of the window

            Returns: int
            """
            kernel_h, kernel_w = kernel_dims
            starting_col_ndx = starting_col[0]
            return np.sum(
                image[
                    starting_row_ndx : int(starting_row_ndx + kernel_h),
                    starting_col_ndx : int(starting_col_ndx + kernel_w),
                ]
            )

        def _compute_row_of_disparity_map_sad(
            compute_patch_sum_helper: Callable,
            ptl: np.ndarray,
            ptr: np.ndarray,
            starting_row_ndx: int,
        ) -> np.ndarray:
            """
            Computes a single row of disparity levels based on SAD.

            Parameters:
                compute_patch_sum_helper(function): same as the _compute_patch_sum, but partially
                                                    filled in so it can be used as a 1D function.
                ptl(2D np.ndarray): the left image, after having been rank transformed
                ptr(2D np.ndarray): the right image, after having been rank transformed
                starting_row_ndx(int): index of the top row to use in calculating patch sums

            Returns: np.ndarray: a 1D array of int's
            """
            # get all patches in the left image row
            _, kernel_w = kernel.shape
            left_image_starting_col_indices = (
                np.arange(0, ptl.shape[1] - kernel_w + self.stride, self.stride)
                .astype(int)
                .reshape(1, -1)
            )
            left_image_center_col_indices = (
                (left_image_starting_col_indices + (kernel_w // 2))
                .astype(int)
                .squeeze()
            )

            left_image_patch_sums = np.apply_along_axis(
                func1d=lambda left_image_starting_col: compute_patch_sum_helper(
                    starting_col=left_image_starting_col,
                    image=padded_transformed_left,
                    starting_row_ndx=starting_row_ndx,
                ),
                axis=0,
                arr=left_image_starting_col_indices,
            )
            assert (
                left_image_center_col_indices.shape[0] == left_image_patch_sums.shape[0]
            )

            # get all patches in the right image row
            right_image_starting_col_indices = (
                np.arange(0, ptr.shape[1] - kernel_w + self.stride, self.stride)
                .astype(int)
                .reshape(1, -1)
            )
            right_image_center_col_indices = (
                (right_image_starting_col_indices + (kernel_w // 2))
                .astype(int)
                .squeeze()
            )

            right_image_patch_sums = np.apply_along_axis(
                func1d=lambda right_image_starting_col: compute_patch_sum_helper(
                    starting_col=right_image_starting_col,
                    image=padded_transformed_right,
                    starting_row_ndx=starting_row_ndx,
                ),
                axis=0,
                arr=right_image_starting_col_indices,
            )
            assert (
                right_image_center_col_indices.shape[0]
                == right_image_patch_sums.shape[0]
            )

            # match correspondences by sorting
            correspondences = np.zeros((right_image_center_col_indices.shape[0], 2))
            correspondences[:, 0] = left_image_center_col_indices[
                np.argsort(left_image_patch_sums)
            ]
            correspondences[:, 1] = right_image_center_col_indices[
                np.argsort(right_image_patch_sums)
            ]

            disparity = np.abs(correspondences[:, 0] - correspondences[:, 1])
            disparity = np.where(disparity > max_disparity_level, 0, disparity)
            if do_logging:
                print(f"left sums: {left_image_patch_sums}")
                print(f"left sums: {right_image_patch_sums}")
                print(f"hypothesized correspondences: {correspondences}")
            assert np.min(disparity) >= 0, f"disparity of {disparity} is too low"
            assert (
                np.max(disparity) <= max_disparity_level
            ), f"disparity of {disparity} is too high"
            return disparity

        ### DRIVER
        # data validations
        assert self.left_image.shape == self.right_image.shape, (
            f"Image shape mismatch between left <{self.left_image.shape}>"
            f" and right <{self.right_image.shape}> views."
        )
        assert len(self.left_image.shape) == 2, (
            f"Expected a 2D array representing a grayscale image, "
            f"actual number of channels is: {self.left_image.shape[2]}."
        )
        # rank transform both images
        transformed_left = RankTransform2D.transform(
            self.left_image, rank_transform_filter_side_length
        )
        transformed_right = RankTransform2D.transform(
            self.right_image, rank_transform_filter_side_length
        )
        # create a square kernel of all 1's using the given window size
        kernel = np.ones((window_size, window_size))
        # pad both images
        padded_transformed_left, _, _ = ops.pad(
            transformed_left, kernel, stride=self.stride, padding_type=self.padding_type
        )
        ptl: np.ndarray = padded_transformed_left
        padded_transformed_right, _, _ = ops.pad(
            transformed_right,
            kernel,
            stride=self.stride,
            padding_type=self.padding_type,
        )
        ptr: np.ndarray = padded_transformed_right
        assert (
            ptl.shape == ptr.shape
        ), f"Shape mismatch after padding: {ptl.shape} != {ptr.shape}"
        # create the output image
        output = np.zeros_like(self.left_image)

        # TODO[Zain] - debug this faster implementation in the future!
        # compute_patch_sum_helper = functools.partial(
        #     _compute_patch_sum,
        #     kernel_dims=kernel.shape,
        # )

        # # for every row in the first image
        # kernel_h, _ = kernel.shape
        # for starting_row_ndx in np.arange(
        #     0, ptl.shape[0] - kernel_h + self.stride, self.stride
        # ).astype(int):
        #     # compute the next row of the output disparity map, using the given similarity metric
        #     disparity = None
        #     if similarity_measure == SimilarityMeasure.SAD:
        #         disparity = _compute_row_of_disparity_map_sad(
        #             compute_patch_sum_helper, ptl, ptr, starting_row_ndx,
        #         )
        #     else:  # similarity_measure is something other than SAD
        #         raise NotImplementedError(
        #             f"Sorry, using similarity_measure = {similarity_measure} is not supported."
        #         )
        #     # fill in the output
        #     output[starting_row_ndx, :disparity.shape[0]] = disparity

        # "Brute force"
        # for each row
        kernel_h, kernel_w = kernel.shape
        output_row_ndx = 0
        for starting_row_ndx in np.arange(
            0, ptl.shape[0] - kernel_h + self.stride, self.stride
        ):
            # for each pixel
            disparity_row = list()
            for left_starting_col_ndx in np.arange(
                0, ptl.shape[1] - kernel_w + self.stride, self.stride
            ):
                left_center_col_ndx = left_starting_col_ndx + (kernel_w // 2)
                # init lowest_sum
                lowest_sum, best_center_index = float("inf"), left_center_col_ndx
                left_patch = ptl[
                    starting_row_ndx : starting_row_ndx + kernel_h,
                    left_starting_col_ndx : left_starting_col_ndx + kernel_w,
                ]
                disparity_search_range = min(max_disparity_level, left_center_col_ndx)
                disparity_search_stop_index = (
                    max(0, left_center_col_ndx - disparity_search_range) - 1
                )
                # for every column (from the current_col to (current_col - disp_level=current_col))
                for right_center_col_ndx in range(
                    left_center_col_ndx, disparity_search_stop_index, -1 * self.stride
                ):

                    # in the right img, same row --> compute every SAD
                    right_starting_col_ndx = right_center_col_ndx - (kernel_w // 2)
                    if right_starting_col_ndx > -1:
                        right_patch = ptr[
                            starting_row_ndx : starting_row_ndx + kernel_h,
                            right_starting_col_ndx : right_starting_col_ndx + kernel_w,
                        ]
                        sad_val = np.sum(np.abs(left_patch - right_patch))

                        # take the min to get to the disparity
                        if sad_val < lowest_sum:
                            lowest_sum = sad_val
                            best_center_index = right_center_col_ndx

                disparity_row.append(left_center_col_ndx - best_center_index)

            output[output_row_ndx, :] = np.array(disparity_row)
            output_row_ndx += 1

        # final checks
        assert (
            output.shape == self.left_image.shape
        ), f"Shape mismatch {output.shape} != {self.left_image.shape}"
        return output

    @classmethod
    def compute_and_visualize(
        cls: SimpleStereoDisparityMap,
        image1: np.ndarray,
        image2: np.ndarray,
        similarity_measure: SimilarityMeasure,
        rank_transform_filter_side_length: int,
        window_size: int,
        scene_name: str,
        do_logging: bool = False,
        stride: int = 1,
        max_disparity_level: int | float = float("inf"),
        padding_type: Union[Literal["zero"], Literal["repeat"]] = "zero",
    ) -> None:
        """
        Convenience wrapper that both computes and plots
        1 disparity map for a pair of stereo images.

        Parameters:
            image1, image2 (np.ndarrays): the left and right views, respectively
            similarity_measure(SimilarityMeasure): determines how the hypotheses for the correspondence are determined.
                Currently, only Sum of Absolute Difference (aka `SimilarityMeasure.SAD`) is supported.
            rank_transform_filter_side_length(int): the side length of the kernel used for rank filtering.
            window_size(int): the side length of the window you want to use to compare local patches in the left/right views.
            scene_name(str): used for titling the image plot at the end, for the disparity map
            do_logging(bool): enables certain debugging statements in the code to be reached during execution. Defaults to False.
            stride(int): hyperparameter that decides how many "steps" we take when sliding a local window over the images
                (both for the rank filtering and stereo algorithm).
            max_disparity_level(int | float): if known a priori, we use this as an upper threshold for disparity levels.
                It means that we think that if a disparity > max_disparity_level, there must not actually be any correspondence in
                the right image, for some pixel in the left.
            padding_type("zero" | "repeat"): type of padding to use while computing the disparity map

        Returns: None
        """
        mapper: SimpleStereoDisparityMap = cls(
            left_image=image1,
            right_image=image2,
            stride=stride,
            padding_type=padding_type,
        )
        disparity_map = mapper.compute(
            similarity_measure=similarity_measure,
            rank_transform_filter_side_length=rank_transform_filter_side_length,
            window_size=window_size,
            do_logging=do_logging,
            max_disparity_level=max_disparity_level,
        )
        plt.imshow(disparity_map, cmap="gray")
        plt.title(f'Disparity for "{scene_name}", {window_size}x{window_size} Window')
        plt.axis("off")
