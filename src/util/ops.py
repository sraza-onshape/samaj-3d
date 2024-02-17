from enum import Enum
from typing import (
    Callable,
    List,
    Literal,
    Tuple,
    Union,
)

import numpy as np
from PIL import Image


class Filter2D(Enum):
    HORIZONTAL_SOBEL_FILTER = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    IDENTITY_FILTER = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    VERTICAL_SOBEL_FILTER = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


class SimilarityMeasure(Enum):
    SSD = "sum_squared_difference"
    NCC = "normalized_cross_correlation"  # aka, the Pearson Correlation Coef
    COS = "cosine_similarity"
    NULL = "randomness"  #  when selected, this means we don't actually care about similarity


def convert_1d_indices_to_2d(
    matrix: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    When given a list of integers representing
    the zero-indexed indices of a 'flattened' matrix, reshape them
    into a list of tuples that gives the corresponding indices
    into the original matrix.

    Example:
    >>> a = np.array([2, 2], [3, 3])
    >>> b = np.array([0, 1, 2, 3])
    >>> convert_1d_indices_to_2d(a, b)
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

    Parameters:
        matrix(2d np.ndarray): the original matrix of shape (n, m)
        indices(1d np.ndarray): the list of ints of shape (n * m)

    Returns: np.ndarray: has shape of (n*m, 2)
    """
    return (indices // matrix.shape[1], indices % matrix.shape[1])


def compute_similarity(
    mode: Literal[
        SimilarityMeasure.NCC,
        SimilarityMeasure.SSD,
        SimilarityMeasure.COS,
    ],
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> float:
    """
    Convenience wrapper that routes you to
    whatever similiarity measure you want to compute.

    Parameters:
        mode(SimilarityMeasure): specify if you want normalized cross-correlation, SSD, or cosine similarity
        arr1, arr2: two array-likes of the same shape
    
    Returns: array-like
    """

    ### HELPERS
    def _compute_ssd(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Output array has a shape of (1,)."""
        return np.sum((arr1 - arr2) ** 2)

    def _compute_ncc(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Output array has a shape of (1,)."""
        deviations1 = arr1 - arr1.mean()
        deviations2 = arr2 - arr2.mean()

        numerator = np.sum(deviations1 * deviations2)
        denominator = np.sqrt(np.sum(deviations1**2) * np.sum(deviations2**2))

        return numerator / denominator

    def _compute_cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Output array has a shape of (1,)."""
        return (arr1 @ arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

    ### DRIVER
    measure_funcs = {
        SimilarityMeasure.SSD: _compute_ssd,
        SimilarityMeasure.NCC: _compute_ncc,
        SimilarityMeasure.COS: _compute_cosine_similarity,
    }
    return measure_funcs[mode](arr1, arr2)


def load_image(
    filename: str,
    rotation_angle: int = 0,
    return_grayscale: bool = True,
    return_array: bool = False,
    target_size: Tuple[int, int] = None,
    verbosity: bool = True,
) -> Union[List[List[int]], np.ndarray]:
    """
    Allows us to convert images from its binary form
    to a 2D array-like representing the image.

    Parameters:
        filename(str): relative path to the image file
        rotation_angle(int): in degrees
        return_array(bool): if True, the output returned is an ndarray
        return_grayscale(bool): if True, the output has only 1 channel. It will be a 2D NumPy array.
        target_size(2-tuple): (width, height) that you want the output to have.

    Returns: array-like, pixel raster matrix
    """
    with Image.open(filename) as img:
        # Convert the image to grayscale, and do any rotations as needed
        if return_grayscale:
            img = img.convert("L")

        img = img.rotate(rotation_angle, expand=1)

        if target_size is not None:
            img = img.resize(size=target_size)

        # Get image data as a list of lists (2D list)
        image_data = list(img.getdata())  # currently, this is 1D
        width, height = img.size
        image_data = [image_data[i * width : (i + 1) * width] for i in range(height)]

        if return_array is True:
            image_data = np.array(image_data)

        if verbosity is True:
            if len(image_data.shape) == 2:
                print(f"Dimensions of {filename}: {height} x {width}")
            elif len(image_data.shape) == 3:
                print(
                    f"Dimensions of {filename}: {height} x {width} x {image_data.shape[2]}"
                )

    return image_data


def convolve_matrices(matrix1: List[List[float]], matrix2: List[List[float]]) -> float:
    """asumes both matrices have the same, non-zero dimensions"""
    width, height = len(matrix1[0]), len(matrix1)

    product = 0

    for row_i in range(height):
        for col_i in range(width):
            product += matrix1[row_i][col_i] * matrix2[row_i][col_i]

    return product


def apply_kernel_dot_product(
    channel: List[List[float]],
    kernel: List[List[float]],
    row_index: int,
    col_index: int,
) -> float:
    """Applies the 2D kernel to 1 block of pixels on the image.

    Args:
        channel: 2D array - one of the channels of the input image
        kernel: 2D array representing the parameters to use
        row_index, col_index: int: the coordinates of the upper left corner
                            of the block of pixels being convolved

    Returns: float: the dot product of the kernel and the image pixels
    """
    # A: define useful vars
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    # B: get the block of pixels needed for the convolution
    block_of_pixels = [
        row[col_index : (kernel_w + col_index)]
        for row in channel[row_index : (kernel_h + row_index)]
    ]
    # C: compute the convolution
    return convolve_matrices(block_of_pixels, kernel)


def slide_kernel_over_image(
    channel: List[List[float]],
    kernel: List[List[float]],
    row_index: int,
    stride: int,
    apply: Callable = apply_kernel_dot_product,
) -> List[float]:
    """Applies the 2D kernel across the columns of 1 image channel.

    Args:
        channel: 2D array - one of the channels of the input image
        kernel: 2D array representing the parameters to use
        row_index, col_index: int: the coordinates of the upper left corner
                            of the block of pixels being convolved
        apply: function - the operation computed at each window location

    Returns: np.array: 1D array of the resulting values from performing
                        the convolution at each "block" of pixels on the channel
    """
    # A: define useful vars + output
    _, kernel_w = len(kernel), len(kernel[0])
    conv_channel_row = list()
    # B: get the starting column
    starting_col_ndx = 0
    while starting_col_ndx <= len(channel[0]) - kernel_w:
        # compute the convolution
        conv_block_of_pixels = apply(channel, kernel, row_index, starting_col_ndx)
        # add it to the output
        conv_channel_row.append(conv_block_of_pixels)
        # move on to the next starting column, using the stride
        starting_col_ndx += stride
    return conv_channel_row


def convolve_2D(
    channel: List[List[float]], kernel: List[List[float]], stride: int
) -> List[List[float]]:
    """Performs a 2D convolution over 1 channel.

    Args:
        channel: 2D array - one of the channels of the input image
        filter: 2D array representing the parameters to use
        stride: int - using the same stride length for both directions

    Returns: np.array: the convolved channel
    """
    conv_channel = list()
    kernel_h, _ = len(kernel), len(kernel[0])
    # iterate over the rows and columns
    starting_row_ndx = 0
    while starting_row_ndx <= len(channel) - kernel_h:
        # convolve the next row of this channel
        conv_channel_row = slide_kernel_over_image(
            channel, kernel, starting_row_ndx, stride
        )
        # now, add the convolved row to the list
        conv_channel.append(conv_channel_row)
        # move to the next starting row for the convolutions
        starting_row_ndx += stride
    return conv_channel


def pad(
    image: List[List[float]],
    img_filter: List[
        List[int]
    ],  # TODO[make it so users can just specify dims of the filter)
    stride: int,
    padding_type: str,
) -> Tuple[np.array, int, int]:
    padded_image = list()

    # compute the # of pixels needed to pad the image (in x and y)
    padding_dist_x = (
        len(img_filter) - stride + (len(image) * (stride - 1))
    )  # TODO[turn into helper func]
    padding_dist_y = (
        len(img_filter[0]) - stride + (len(image[0]) * (stride - 1))
    )  # TODO[extract into helper func]

    # zero-padding
    if padding_type == "zero":
        # add the rows (at the beginning) that are all 0
        for _ in range(padding_dist_y // 2):
            new_row = [0 for _ in range(padding_dist_x + len(image[0]))]
            padded_image.append(new_row)
        # add the original image (extend its rows with zeros)
        for row in image:
            zeros = [0 for _ in range(padding_dist_x // 2)]
            padded_row = np.concatenate([zeros, row, zeros])
            padded_image.append(padded_row)
        # add the rows (at the end) that are all 0  - TODO[Zain]: remove duplicated code later
        for _ in range(padding_dist_y // 2):
            new_row = [0 for _ in range(padding_dist_x + len(image[0]))]
            padded_image.append(new_row)

    # replicate boundary pixels
    elif padding_type == "repeat":
        padded_image = np.zeros(
            (len(image) + padding_dist_y, len(image[0]) + padding_dist_x)
        )
        side_padding_y, side_padding_x = padding_dist_y // 2, padding_dist_x // 2
        # fill corners
        padded_image[0:side_padding_y][0:side_padding_x] = image[0][0]  # top-left
        padded_image[0:side_padding_y][side_padding_x + len(image[0]) :] = image[0][
            -1
        ]  # top-right
        padded_image[side_padding_y + len(image) :][0:side_padding_x] = image[-1][
            0
        ]  # bottom-left
        padded_image[side_padding_y + len(image) :][
            side_padding_x + len(image[0]) :
        ] = image[-1][
            -1
        ]  # bottom-right
        # fill in the pixels above the top rows
        for row_index in range(0, side_padding_y):
            padded_image[row_index][
                side_padding_x : side_padding_x + len(image[0])
            ] = image[0][:]
        # fills the pixels below the last rows
        for row_index in range(side_padding_y + len(image), padded_image.shape[0]):
            padded_image[row_index][
                side_padding_x : side_padding_x + len(image[0])
            ] = image[-1][:]
        # fills the pixels to the left of the first col
        for row_index in range(len(image)):
            padded_image[side_padding_y : side_padding_y + len(image)][row_index][
                0:side_padding_x
            ] = image[row_index][0]
        # fills the pixels to the right of the last col
        for row_index in range(len(image)):
            padded_image[side_padding_y : side_padding_y + len(image)][row_index][
                side_padding_x + len(image[0]) :
            ] = image[row_index][-1]
        # fill in the center - "easiest part"
        for row_index in range(len(image)):
            padded_image[side_padding_y : side_padding_y + len(image)][row_index][
                side_padding_x : side_padding_x + len(image[0])
            ] = image[row_index][:]

    return np.array(padded_image), padding_dist_x, padding_dist_y


def convolution(
    image: List[List[float]], filter: List[List[float]], stride=1, padding_type="repeat"
) -> List[List[float]]:
    """Performs a convolution on an input image.

    Padding is used to ensure the output had the same dims as the input.

    Assumptions:
        1. filter is square and the size is an odd number.
        2. the filter is smaller than the image size

    Args:
        image: 2D array - a grayscale raster image, aka a "pixel matrix"
        filter: 2D array representing the parameters to use
        stride: int - using the same stride length for both directions
        padding_type: str - one of either 'zero' or 'repeat'

    Returns: np.array: a new RGB image
    """
    ### DRIVER
    image, _, _ = pad(image, filter, stride, padding_type)
    convolved_channel = convolve_2D(image, filter, stride)
    return convolved_channel


def non_max_suppression_2D(matrix: np.array) -> np.array:
    """After the determinant has been thresholded, use non-max suppression to recover more distinguishable keypoints."""
    # prevent potential loss of keypoints via padding
    padded_matrix, num_added_rows, num_added_cols = pad(
        matrix,
        img_filter=Filter2D.IDENTITY_FILTER.value,
        stride=1,
        padding_type="zero",
    )
    # traverse the matrix, to begin non-max suppression
    for center_val_row in range(
        num_added_rows // 2, padded_matrix.shape[0] - (num_added_rows // 2)
    ):
        for center_val_col in range(
            num_added_cols // 2, padded_matrix.shape[1] - (num_added_cols // 2)
        ):
            # determine if the given value should be suppressed, or its neighbors
            center_val = padded_matrix[center_val_row][center_val_col]
            neighbors = padded_matrix[
                center_val_row - 1 : center_val_row + 2,
                center_val_col - 1 : center_val_col + 2,
            ]
            neighbors[1][
                1
            ] = 0  # hack to prevent the center value from "self-suppressing" (I have no idea, I made that term up)
            # zero out the appropiate value(s)
            if center_val > neighbors.max():  # suppression of neighbors
                padded_matrix[
                    center_val_row - 1 : center_val_row + 2,
                    center_val_col - 1 : center_val_col + 2,
                ] = 0
                padded_matrix[center_val_row][center_val_col] = center_val
            else:  # suppression of the center
                padded_matrix[center_val_row][center_val_col] = 0

    # return the modified matrix
    return padded_matrix[
        num_added_rows // 2 : matrix.shape[0] - (num_added_rows // 2),
        num_added_cols // 2 : matrix.shape[1] - (num_added_cols // 2),
    ]


class TLSFitter:
    """This class is a useful abstraction for using Total Least Sqaures to fit lines."""

    # TODO: utilize this Python gist for implementation: https://gist.github.com/galenseilis/29935da21d5c34a197bf1ec91dd30f9e
    pass


if __name__ == "__main__":
    # a few small test cases
    matrix1 = np.arange(9).reshape(3, 3) + 1
    matrix2 = np.arange(36).reshape(6, 6) + 1
    fake_filter = np.ones(1).reshape(1, 1)
    even_sized = np.arange(4).reshape(2, 2) + 1

    # print(convolution(matrix1.tolist(), fake_filter.tolist()))  # ✅ no padding used
    # print(convolution(matrix1.tolist(), matrix1.tolist()))  # ✅ padding used
    # print(convolution(even_sized.tolist(), fake_filter.tolist()))  # ✅ no padding used
    print(convolution(matrix2.tolist(), matrix1.tolist()))  # ✅ padding used
