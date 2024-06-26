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


def estimate_plane_parameters_from_points(
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray,
    ) -> np.ndarray:
    """Estimate a plane using three XYZ points assumed to be non-colinear."""
    # compute two vectors in the plane
    vector1 = point2 - point1  # Subtracting two points gives a vector
    vector2 = point3 - point1

    # compute the cross product
    normal_vector = np.cross(vector1, vector2)

    # normalize
    normal_vector /= np.linalg.norm(normal_vector)

    # compute the constant d using one of the points
    d = -1 * (normal_vector @ point1)

    # Return the coefficients (a, b, c, d)
    return np.array([
        normal_vector[0],
        normal_vector[1],
        normal_vector[2],
        d
    ])


def are_non_collinear(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    """Determine if three sampled points in 3D are non-collinear."""
    # Calculate vectors between the points
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate cross product of the vectors
    cross_product = np.cross(v1, v2)

    # Check if cross product is non-zero
    return not np.allclose(cross_product, 0)


def compute_disparity_from_depth(
    baseline: float, focal_length: float, depth: float
) -> float:
    """
    Utilize the relationship d = b*f / Z to arrive at disparity.
    """
    b, f, Z = baseline, focal_length, depth
    return (b * f) / Z


def compute_stereo_camera_baseline(
    R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray
) -> float:
    """
    Calculate the baseline in meters.

    Parameters:
        R1(np.ndarray): 3x3 rotation matrix for the 1st image
        t1(np.ndarray): 3x1 translation vector for the 2nd image
            (units assumed to be in meters)
        R2(np.ndarray): 3x3 rotation matrix for the 2nd image
        t2(np.ndarray): 3x1 translation vector for the 2nd image
            (units assumed to be in meters)

    Returns: float: the stereo camera basline
    """
    camera_center_1 = (-1 * np.linalg.inv(R1)) @ t1
    camera_center_2 = (-1 * np.linalg.inv(R2)) @ t2
    return np.linalg.norm(camera_center_1 - camera_center_2, ord=2)


def evaluate_disparity_map(
    predicted: np.ndarray,
    true: np.ndarray,
    scale_factor: int = 1,
    threshold: float = 0,
) -> None:
    """
    Prints the error rate in a disparity levels.

    Assumes both maps are of the same shape.

    Parameters:
        predicted(np.ndarray): the computed disparity map
        true(np.ndarray): the ground truth disparity map
        scale_factor(int): optional, the amount we divide the disparities in
            the ground truth by
        threshold(float): the amount of allowed difference between the
            the disparity maps

    Returns: None
    """
    ground_truth = true.copy() / scale_factor
    difference = np.abs(predicted - ground_truth)
    num_bad_pixels = np.where(difference > threshold, 1, 0).sum()
    num_total_pixels = np.prod(predicted.shape)
    print(
        f"Error rate: {np.round((num_bad_pixels / num_total_pixels), decimals=4) * 100}%."
    )


class Filter2D(Enum):
    """Two-dimensional filters commonly used in computer vision."""

    HORIZONTAL_SOBEL_FILTER = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    IDENTITY_FILTER = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    VERTICAL_SOBEL_FILTER = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


class SimilarityMeasure(Enum):
    """Similarity measures commonly used in computer vision."""

    SSD = "sum_squared_difference"
    NCC = "normalized_cross_correlation"  # aka, the Pearson Correlation Coef
    COS = "cosine_similarity"
    SAD = "sum_absolute_difference"
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
    return np.column_stack([indices // matrix.shape[1], indices % matrix.shape[1]])


def compute_similarity(
    mode: Literal[
        SimilarityMeasure.NCC,
        SimilarityMeasure.SSD,
        SimilarityMeasure.COS,
        SimilarityMeasure.SAD,
    ],
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> float:
    """
    Convenience wrapper that routes you to
    whatever similiarity measure you want to compute.

    Parameters:
        mode(SimilarityMeasure): specify if you want normalized cross-correlation, SSD, cosine similarity, or SAD
        arr1, arr2: two array-likes of the same shape

    Returns: float: the computed similarity value
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

    def _compute_sum_absolute_difference(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Output array has a shape of (1,)."""
        return np.sum(np.linalg.norm(arr1 - arr2, ord=1))

    ### DRIVER
    measure_funcs = {
        SimilarityMeasure.SSD: _compute_ssd,
        SimilarityMeasure.NCC: _compute_ncc,
        SimilarityMeasure.COS: _compute_cosine_similarity,
        SimilarityMeasure.SAD: _compute_sum_absolute_difference,
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
    """
    Convolution in 2D, assuming both matrices have the same, non-zero dimensions.
    """
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
    padding_type: Union[Literal["zero"], Literal["repeat"]],
) -> tuple[np.ndarray, int, int]:
    """
    Add additional pixels along the border of an image.

    This auto-computes the dimensions that a padded image would need to have,
    in order for the output of a filtering operation to have the same dimensions
    as a given input image.

    Parameters:
        image(array-like): 2D array representing an image
        img_filter(array-like): 2D array representing some linear
                                operator you want to eventually use to
                                perform some kind of processing on the image
        stride(int): what distance you would want there to be in between each
                     local neighborhood of the image used for a filtering operation
        padding_type("zero" or "repeat"): determines the value used in the added pixels

    Returns: tuple[array-like, int, int]: the padded image, as
                                          well as two ints reporting how
                                          much bigger the height and width of it are
                                          vs. the original image
    """
    assert isinstance(image, (list, tuple, np.ndarray))
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
            ] = 0  # hack to prevent the center value from "self-suppressing" (I have no idea if that's a real term, I made that term up)
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
