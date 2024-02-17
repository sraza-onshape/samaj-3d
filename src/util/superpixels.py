from collections import defaultdict
import random
from typing import Dict, List, Tuple

# import cv2
import functools
import matplotlib.pyplot as plt
import numpy as np

from .gaussian_derivative import GaussianDerivativeFilter
from . import ops


class SLIC:
    @classmethod
    def execute_and_visualize(
        cls: "SLIC",
        img: np.array,
        step_size: int = 50,
        max_iter: int = 3,
        xy_scaling_factor: int = 2,
        plot_title: str = "",
        logging: bool = True,
    ) -> None:
        """Custom implementation of SLIC, a popular superpixels algorithm used in image segmentation."""

        ### HELPERS
        def _common_elements(*lists: Tuple[List]) -> List:
            """given n lists, find the common elements found throughout all"""
            # Convert each list to a set and find the intersection using reduce
            common_elements_set = functools.reduce(
                set.intersection, map(set, lists), set()
            )
            # Return the common elements as a list
            return list(common_elements_set)

        def _scale_5d_coordinate(coords: np.ndarray) -> np.ndarray:
            coords = coords.reshape(-1, 5)
            return np.concatenate(
                [coords[:, :2] / xy_scaling_factor, coords[:, 2:]], axis=1
            )

        def _dist_between_coordinates(
            coords_a: np.ndarray, coords_b: np.ndarray
        ) -> np.ndarray:
            """Compute Euclidean distance between 2, broadcastable 2D NumPy arrays."""
            return np.sqrt(
                np.sum(
                    np.diff(coords_a - coords_b, axis=1) ** 2,
                    axis=1,
                )
            )

        def _divide_image_along_dimension(img: np.ndarray, axis: int) -> np.ndarray:
            dimension_length = img.shape[axis]
            pixel_block_boundaries = np.arange(0, dimension_length, S)
            if pixel_block_boundaries[-1] < dimension_length - 1:
                pixel_block_boundaries = np.concatenate(
                    [pixel_block_boundaries, [dimension_length - 1]]
                )
            return pixel_block_boundaries

        def _assign_centroid(
            pixel_5d: np.ndarray,
            centroids_assigned_pts: Dict,
            centroids: np.ndarray,
            threshold_distance: float,
        ):
            centroids_in_range_along_y_mask = (
                np.absolute(pixel_5d[0] - centroids[:, 0]) < threshold_distance
            )
            centroids_in_range_along_x_mask = (
                np.absolute(pixel_5d[1] - centroids[:, 1]) < threshold_distance
            )
            centroids_in_range_along_xy_mask = np.logical_and(
                centroids_in_range_along_x_mask, centroids_in_range_along_y_mask
            )

            centroid_assignment = -1
            if centroids_in_range_along_xy_mask.sum() == 0 and logging:
                print(
                    f"Warning: please double check indexing there is at least 1 centroid in the threshold. Sticking with -1..."
                )
            else:  # centroids_in_range_along_xy_mask.sum() > 0
                scaled_pixel = _scale_5d_coordinate(pixel_5d).reshape(1, 5)
                scaled_centroids = _scale_5d_coordinate(centroids).reshape(-1, 5)

                xy_dist = _dist_between_coordinates(
                    scaled_centroids[:, :2], scaled_pixel[:, :2]
                )
                rgb_dist = _dist_between_coordinates(
                    scaled_centroids[:, 2:], scaled_pixel[:, 2:]
                )
                distances = rgb_dist + (xy_scaling_factor * xy_dist)

                closest_centroids = np.argsort(distances).astype(int).squeeze()
                for centroid_index in closest_centroids:
                    if centroids_in_range_along_xy_mask[centroid_index] == True:
                        centroid_assignment = centroid_index
                        break

            centroids_assigned_pts[centroid_assignment].append(pixel_5d)

        def _find_smallest_grad_position(
            current_coordinates: np.ndarray, combined_grad_magnitude: np.ndarray
        ) -> np.ndarray:
            """
            For one x-y position,
            find out which neighboring pixel has lowest gradient.

            Returns a 5D coordinate in the form [y, x, r, g, b].
            """
            window = np.zeros((3, 3))
            window[:, :] = np.inf
            # try to fill as much of the window as possible, with true values
            current_y, current_x = current_coordinates.astype(int)

            if (
                combined_grad_magnitude.shape[0] - current_y >= 1
                and combined_grad_magnitude.shape[1] - current_x >= 1
            ):
                sub_image = combined_grad_magnitude[
                    current_y - 1 : current_y + 2,
                    current_x - 1 : current_x + 2,
                ]

                window[: sub_image.shape[0], : sub_image.shape[1]] = sub_image
            elif logging:
                # special case: I guess our image has only a single pixel in one of its dims?
                print(
                    f"I've encountered a weird image... please check its dimensions: {img.shape}"
                )

            smallest_magnitude_coords_in_window_space_1d = np.argsort(
                window, axis=None
            )[0]
            smallest_magnitude_coords_in_window_space_2d = ops.convert_1d_indices_to_2d(
                window, np.array([smallest_magnitude_coords_in_window_space_1d])
            )

            smallest_magnitude_coords_delta = np.array(
                [
                    -1 * (1 - smallest_magnitude_coords_in_window_space_2d[0]),
                    -1 * (1 - smallest_magnitude_coords_in_window_space_2d[1]),
                ]
            ).squeeze()
            # return the coords of the smallest grad magnitude in "channel space"
            smallest_magnitude_coords_in_channel_space_2d = (
                current_coordinates + smallest_magnitude_coords_delta
            ).astype(int)
            rgb = (
                img[
                    smallest_magnitude_coords_in_channel_space_2d[0],
                    smallest_magnitude_coords_in_channel_space_2d[1],
                    :,
                ]
                .reshape(1, 3)
                .squeeze()
            )
            return np.concatenate([smallest_magnitude_coords_in_channel_space_2d, rgb])

        ### DRIVER
        S = step_size  # aliasing for convenience

        # Divide the image in blocks
        pixel_block_boundaries_x = _divide_image_along_dimension(img, 1)
        pixel_block_boundaries_y = _divide_image_along_dimension(img, 0)
        # initialize a centroid at the center of each block.
        centroid_coordinates = np.zeros(
            (pixel_block_boundaries_x.shape[0] * pixel_block_boundaries_y.shape[0], 2)
        )

        centroid_coordinates_index = 0
        for index_x in range(pixel_block_boundaries_x.shape[0] - 1):
            for index_y in range(pixel_block_boundaries_y.shape[0] - 1):
                block_coords_x = np.array(
                    [
                        pixel_block_boundaries_x[index_x],
                        pixel_block_boundaries_x[index_x + 1],
                    ]
                )
                block_coords_y = np.array(
                    [
                        pixel_block_boundaries_y[index_y],
                        pixel_block_boundaries_y[index_y + 1],
                    ]
                )
                # for now, assume we only care about 50x50 blocks, and can throw away the others
                if (
                    block_coords_x[1] - block_coords_x[0] == S
                    and block_coords_y[1] - block_coords_y[0] == S
                ):
                    centroid_coordinates[centroid_coordinates_index] = np.array(
                        [
                            block_coords_y.mean(),
                            block_coords_x.mean(),
                        ]
                    ).astype(int)
                centroid_coordinates_index += 1

        # compute gradient magnitude
        grad_img = img.copy()
        derivator = GaussianDerivativeFilter()
        for channel_index in range(img.shape[2]):
            channel = img[:, :, channel_index]
            partial_derivative_x, partial_derivative_y = derivator._compute_derivatives(
                channel
            )
            magnitude_matrix = derivator._compute_magnitude(
                partial_derivative_x, partial_derivative_y
            )
            grad_img[:, :, channel_index] = magnitude_matrix
        combined_grad_magnitude = np.sqrt(np.sum(grad_img**2, axis=2))

        assert (
            centroid_coordinates.shape[1] == 2
        ), f"Before local shift, are we not in 2D space? ({centroid_coordinates.shape[1]})"
        # Local Shift: move centroids to the smallest magnitude position in 3x3 windows
        _find_smallest_grad_position_short = functools.partial(
            _find_smallest_grad_position,
            combined_grad_magnitude=combined_grad_magnitude,
        )

        shifted_centroid_centers = np.apply_along_axis(
            _find_smallest_grad_position_short,
            axis=1,
            arr=centroid_coordinates,
        )
        assert (
            len(shifted_centroid_centers.shape) == 2
            and shifted_centroid_centers.shape[1] == 5
        ), f"After local shift, are we not in 5D space? ({shifted_centroid_centers.shape[1]})"

        # TODO[Zain][generalize with KMeans] Centroid Update, via clustering
        centroids = shifted_centroid_centers
        assert (
            centroids.shape[1] == 5
        ), f"Before clustering, are we not in 5D space? ({centroids.shape[1]})"

        pixel_num = 0
        pixel_5d_coords = np.zeros((img.shape[0] * img.shape[1], 5))
        for y in np.arange(img.shape[0]):
            for x in np.arange(img.shape[1]):
                pixel_5d_coords[pixel_num] = np.concatenate(([x, y], img[y, x, :]))
                pixel_num += 1
        has_converged = False
        iter_num = 0
        while has_converged is False and iter_num < max_iter:
            # centroid IDs (as array indices) mapped to 2D arrays
            centroids_assigned_pts = defaultdict(list)
            _assign_centroid_1d = functools.partial(
                _assign_centroid,
                centroids_assigned_pts=centroids_assigned_pts,
                centroids=centroids,
                threshold_distance=2 * S,  # or a 100, as per the assignment
            )
            np.apply_along_axis(_assign_centroid_1d, axis=1, arr=pixel_5d_coords)

            # 2: update centroid placements themselves
            cap = centroids_assigned_pts  # just an abbreviation
            new_centroids = np.array(
                [
                    # CHeckpoint: are we effectively getting the new centroid?
                    np.mean(np.array(cap[centroid_label]), axis=0).astype(
                        int
                    )  # TODO[perhaps only round xy to int?]
                    for centroid_label in centroids_assigned_pts.keys()
                ]
            )
            centroids = new_centroids[:]
            assert (
                centroids.shape[1] == 5
            ), f"After clustering, are we not in 5D space? ({centroids.shape[1]})"

            # 3: decide if we should continue
            iter_num += 1
            if np.equal(centroids, new_centroids).all() or iter_num == max_iter:
                has_converged = True

        # C: collect the results - get a mapping of pixels to clusters
        pixel_to_cluster = dict()
        for centroid_coords, pixel_coords in centroids_assigned_pts.items():
            for single_pixel in pixel_coords:
                pixel_2d = tuple(single_pixel[:2].astype(int).tolist())
                if pixel_2d in pixel_to_cluster:
                    pixel_to_cluster[pixel_2d].append(centroid_coords)
                else:
                    pixel_to_cluster[pixel_2d] = [centroid_coords]

        # D: for plotting, start by assuming all pixels black (boundary of a cluster)
        superpixel_img = np.zeros_like(img)

        # assign a color to each centroid
        hashable_centroids = [tuple(centroid) for centroid in centroids.tolist()]
        rgb_colors_for_centroids = [
            tuple(
                (
                    random.uniform(0, 1) * 255.,
                    random.uniform(0, 1) * 255.,
                    random.uniform(0, 1) * 255.,
                )
            )
            for _ in centroids
        ]
        centroid_colors = dict(zip(hashable_centroids, rgb_colors_for_centroids))

        # give color to the pixels on the "interior" of some cluster
        for y in np.arange(superpixel_img.shape[0]):
            for x in np.arange(superpixel_img.shape[1]):
                coords = (
                    (x, y),
                    (x + 1, y),  # right
                    (x - 1, y),  # left
                    (x, y + 1),  # down
                    (x, y - 1),  # up
                )
                cluster_list = []
                for x_coord, y_coord in coords:
                    if 0 < x_coord < superpixel_img.shape[1]:
                        if 0 < y_coord < superpixel_img.shape[0]:
                            if (x_coord, y_coord) in pixel_to_cluster:
                                cluster_list.append(
                                    pixel_to_cluster[(x_coord, y_coord)]
                                )
                            elif logging:
                                print(
                                    f"Warning: couldn't find cluster for coords ({(x_coord, y_coord)}), which is suspicious..."
                                )
                                cluster_list.append([])

                clusters_in_common = _common_elements(*cluster_list)

                if len(clusters_in_common) > 0:
                    centroid_color_to_assign = centroid_colors[clusters_in_common[0]]
                    superpixel_img[y, x, :] = centroid_color_to_assign

        plt.imshow(superpixel_img)
        plt.title(plot_title)
        plt.show()


if __name__ == "__main__":
    white_tower_img = ops.load_image(
        "/Users/zainraza/Downloads/dev/courses/Stevens/CS-558/Image-Segmentation-and-Pixel-Classification/cs558F21_HW4_data/white-tower.png",
        return_array=True,
        return_grayscale=False,
    )
    SLIC.execute_and_visualize(
        white_tower_img,
        step_size=50,
        max_iter=3,
        xy_scaling_factor=2,
        plot_title="2A. SLIC on the White Tower",
    )
