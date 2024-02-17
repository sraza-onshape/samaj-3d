#!/usr/bin/env/python
from typing import Literal, Tuple, Union

import argparse
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from dolly_data_code_python import SampleCameraPath
from util import ops


def compute_DLT(
    source_points: np.ndarray[int],
    destination_points: np.ndarray[int],
    use_normalization: bool = False,
) -> np.ndarray[float]:
    """
    Compute the homography matrix using the Direct Linear Transformation (DLT) algorithm.

    Parameters:
        source_points (np.ndarray): Source points of shape (num_points, 2).
        destination_points (np.ndarray): Destination points of shape (num_points, 2).
        use_normalization (bool): Whether to use normalization.

    Returns:
        np.ndarray: Homography matrix of shape (3, 3).
    """

    ### HELPERS
    def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Scales the data so as not to throw off the solution to DLT."""
        # Compute the centroid of the points
        centroid = np.mean(points, axis=0).squeeze()  # (2,)

        # Compute the average distance from the centroid
        manhattan_distance = points - centroid
        euclidean_distances = np.linalg.norm(points - centroid)
        average_distance = (
            euclidean_distances.mean().squeeze()
        )  # this should be a scalar

        # Scale the points by the reciprocal of the average distance
        # normalized_points = [(point - centroid) / average_distance for point in points]
        normalized_points = manhattan_distance / average_distance

        return normalized_points, centroid, average_distance

    def _denormalize_homography_matrix(
        H: np.ndarray,
        centroid1: np.ndarray,
        centroid2: np.ndarray,
        scale1: float,
        scale2: float,
    ) -> np.ndarray:
        """
        Inverts the effects of scaling in xy,
        so the homography matrix H will properly warp pixels in image space.
        """
        # Denormalize the homography matrix based on the centroids and scales
        T1 = [
            [1 / scale1, 0, -1 * (centroid1[0] / scale1)],
            [0, 1 / scale1, -1 * (centroid1[1] / scale1)],
            [0, 0, 1],
        ]

        T2 = [[scale2, 0, centroid2[0]], [0, scale2, centroid2[1]], [0, 0, 1]]

        denormalized_h = np.linalg.inv(T2) @ H @ T1

        return denormalized_h

    ### DRIVER
    num_points = source_points.shape[0]
    if use_normalization:
        # Normalize source and destination points
        (source_points, centroid_source, scale_source) = _normalize_points(
            source_points
        )
        (
            destination_points,
            centroid_destination,
            scale_destination,
        ) = _normalize_points(destination_points)

    # Construct the homogeneous system of equations using normalized points
    A = []
    for i in range(num_points):
        y, x = source_points[i]
        y_prime, x_prime = destination_points[i]

        # this expression is adapted from the CMU slide 15 in this deck by Prof Kris Kitani: https://www.cs.cmu.edu/~16385/s17/Slides/10.2_2D_Alignment__DLT.pdf
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, x_prime * y, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    # Use a linear solver to solve the system of equations Ax = 0
    V = np.linalg.svd(A).Vh
    # X = V[:, -1]  # as mentioned in slide 89, we need the last column of V
    X = V[-1]
    # Reshape the solution into a 3x3 normalized homography matrix
    H = X.reshape(3, 3)

    if use_normalization:
        # Denormalize the homography matrix
        H = _denormalize_homography_matrix(
            H,
            centroid_source,
            centroid_destination,
            scale_source,
            scale_destination,
        )

    return H


def generate_image(
    original_image: np.ndarray,
    forward_transformation_matrix: np.ndarray,
    new_image_dimensions: Tuple[int, int],
    use_rgb: bool = True,
    use_logging: bool = False,
) -> np.ndarray:
    """
    Applies an inverse warp to a 2D image (with bilinear interpolation).

    Parameters:
        original_image(array-like): a 2d or 3d array, depending on the number of channels. Should be in channels-last format.
        forward_transformation_matrix(array-like): the 2D homography matrix
        new_image_dimensions(int, int): specify the (height, width) you want the new image to have. Doesn't have to match those of the input image.
        use_rgb(bool): specify if the output image should have 3 color channels
        use_logging

    Returns: nd.ndarray: the transformed image
    """

    ### HELPER(S)
    def _index_image_with_default(
        image: np.ndarray,
        row: int,
        col: int,
        default: np.ndarray = np.zeros(3),
    ) -> float:
        """
        Try to get the value at the specified location
        in a RGB image (channels-last format),
        or else return a default.
        """
        if -1 < row < image.shape[0] and -1 < col < image.shape[1]:
            return image[row, col, :].squeeze()
        else:
            return default

    def _interpolate(
        lower_left: np.ndarray,
        upper_left: np.ndarray,
        lower_right: np.ndarray,
        upper_right: np.ndarray,
        a: float,
        b: float,
        image_channel: int,
    ) -> float:
        x_weight = 1 - a
        y_weight = 1 - b
        new_pixel_channel_value = (
            (x_weight * y_weight * lower_left[image_channel])
            + (a * y_weight * lower_right[image_channel])
            + (a * b * upper_right[image_channel])
            + (x_weight * b * upper_left[image_channel])
        )
        return new_pixel_channel_value.astype(np.uint8)

    ### DRIVER
    # 1. init a blank image of 940 x 500 x 3
    new_height, new_width = new_image_dimensions
    new_depth = 1
    if use_rgb:
        new_depth = 3
    new_image = np.zeros((new_height, new_width, new_depth))

    # 2. get the inverse of the homography matrix
    inverse_transform = np.linalg.inv(forward_transformation_matrix)

    # 3. fill in the image
    for y in np.arange(new_height):
        for x in np.arange(new_width):
            # get the corresponding location in the original image
            new_image_location = np.array([x, y, 1]).reshape(1, 3)
            origin_image_location = (
                (new_image_location @ inverse_transform).reshape(1, 3).squeeze()
            )
            origin_image_x, origin_image_y = (
                origin_image_location[:2] / origin_image_location[2]
            )

            # copy the pixel value over, if possible
            if use_logging:
                print(
                    f"I compute old coords ({origin_image_x}, {origin_image_y}) ==> new coords ({x}, {y})"
                )
            if (
                -1 < origin_image_y < original_image.shape[0]
                and -1 < origin_image_x < original_image.shape[1]
                and origin_image_y.is_integer()
                and origin_image_x.is_integer()
            ):
                # copy the pixel value over, if possible
                if use_logging:
                    print("copying pixel values...")
                origin_image_x = int(origin_image_x)
                origin_image_y = int(origin_image_y)
                new_image[y, x, :] = original_image[origin_image_y, origin_image_x, :]
            # otherwise, interpolate the pixel value using bilinear interpolation
            else:  #
                lower_y, upper_y = np.floor(origin_image_y).astype(int), np.ceil(
                    origin_image_y
                ).astype(int)
                lower_x, upper_x = np.floor(origin_image_x).astype(int), np.ceil(
                    origin_image_x
                ).astype(int)

                # get the pixels usable for interpolation
                lower_left = _index_image_with_default(original_image, lower_y, lower_x)
                upper_left = _index_image_with_default(original_image, upper_y, lower_x)
                lower_right = _index_image_with_default(
                    original_image, lower_y, upper_x
                )
                upper_right = _index_image_with_default(
                    original_image, upper_y, upper_x
                )
                if use_logging:
                    print(
                        f"getting the four points of interest at: {lower_left, upper_left, lower_right, upper_right}"
                    )
                delta_y = abs(origin_image_y - lower_y)
                delta_x = abs(origin_image_x - lower_x)

                # interpolate the value, across all three RGB channels
                new_pixel_value = np.column_stack(
                    [
                        _interpolate(
                            lower_left,
                            upper_left,
                            lower_right,
                            upper_right,
                            delta_x,
                            delta_y,
                            image_channel=channel_index,
                        )
                        for channel_index in np.arange(3)
                    ]
                )
                if use_logging:
                    print(f"getting the new_pixel_value: {new_pixel_value}")
                new_image[y, x, :] = new_pixel_value

    # 4. done!
    return new_image


def render_video_in_circular_path(
    save_directory: str,
    camera_start_position: np.ndarray,
    object_position_worldspace: np.ndarray,
    video_length_in_sec: float = 5.0,
    frames_per_sec: int = 5,
    scale_factor_focal_length: float = 1.125,
    total_rotation_in_radians: float = np.pi,
    path_to_point_cloud: str = "data.obj",
    direction: Union[Literal["clockwise"], Literal["counter-clockwise"]] = "clockwise",
    render_synchronously: bool = False,
) -> None:
    """
    Create a video by circling a pinhole camera around the fish statue.

    We try to keep the camera at a constant radius throughout its path.
    The video is saved in Windows Media Video (WMV) format.

    Parameters:
        save_directory(str): relative path to the directory which will store the video
        camera_start_position(np.ndarray): a XYZ coordinate in world space.
                                           Only used to deternp.ndarray
        object_position_worldspace(np.ndarray): the center of the circle you
                                                want the camera to traverse
        video_length_in_sec(float): specify the duration of the output video
        frames_per_sec(int): the frequency of frames being shown in the video
        scale_factor_focal_length(float): can be used to make the camera zoom in/out
        total_rotation_in_radians(float): the angle measure of the path to travel.
                                          Please only pass an unsigned value.
                                          The default is to only travel a half circle,
                                          or 180 degrees.
        path_to_point_cloud(str): relative path to the 3D scene
        direction("clockwise" | "counter-clockwise"): whether you want to travel in the
                                                      positive or negative direction along
                                                      unit circle.
        render_synchronously(bool): if one wishes, we can show the video right after it is created

    Returns: None
    """

    ########## HELPER(S) ##########
    def _compute_rotation_matrix(
        camera_position: np.ndarray, object_position: np.ndarray
    ):
        """### Attempt 8 - who cares what angle of rotation is used, just point the camera at the object"""
        # Compute direction vector from camera to object
        direction_vector = object_position - camera_position

        # Normalize direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute rotation angles to align direction vector with the camera's viewing direction (the positive Z-axis)
        pitch = theta = np.arctan(direction_vector[0], direction_vector[2]).squeeze()
        yaw = psi = np.arcsin(direction_vector[1]).squeeze()

        # Construct rotation matrix based on computed rotation angles (yaw, pitch)
        rotation_about_y = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        rotation_about_x = np.eye(3)
        rotation_matrix_worldspace = rotation_about_x @ rotation_about_y

        return rotation_matrix_worldspace

    ########## DRIVER ##########
    # Setup
    num_frames = int(video_length_in_sec * frames_per_sec)
    circle_radius_xz = np.linalg.norm(
        camera_start_position - object_position_worldspace
    )
    if direction == "counter-clockwise":
        total_rotation_in_radians = -1 * total_rotation_in_radians
    angles_in_xz = np.linspace(0, total_rotation_in_radians, num_frames)
    camera_positions_x = object_position_worldspace[0] + (
        (circle_radius_xz * np.cos(angles_in_xz))
    )
    camera_positions_z = object_position_worldspace[2] + (
        (circle_radius_xz * np.sin(angles_in_xz))
    )
    camera_positions = np.column_stack([camera_positions_x, camera_positions_z])
    assert camera_positions.shape == (num_frames, 2)

    # load object file to retrieve data
    frame_names = list()
    with open(path_to_point_cloud, "rb") as file_p:
        camera_objs = pickle.load(file_p)

        # extract objects from object array
        crop_region = camera_objs[0].flatten()
        filter_size = camera_objs[1].flatten()
        K = camera_objs[2]
        print("K  (before change): ", K)
        K[1, 1] *= scale_factor_focal_length
        K[0, 0] *= scale_factor_focal_length
        print("K  (after change): ", K)
        ForegroundPointCloudRGB = camera_objs[3]
        BackgroundPointCloudRGB = camera_objs[4]

        # create variables for computation
        # - background has the XYZ coords,
        # - foreground has the RGB values
        data3DC = (BackgroundPointCloudRGB, ForegroundPointCloudRGB)
        object_position_worldspace = object_position_worldspace.reshape(3, 1)

        # Render image from camera viewpoint
        vertical_height = camera_start_position[1]
        for i, camera_pos in enumerate(camera_positions):
            start = time.time()
            t = np.array([camera_pos[0], vertical_height, camera_pos[1]]).reshape(3, 1)
            print(f"Iter {i}, position: {t}")
            fname = f"video-frame-{i}.jpg"
            print("\nGenerating {}".format(fname))
            R = _compute_rotation_matrix(t, object_position_worldspace)
            M = np.matmul(K, (np.hstack((t, R))))
            print("M:", M)
            img = SampleCameraPath.PointCloud2Image(
                M, data3DC, crop_region, filter_size
            )

            # Convert image values from (0-1) to (0-255) and change type from float64 to float32
            img = 255 * (np.array(img, dtype=np.float32))

            # convert image from RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # write image to file 'fname'
            img_path = f"{save_directory}/{fname}"
            cv2.imwrite(img_path, img_bgr)
            frame_names.append(img_path)

            end = time.time()
            print("{0:.4g} s".format(end - start))

    # Combine rendered images into a video
    frames = [cv2.imread(file_path) for file_path in frame_names]
    height, width, _ = frames[0].shape
    video_path = f"{save_directory}/output_video.wmv"
    video_writer = cv2.VideoWriter(
        video_path,
        fourcc=cv2.VideoWriter_fourcc(*"WMV2"),
        fps=5,
        frameSize=(width, height),
        isColor=True,
    )
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

    # Display the video file
    if render_synchronously:
        cap = cv2.VideoCapture(video_path)

        # Check if the video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        # Read and display frames
        while True:
            ret, frame = cap.read()  # Read a frame
            if not ret:
                break  # Break the loop if no more frames are available
            cv2.imshow("Frame", frame)  # Display the frame
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break  # Press 'q' to quit

        # Release the VideoCapture object and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        required=False,
        type=int,
        default=0,
        help="Pass '0' to produce the image that uses pixel features, '1' to produce an image using line features, and '2' to produce a video of the 3D scene.",
    )
    args = parser.parse_args()

    # read the data
    original_bball_court = ops.load_image(
        "./basketball-court.ppm",
        return_array=True,
        return_grayscale=False,
    )
    court_length_along_x = 450
    court_length_along_y = 900

    # peform the appropiate function
    if args.operation == 0:
        print("I'm launching a Python app to visualize your warped image shortly...")
        # Let's pick the four corners of the court as the points we'll use for homography estimation:
        x = np.array([23, 246, 402, 279])
        y = np.array([193, 51, 74, 279])
        source_points = np.dstack([y, x]).squeeze()
        destination_points = np.vstack(
            [
                (940, 0),
                (0, 0),
                (0, 500),
                (940, 500),
            ]
        )
        homography_estimate_based_on_pixel_features = compute_DLT(
            source_points, destination_points, use_normalization=False
        )
        image_from_top = generate_image(
            original_bball_court,
            homography_estimate_based_on_pixel_features,
            new_image_dimensions=(court_length_along_y, court_length_along_x),
            use_rgb=True,
            # WARNING: setting `use_logging=True` causes a
            # VERY high increase in memory usage, and can cause
            # your machine to crash - don't do it if you're not certain you can avoid OOM failures!
            use_logging=False,
        )
        plt.imshow(image_from_top.astype(np.uint8))
        plt.title("1.b. Attempted Warping of Basketball Court (with 4 Selected Points)")
        plt.axis("off")
        plt.show()
    elif args.operation == 1:
        print("I'm launching a Python app to visualize your warped image shortly...")

        # define lines in the input
        right_perimeter_of_court_source = np.cross(
            (402, 74, 1),
            (279, 279, 1),
        )
        left_perimeter_of_court_source = np.cross(
            (23, 193, 1),
            (246, 51, 1),
        )
        half_court_line_source = np.cross(
            (362, 141, 1),
            (168, 101, 1),
        )
        right_bleachers_courtside_source = np.cross(
            (429, 71, 1),
            (318, 308, 1),
        )
        left_bleachers_courtside_source = np.cross((17, 173, 1), (227, 49, 1))
        source_lines = np.vstack([
            right_perimeter_of_court_source,
            left_perimeter_of_court_source,
            half_court_line_source,
            right_bleachers_courtside_source,
            left_bleachers_courtside_source,
        ])
        source_lines = np.stack(
            [
                source_lines[:, 0] / source_lines[:, 2],
                source_lines[:, 1] / source_lines[:, 2],
            ],
            axis=1,
        )

        # define lines in the output
        destination_image_dims = (940, 500)
        right_perimeter_of_court_destination = np.array([1, 0, destination_image_dims[1] * (7/8)])
        left_perimeter_of_court_destination = np.array([1, 0, destination_image_dims[1] * (3/8)])
        half_court_line_destination = np.array([0, 1, destination_image_dims[0]/2])
        right_bleachers_courtside_destination = np.array([1, 0, destination_image_dims[1] * (3/4)])
        left_bleachers_courtside_destination = np.array([1, 0, destination_image_dims[1] * (1/4)])
        destination_lines = np.vstack([
            right_perimeter_of_court_destination,
            left_perimeter_of_court_destination,
            half_court_line_destination,
            right_bleachers_courtside_destination,
            left_bleachers_courtside_destination,
        ])
        destination_lines = np.stack(
            [
                destination_lines[:, 0] / destination_lines[:, 2],
                destination_lines[:, 1] / destination_lines[:, 2],
            ],
            axis=1,
        )

        homography_estimate_based_on_line_features = compute_DLT(
            source_lines, destination_lines, use_normalization=False
        )
        homography_estimate_based_on_line_features

        # Render a New Attempt at Image Warping
        image_from_top_attempt_3 = generate_image(
            original_bball_court,
            homography_estimate_based_on_line_features,
            new_image_dimensions=(940, 500),
            use_rgb=True,
            use_logging=False,
        )
        plt.imshow(image_from_top_attempt_3.astype(np.uint8))
        plt.title("1.c. Attempted Warping of Basketball Court (with Line-Based Homography)")
        plt.axis("off")
        plt.show()
    elif args.operation == 2:
        print("Rendering a new video...")
        render_video_in_circular_path(
            save_directory="./generated-images/problem-2/",
            camera_start_position=np.array([0.12, 0.30, -2.32]),
            object_position_worldspace=np.array([0.12, 0.30, -3.265]),
            video_length_in_sec=5.0,
            frames_per_sec=5,
            scale_factor_focal_length=1.125,
            total_rotation_in_radians=np.pi,
            path_to_point_cloud="./code/dolly_data_code_python/data.obj",
            render_synchronously=True
        )


if __name__ == "__main__":
    main()
