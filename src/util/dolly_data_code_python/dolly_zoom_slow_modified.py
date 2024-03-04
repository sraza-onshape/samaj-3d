import argparse
import numpy as np
import pickle
import time
import cv2
from scipy.signal import medfilt
from scipy.ndimage import maximum_filter as maxfilt


def PointCloud2Image(
    M: list[list[int | float]],  #
    Sets3DRGB: list,
    viewport: np.ndarray,
    filter_size: int | list[int],
    enable_max_filter: bool = False,
    resolution_scale_factor: float = 1.0,
    use_depth_logging: bool = False,
):
    """
    Renders one image from the 3D point cloud.

    Parameters:
        M: the 3x4 projection matrix
        Sets3DRGB: a 2x3xn list, containing n points in the scene.
            The first 3 rows encode XYZ locations. The last 3 encode RGB values.
        viewport: 4-vector that encodes the image dimensions.
        filter_size: ignored for now - in theory, we could use this when
            calling `scipy.signal.medfilt`
        enable_max_filter: bool that can turn on 2D max_filtering
            (an operation used to "fill-in" blanks in the image).
        resolution_scale_factor: A multiplier used to reduce the image
            heightxwidth (from the default of 2048x3072).
        use_depth_logging: print the closest and farthest depth of 3D scene points (assumed to be in meters).

    Returns: np.ndarray: a RGB image in channels-last format
    """
    # setting up output image
    print("...Initializing 2D image...")
    top = viewport[0]
    left = viewport[1]

    if resolution_scale_factor == 1.0:
        h, w = viewport[2], viewport[3]

    elif resolution_scale_factor != 1.0:
        h = int(viewport[2] / (resolution_scale_factor // 2))
        w = int(viewport[3] / (resolution_scale_factor // 2))
    bot = top + h + 1
    right = left + w + 1

    output_image = np.zeros((h + 1, w + 1, 3))

    depth_range: list = [float("inf"), float("-inf")]
    for counter in range(len(Sets3DRGB)):
        print("...Projecting point cloud into image plane...")

        # clear drawing area of current layer
        canvas = np.zeros((bot, right, 3))

        # segregate 3D points from color
        dataset = Sets3DRGB[counter]
        P3D = dataset[:3, :]
        color = (dataset[3:6, :]).T

        # form homogeneous 3D points (4xN)
        len_P = len(P3D[1])
        ones = np.ones((1, len_P))
        X = np.concatenate((P3D, ones))

        # collect the depth range
        depth_range[0] = min(depth_range[0], np.min(X[2, :]))
        depth_range[1] = max(depth_range[1], np.max(X[2, :]))

        # apply (3x4) projection matrix
        x = np.matmul(M, X)

        # normalize by 3rd homogeneous coordinate
        x = np.around(np.divide(x, np.array([x[2, :], x[2, :], x[2, :]])))

        # truncate image coordinates
        x[:2, :] = np.floor(x[:2, :])

        # determine indices to image points within crop area
        i1 = x[1, :] > top
        i2 = x[0, :] > left
        i3 = x[1, :] < bot
        i4 = x[0, :] < right
        ix = np.logical_and(i1, np.logical_and(i2, np.logical_and(i3, i4)))

        # make reduced copies of image points and cooresponding color
        rx = x[:, ix]
        rcolor = color[ix, :]

        for i in range(len(rx[0])):
            canvas[int(rx[1, i]), int(rx[0, i]), :] = rcolor[i, :]

        # crop canvas to desired output size
        cropped_canvas = canvas[top : top + h + 1, left : left + w + 1]

        if enable_max_filter:
            # filter individual color channels
            shape = cropped_canvas.shape
            filtered_cropped_canvas = np.zeros(shape)
            print("...Running 2D filters...")
            for i in range(3):
                # max filter
                filtered_cropped_canvas[:, :, i] = maxfilt(cropped_canvas[:, :, i], 5)
            cropped_canvas = filtered_cropped_canvas

        # get indices of pixel drawn in the current canvas
        drawn_pixels = np.sum(cropped_canvas, 2)
        idx = drawn_pixels != 0
        shape = idx.shape
        shape = (shape[0], shape[1], 3)
        idxx = np.zeros(shape, dtype=bool)

        # make a 3-channel copy of the indices
        idxx[:, :, 0] = idx
        idxx[:, :, 1] = idx
        idxx[:, :, 2] = idx

        # erase canvas drawn pixels from the output image
        output_image[idxx] = 0

        # sum current canvas on top of output image
        output_image = output_image + cropped_canvas

    if use_depth_logging:
        print(f"Range of scene depth (assuming units in meters): {depth_range}")

    print("Done")
    return output_image


def SampleCameraPath(
    enable_max_filter: bool = False,
    resolution_scale_factor: float = 1.0,
    save_dir: str = "./",
    num_frames: int = 8,
    enable_depth_logging: bool = False,
) -> None:
    """
    Example script for rendering images in a "path" around the fish statue.

    None of these variables needs to be modified - they are provided in the data file:
    - BackgroundPointCloudRGB
    - ForegroundPointCloudRGB
    - K
    - crop_region
    - filter_size

    Parameters:
        enable_max_filter(bool): can turn on 2D max_filtering
            (an operation used to "fill-in" blanks in the image).
        resolution_scale_factor(float): A multiplier used to reduce the image
            heightxwidth (from the default of 2048x3072).
        focal_length_scale_factor(float): A multiplier used to reduce
            the focal length - in effect adjusting the camera FOV.
        save_dir(str): Relative path to a folder for storing rendered images.
        enable_depth_logging(bool): print the closest and farthest depth of 3D scene points (assumed to be in meters).

    Returns: None
    """
    # load object file to retrieve data
    file_p = open("./data/data.obj", "rb")
    camera_objs = pickle.load(file_p)

    # extract objects from object array
    crop_region = camera_objs[0].flatten()
    filter_size = camera_objs[1].flatten()
    K = camera_objs[2]
    ForegroundPointCloudRGB = camera_objs[3]
    BackgroundPointCloudRGB = camera_objs[4]

    # scale the focal lengths
    if resolution_scale_factor != 1.0:
        print(f"intrinisic matrix (before changes): ", K)
        focal_length_scale_factor = 1 / (resolution_scale_factor // 2)
        K[0, 0] *= focal_length_scale_factor
        K[1, 1] *= focal_length_scale_factor

        # adjust the principal point
        K[0, 2] *= focal_length_scale_factor
        K[1, 2] *= focal_length_scale_factor
        print(f"intrinisic matrix (after changes): ", K)

    # create variables for computation
    data3DC = (BackgroundPointCloudRGB, ForegroundPointCloudRGB)
    R = np.identity(3)
    move = np.array([0.155, 0.0, 0]).reshape((3, 1))

    for step in range(num_frames):
        tic = time.time()

        fname = "StereoOutput{}.jpg".format(step)
        print("\nGenerating {}".format(fname))
        t = step * move
        print(f"Rotation matrix (R): {R}")
        print(f"Translation (t): {t}")
        M = np.matmul(K, (np.hstack((R, t))))

        img = PointCloud2Image(
            M,
            data3DC,
            crop_region,
            filter_size,
            enable_max_filter=enable_max_filter,
            resolution_scale_factor=resolution_scale_factor,
            use_depth_logging=enable_depth_logging,
        )

        # Convert image values form (0-1) to (0-255) and cahnge type from float64 to float32
        img = 255 * (np.array(img, dtype=np.float32))

        # convert image from RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # write image to file 'fname'
        cv2.imwrite(f"{save_dir}{fname}", img_bgr)

        toc = time.time()
        toc = toc - tic
        print("{0:.4g} s".format(toc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-max-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="turn on/off 2D max_filtering",
    )
    parser.add_argument(
        "--resolution-scale-factor",
        required=False,
        type=float,
        default=1.0,
        help="A multiplier used to reduce the image heightxwidth (from the default of 2048x3072).",
    )
    parser.add_argument(
        "--save-dir",
        required=False,
        type=str,
        default="./",
        help="Relative path to a folder for storing rendered images.",
    )
    parser.add_argument(
        "--use-depth-logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the closest and farthest depth of 3D scene points (assumed to be in meters).",
    )
    parser.add_argument(
        "--num-frames",
        required=False,
        type=int,
        default=8,
        help="Number of images to render along the camera path.",
    )
    args = parser.parse_args()

    SampleCameraPath(
        enable_max_filter=args.use_max_filter,
        resolution_scale_factor=args.resolution_scale_factor,
        save_dir=args.save_dir,
        num_frames=args.num_frames,
        enable_depth_logging=args.use_depth_logging,
    )


if __name__ == "__main__":
    main()
