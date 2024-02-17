import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


class PanoramaStitcher:
    @staticmethod
    def composite_images(
        left_img,
        right_img,
        affine_transform_matrix_from_left_to_right,
        affine_transform_offset,
        overlap_start_coordinate_x: int,
        plot_title: str = "",
    ) -> np.ndarray:
        """
        Horizontally align two overlapping images using an affine transform.
        As per the hw 3 description, we composite by averaging where the images
        overlap.
        """
        panorama_output_shape = (
            max(left_img.shape[0], right_img.shape[0]),
            left_img.shape[1] + right_img.shape[1],
        )

        # Prepare for the affine transformation
        affine_transform_matrix_from_right_to_left = np.linalg.inv(
            affine_transform_matrix_from_left_to_right
        ).T
        affine_transform_offset_inv = affine_transform_offset
        affine_transform_offset_inv[:2] *= -1

        # Apply the affine transformation to warp the left image onto the right
        left_img_transformed = ndimage.affine_transform(
            left_img,
            affine_transform_matrix_from_right_to_left,
            offset=np.squeeze(affine_transform_offset_inv),
        )

        # Composite the two images into 1 panorama
        panorama_img = np.zeros(panorama_output_shape)
        panorama_img[:, :overlap_start_coordinate_x] = left_img_transformed[:, :overlap_start_coordinate_x]
        panorama_img[:, overlap_start_coordinate_x:left_img_transformed.shape[1]] = (
            left_img_transformed[:, overlap_start_coordinate_x:left_img_transformed.shape[1]]
            + right_img[:, 0:(left_img_transformed.shape[1] - overlap_start_coordinate_x)]
        ) / 2
        panorama_img[
            :, left_img_transformed.shape[1]:left_img_transformed.shape[1] 
                + overlap_start_coordinate_x
        ] = right_img[:, (left_img_transformed.shape[1] - overlap_start_coordinate_x):]


        # Display the result
        plt.imshow(panorama_img, cmap="gray")
        plt.title(plot_title)
        plt.axis("off")  # for cleanliness, hide the axis lines and ticks
        plt.show()
