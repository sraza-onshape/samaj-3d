import numpy as np

from util import ops


class RankTransform2D:
    """Performs rank filtering on an image."""

    @staticmethod
    def transform(
        image: np.ndarray,
        filter_side_length: int,
        do_logging: bool = False,
    ) -> np.ndarray:
        """
        Produce a new image where each cell value
        represents the "rank" of the corresponding pixel in the input
        (i.e., the index of said pixel in a sorted list of itself &
        the neighboring pixel values).

        Parameters:
            image(np.ndarray): a 2D array representing a single grayscale image
            filter_side_length(int): this is k. The size of each local neighborhood
                              will be kxk. Please pass an odd value > 0.
            do_logging(bool): enables print statements to display intermediate
                              values during execution

        Returns: np.ndarray: the transformed image
        """

        ### HELPER(S)
        def compute_rank(
            image: np.ndarray,
            kernel: np.ndarray,
            row_index: int,
            col_index: int,
        ) -> float:
            """
            Computes the rank of 1 local window of the image.

            Parameters:
                image(array-like): see above
                kernel(array-like): tells us the size of the window
                row_index, col_index: int: the coordinates of the upper left corner
                                            of the block of pixels being ranked

            Returns: int: the rank of the center pixel of the window
            """
            # A: define useful vars
            kernel_h, kernel_w = kernel.shape
            # B: get the block of pixels needed for the convolution
            block_of_pixels = image[
                row_index : (kernel_h + row_index),
                col_index : (kernel_w + col_index)
            ]
            # C: count the of # higher than the center
            center_val = block_of_pixels[kernel_h // 2, kernel_w // 2]
            if do_logging:
                print(
                    f"I think that {center_val} is at the center of {block_of_pixels}"
                )
            transformed_block = np.where(block_of_pixels < center_val, 1, 0)
            if do_logging:
                print(
                    f"Transformed block <{block_of_pixels}> into: <{transformed_block}>"
                )
            return np.sum(transformed_block)

        ### DRIVER
        # data validation
        assert isinstance(image, np.ndarray)
        assert image.shape > (0, 0)
        assert isinstance(filter_side_length, int)
        assert filter_side_length > 0 and filter_side_length % 2 == 1

        # make a copy of the img, padded - will be an intermediate repr
        kernel = np.ones((filter_side_length, filter_side_length))
        padded_image, _, _ = ops.pad(
            image, kernel, stride=1, padding_type="zero"
        å)
        # fill in the output
        stride = 1
        output_image = list()
        kernel_h, _ = kernel.shape
        # iterate over the rows and columns
        starting_row_ndx = 0
        while starting_row_ndx <= len(padded_image) - kernel_h:
            # convolve the next row of this channel
            next_channel_row = ops.slide_kernel_over_image(
                padded_image,
                kernel,
                starting_row_ndx,
                stride,
                apply=compute_rank,
            )
            # now, add the convolved row to the list
            output_image.append(next_channel_row)
            # move to the next starting row for the filtering
            starting_row_ndx += stride
        # stack the channels, and return
        return np.array(output_image).squeeze()
