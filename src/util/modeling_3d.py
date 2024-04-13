from io import TextIOWrapper

import functools
import numpy as np


class PlySerializer:
    @staticmethod
    def save_colored_point_cloud(
        xyz_points: np.ndarray,
        rgb_values: np.ndarray,
        path_to_save: str = "./myColoredPtCloud.ply",
    ) -> None:
        """
        Serializes a point cloud as a PLY file in the ASCII format.

        We use 3 `float` properties to represent the XYZ locations of each point,
        and 3 `uchar` properties to represent their respective RGB values.

        Parameters:
            xyz_points(2D NumPy array): contains nx3 values - the XYZ worldspace
                coordinates of each point included in the 3D model
            rgb_values(2D NumPy array): contains nx3 values - the corresponding
                RGB assignments for each input point.
            path_to_save(str): save location of the output PLY.
                Assumes the file name and the ".ply" extension are included at
                the end. It will be displayed in a log message if/when the
                save process is successfully completed.

        Returns: None
        """
        ### HELPER(S)
        def _write_one_point(xyz_rgb_1point: np.ndarray, f: TextIOWrapper) -> None:
            xyz_1point, rgb_1point = xyz_rgb_1point[:3], xyz_rgb_1point[3:]

            xyz_list_repr = xyz_1point.tolist()
            xyz_str_repr = str(xyz_list_repr)
            xyz_ply_repr = " ".join(xyz_str_repr[1:-1].split(","))

            rgb_list_repr = rgb_1point.astype(int).tolist()  # because it fits into the uchar type
            rgb_str_repr = str(rgb_list_repr)
            rgb_ply_repr = " ".join(rgb_str_repr[1:-1].split(","))

            f.write(f"{xyz_ply_repr} {rgb_ply_repr}\n")

        ### DRIVER
        num_vertices = xyz_points.shape[0]
        xyz_rgb_all = np.column_stack([xyz_points, rgb_values])

        with open(path_to_save, "w") as f:
            # write the header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("element face 0\n")
            f.write("end_header\n")
            # write the points and their colors
            _write_one_point_simplified = functools.partial(_write_one_point, f=f)
            np.apply_along_axis(_write_one_point_simplified, 1, xyz_rgb_all)

        print(f"Point cloud saved to {path_to_save}.")
