import functools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, k: int):
        """
        The k-means algorithm in Python.
        """
        self.centroid_coords = dict()
        self.num_clusters = k

    def fit(self, X: np.array, max_iter: int = float("inf")) -> dict:
        """
        Assumes k is a positive integer.

        Returns: dict: contains the dict and covariance of each centroid
        """

        ### HELPER(S)
        def _assign_centroid(sample, centroids_assigned_pts, distances_to_centroids):
            centroid_assignment = np.argmin(
                # Euclidean distance
                (np.linalg.norm(sample) - distances_to_centroids)
                ** 2
            )
            centroids_assigned_pts[centroid_assignment].append(sample)

        ### DRIVER
        # A: init k random centroids from existing data
        rng = np.random.default_rng()
        centroids = rng.choice(X, size=(self.num_clusters,))

        # B: converge on optimal centroids
        has_converged = False

        iter_num = 0
        while has_converged is False and iter_num < max_iter:
            # 1: assign each point to a centroid
            centroids_assigned_pts = dict(
                zip(
                    # scalars mapped to 2D arrays
                    range(self.num_clusters),
                    [[] for _ in range(self.num_clusters)],
                )
            )
            distances_to_centroids = np.linalg.norm(centroids, axis=1)
            _assign_centroid_1d = functools.partial(
                _assign_centroid,
                centroids_assigned_pts=centroids_assigned_pts,
                distances_to_centroids=distances_to_centroids,
            )
            np.apply_along_axis(_assign_centroid_1d, axis=1, arr=X)

            # 2: update centroid placements themselves
            cap = centroids_assigned_pts  # just an abbreviation
            new_centroids = np.array(
                [
                    np.mean(np.array(cap[centroid_label]), axis=0)
                    for centroid_label in centroids_assigned_pts.keys()
                ]
            )
            centroids = new_centroids[:]

            # 3: decide if we should continue
            iter_num += 1
            if np.equal(centroids, new_centroids).all() or iter_num == max_iter:
                has_converged = True

        # C: collect the results
        for centroid_label in centroids_assigned_pts:
            self.centroid_coords[centroid_label] = centroids[centroid_label, :]
        return self.centroid_coords

    def predict(self, X: np.array) -> List[int]:
        centroids = [coords for coords in self.centroid_coords.values()]
        centroid_assignment = [
            np.argmin(
                [np.linalg.norm(sample - centroids, axis=1)]  # Euclidean distance
            )
            for sample in X
        ]

        return centroid_assignment

    @classmethod
    def fit_and_visualize(
        cls: "KMeans",
        points: np.ndarray,
        axis_labels: Dict[str, str],
        num_clusters: int = 10,  # as per Hw 4 description
        plot_title: str = "",
        max_iter: int = float("inf"),
        figsize: Tuple[int, int] = (12, 12),
    ) -> None:
        """TODO[Zain]: add docstring"""
        # find the clusters in the dataset
        kmeans = cls(k=num_clusters)
        kmeans.fit(points, max_iter)

        # visualize the points - TODO, in the future we can generalize this to work for 2D also
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        ax.scatter(xs, ys, zs, color="g", s=.05, alpha=.05)

        # visualize the cluster centers
        centroids = np.array(list(kmeans.centroid_coords.values()))
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        zs = centroids[:, 2]
        ax.scatter(xs, ys, zs, color="r", s=550, alpha=1)

        ax.set_xlabel(axis_labels["x"])
        ax.set_ylabel(axis_labels["y"])
        ax.set_zlabel(axis_labels["z"])

        plt.title(plot_title)
