from abc import ABC, abstractmethod
import cv2
import numpy as np


class PointSelectionStrategy(ABC):
    """
    The base class of the strategy, defining how points should be selected.
    """

    @abstractmethod
    def get_points(self, image, num_points):
        pass

    @staticmethod
    def get_valid_point(image, point):
        """
        Ensure that the point falls within the white region. If not, select a new random
        point in the white region.
        """
        white_pixels = np.argwhere(image == 255)
        while image[point[1], point[0]] != 255:
            random_pixel = white_pixels[np.random.choice(white_pixels.shape[0])]
            point = [random_pixel[1], random_pixel[0]]  # Switch order to X,Y

        return point


class RandomPointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting random points from each region in the image.
    """

    def get_points(self, image, num_points):
        # Find the connected components (regions) in the image
        _, labels, _, _ = cv2.connectedComponentsWithStats(image, connectivity=4)

        random_points = []
        for region_id in range(1, np.max(labels) + 1):
            region_mask = np.uint8(labels == region_id) * 255
            region_white_pixels = np.argwhere(region_mask == 255)

            if len(region_white_pixels) > 0:
                points_to_sample = min(num_points, len(region_white_pixels))
                random_indices = np.random.choice(
                    range(len(region_white_pixels)),
                    size=points_to_sample,
                    replace=False,
                )
                random_points.extend(region_white_pixels[random_indices])

        valid_points = [
            self.get_valid_point(image, [point[1], point[0]]) for point in random_points
        ]

        return valid_points


class CentroidPointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting the centroid point from each region in the image.
    """

    def get_points(self, image, num_points):
        # Find the connected components (regions) in the image
        _, labels, _, _ = cv2.connectedComponentsWithStats(image, connectivity=4)

        centroid_points = []
        for region_id in range(1, np.max(labels) + 1):
            region_mask = np.uint8(labels == region_id) * 255
            region_moments = cv2.moments(region_mask)
            if region_moments["m00"] > 0:
                centroid_x = int(region_moments["m10"] / region_moments["m00"])
                centroid_y = int(region_moments["m01"] / region_moments["m00"])
                centroid_points.append(
                    self.get_valid_point(image, [centroid_x, centroid_y])
                )

        return centroid_points


class CombinedPointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for combining multiple point selection strategies.
    """

    def __init__(self, *strategies):
        self.strategies = strategies

    def get_points(self, image, num_points):
        combined_points = []
        remaining_points = num_points

        for strategy in self.strategies:
            if remaining_points <= 0:
                break

            points_to_sample = min(remaining_points, num_points // len(self.strategies))
            points = strategy.get_points(image, points_to_sample)

            combined_points.extend(points)
            remaining_points -= len(points)

        return combined_points


class EdgePointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting points on the edges of each region in the image.
    """

    def get_points(self, image, num_points):
        # Find the connected components (regions) in the image
        _, labels, _, _ = cv2.connectedComponentsWithStats(image, connectivity=4)

        edge_points = []
        for region_id in range(1, np.max(labels) + 1):
            region_mask = np.uint8(labels == region_id) * 255

            edges = cv2.Canny(region_mask, 100, 200)
            edge_pixels = np.argwhere(edges > 0)

            if len(edge_pixels) > 0:
                points_to_sample = min(num_points, len(edge_pixels))
                random_indices = np.random.choice(
                    range(len(edge_pixels)), size=points_to_sample, replace=False
                )
                edge_points.extend(edge_pixels[random_indices])

        valid_points = [
            self.get_valid_point(image, [point[1], point[0]]) for point in edge_points
        ]

        return valid_points


class ExtremePointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting the extreme top, bottom, left, and right points
    within each region in the image.
    """

    def get_points(self, image, num_points):
        # Find the connected components (regions) in the image
        _, labels, _, _ = cv2.connectedComponentsWithStats(image, connectivity=4)

        extreme_points = []
        for region_id in range(1, np.max(labels) + 1):
            region_mask = np.uint8(labels == region_id) * 255
            region_white_pixels = np.argwhere(region_mask == 255)

            if len(region_white_pixels) > 0:
                top_point = region_white_pixels[region_white_pixels[:, 0].argmin()]
                bottom_point = region_white_pixels[region_white_pixels[:, 0].argmax()]
                left_point = region_white_pixels[region_white_pixels[:, 1].argmin()]
                right_point = region_white_pixels[region_white_pixels[:, 1].argmax()]

                valid_points = []
                for point in [top_point, bottom_point, left_point, right_point]:
                    valid_point = self.get_valid_point(
                        region_mask, [point[1], point[0]]
                    )
                    valid_points.append(valid_point)

                extreme_points.extend(valid_points)

        return extreme_points
