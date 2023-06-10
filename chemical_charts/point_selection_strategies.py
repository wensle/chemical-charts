from abc import ABC, abstractmethod
import cv2
import numpy as np


class PointSelectionStrategy(ABC):
    """
    The base class of the strategy, defining how points should be selected.
    """

    @abstractmethod
    def get_points(self, image):
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
    A strategy for selecting random points from the image.
    """

    def get_points(self, image):
        white_pixels = np.argwhere(image == 255)
        random_points = []
        for _ in range(5):
            random_pixel = white_pixels[np.random.choice(white_pixels.shape[0])]
            random_point = [random_pixel[1], random_pixel[0]]  # Switch order to X,Y
            valid_random_point = self.get_valid_point(image, random_point)
            random_points.append(valid_random_point)

        return random_points


class CentroidPointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting the centroid point from the image.
    """

    def get_points(self, image):
        # Calculate the centroid of the image
        moments = cv2.moments(image)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        valid_centroid = self.get_valid_point(image, [centroid_x, centroid_y])

        return [valid_centroid]


class CombinedPointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting both random points and the centroid from the image.
    """

    def get_points(self, image):
        random_strategy = RandomPointSelectionStrategy()
        centroid_strategy = CentroidPointSelectionStrategy()

        return random_strategy.get_points(image) + centroid_strategy.get_points(image)


class EdgePointSelectionStrategy(PointSelectionStrategy):
    """
    A strategy for selecting points on the edges of white regions in the image.
    """

    def get_points(self, image):
        # Detect edges using the Canny edge detection method
        edges = cv2.Canny(image, 100, 200)

        # Find the coordinates of the edge points
        edge_points = np.argwhere(edges > 0)

        # Flip the order to X, Y and ensure the point is valid
        valid_edge_points = [
            self.get_valid_point(image, [point[1], point[0]]) for point in edge_points
        ]

        return valid_edge_points


# class ExtremePointSelectionStrategy(PointSelectionStrategy):
#     """
#     A strategy for selecting the extreme top, bottom, left, and right
#     points within the white regions in the image.
#     """

#     def get_points(self, image):
#         # Find the coordinates of the white pixels
#         white_pixels = np.argwhere(image == 255)

#         # Find the extreme points
#         top_point = white_pixels[white_pixels[:, 0].argmin()]
#         bottom_point = white_pixels[white_pixels[:, 0].argmax()]
#         left_point = white_pixels[white_pixels[:, 1].argmin()]
#         right_point = white_pixels[white_pixels[:, 1].argmax()]

#         # Flip the order to X, Y and ensure the point is valid
