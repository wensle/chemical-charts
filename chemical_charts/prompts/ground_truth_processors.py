from abc import ABC, abstractmethod
import cv2
import numpy as np


class GroundTruthProcessor(ABC):
    """
    The base class for image processors.
    """

    @abstractmethod
    def process(self, image):
        pass


class ErosionProcessor(GroundTruthProcessor):
    """
    A processor that applies the erosion operation to the image.
    """

    def __init__(self, kernel_size=(3, 3), iterations=1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def process(self, image):
        kernel = np.ones(self.kernel_size, np.uint8)
        eroded_image = cv2.erode(image, kernel, iterations=self.iterations)
        return eroded_image
