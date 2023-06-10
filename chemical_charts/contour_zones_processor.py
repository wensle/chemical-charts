import csv

import cv2
import numpy as np

from chemical_charts.input_image_path_utility import InputImagePathUtility
from chemical_charts.point_selection_strategies import CombinedPointSelectionStrategy


class ContourZonesProcessor:
    def __init__(
        self,
        input_image_path_utility: InputImagePathUtility,
        resolution_scaling_factor=20,
        sampling_strategy=CombinedPointSelectionStrategy(),
    ) -> None:
        """
        Initialize the processor with the input image path utility, processing
        parameters, and sampling strategy.
        """
        self.path_util: InputImagePathUtility = input_image_path_utility
        self.resolution_scaling_factor: int = resolution_scaling_factor
        self.sampling_strategy: CombinedPointSelectionStrategy = sampling_strategy
        self.contour_layers = self._load_and_preprocess_contour_layers()

    def _load_and_preprocess_contour_layers(self):
        """
        Load and preprocess all contour layers from the directory.
        Returns a list of preprocessed contour layers.
        """
        contour_layers_dir = self.path_util.contour_layers_dir
        contour_layer_paths = sorted(contour_layers_dir.glob("contour_layer*.png"))

        contour_layers = []
        for path in contour_layer_paths:
            contour_layer: np.ndarray[int, np.dtype[np.generic]] = cv2.imread(
                str(path), cv2.IMREAD_GRAYSCALE
            )
            contour_layer_small: np.ndarray[
                int, np.dtype[np.generic]
            ] = self._downscale_contour_layer(contour_layer)
            contour_layers.append(contour_layer_small)

        return contour_layers

    def _downscale_contour_layer(self, contour_layer):
        """
        Downscale a contour layer by a given factor as an anti-aliasing mitigation
        measure.
        """
        small_size = (
            contour_layer.shape[1] // self.resolution_scaling_factor,
            contour_layer.shape[0] // self.resolution_scaling_factor,
        )
        contour_layer_small = cv2.resize(
            contour_layer, small_size, interpolation=cv2.INTER_NEAREST
        )
        return contour_layer_small

    def get_change_mask(self, contour_layer1, contour_layer2):
        """
        Calculate and return the change mask between two contour layers.
        """
        change_mask = cv2.subtract(contour_layer1, contour_layer2)
        return change_mask

    def get_overlap(self, contour_layer1, contour_layer2):
        """
        Check and return if there's overlap between two contour layers.
        """
        overlap = cv2.bitwise_and(contour_layer1, contour_layer2)
        return cv2.countNonZero(overlap) > 0

    def get_sample_point_prompts(self, contour_zone_mask):
        """
        Generate and return prompts from a contour zone mask based on the current
        sampling strategy.
        """
        return self.sampling_strategy.get_points(contour_zone_mask)

    def save_point_prompts(self):
        """
        Compute and save sample prompts for all ground truth masks to a CSV file.
        """
        csv_file_path = self.path_util.get_point_prompts_csv_path()

        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Ground Truth Mask File", "Prompt_X", "Prompt_Y"])

            i = 1  # Initialize ground truth mask index
            while True:
                # Load ground truth mask
                mask_path = self.path_util.get_ground_truth_mask_path(i)
                if not mask_path.exists():
                    break  # Exit the loop if the mask file does not exist
                ground_truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                sample_prompts = self.get_sample_point_prompts(ground_truth_mask)
                for prompt in sample_prompts:
                    csv_writer.writerow(
                        [f"ground_truth_mask_{i}.png", prompt[1], prompt[0]]
                    )

                i += 1  # Increment mask index

    def save_overlay_images(self):
        """
        Save images where each ground truth mask is overlaid on top of the input image.
        Also plots the point prompts on the overlay.
        """
        input_image_path = self.path_util.get_input_image_path()
        input_image = cv2.imread(str(input_image_path))

        # Define overlay color (BGR format)
        overlay_color = [0, 0, 255]  # Red in BGR
        # Define point color (BGR format)
        point_color = [0, 255, 0]  # Green in BGR

        # Read the point prompts from CSV
        csv_file_path = self.path_util.get_point_prompts_csv_path()
        with open(csv_file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row
            point_prompts = [
                (row[0], int(row[2]), int(row[1])) for row in csv_reader
            ]  # Load the rest

        i = 1  # Initialize ground truth mask index
        while True:
            # Load ground truth mask
            mask_path = self.path_util.get_ground_truth_mask_path(i)
            if not mask_path.exists():
                break  # Exit the loop if the mask file does not exist
            ground_truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Filter point prompts for the current mask
            current_point_prompts = [
                point
                for point in point_prompts
                if point[0] == f"ground_truth_mask_{i}.png"
            ]

            # Convert single channel image to 3 channel image
            mask_3channel = cv2.cvtColor(ground_truth_mask, cv2.COLOR_GRAY2BGR)

            # Assign the desired color to each pixel of the mask
            mask_3channel[
                np.where((mask_3channel == [255, 255, 255]).all(axis=2))
            ] = overlay_color

            # Draw the point prompts on the mask
            for point in current_point_prompts:
                # Convert string coordinates to integers
                x, y = point[1:]
                cv2.circle(
                    mask_3channel, (x, y), radius=3, color=point_color, thickness=-1
                )

            overlay = cv2.addWeighted(input_image, 0.6, mask_3channel, 0.4, 0)
            overlay_path = self.path_util.get_input_with_ground_truth_overlay_path(i)
            cv2.imwrite(str(overlay_path), overlay)

            i += 1  # Increment mask index

    def save_ground_truth_masks(self):
        """
        Save ground truth masks to the output directory.
        """
        if len(self.contour_layers) < 2:
            raise ValueError("Insufficient contour layers to save ground truth masks.")

        mask_index = 1
        for i in range(1, len(self.contour_layers)):
            contour_layer1 = self.contour_layers[i - 1]
            contour_layer2 = self.contour_layers[i]

            if not all(
                isinstance(layer, np.ndarray)
                for layer in (contour_layer1, contour_layer2)
            ):
                raise TypeError("Contour layers must be numpy arrays.")

            if self.get_overlap(contour_layer1, contour_layer2):
                change_mask = self.get_change_mask(contour_layer1, contour_layer2)
                mask_path = self.path_util.get_ground_truth_mask_path(mask_index)

                try:
                    cv2.imwrite(str(mask_path), change_mask)
                except Exception as e:
                    raise RuntimeError(
                        "Error occurred while saving change mask: " + str(e)
                    )
                mask_index += 1
            else:
                masks = [contour_layer1, contour_layer2]

                for mask in masks:
                    mask_path = self.path_util.get_ground_truth_mask_path(mask_index)
                    try:
                        cv2.imwrite(str(mask_path), mask)
                    except Exception as e:
                        raise RuntimeError(
                            "Error occurred while saving contour layer mask: " + str(e)
                        )
                    mask_index += 1


if __name__ == "__main__":
    input_image_path_utility = InputImagePathUtility(
        "/home/wensley/chemical-charts/input_image1",
    )  # Create an instance of InputImagePathUtility
    processor = ContourZonesProcessor(
        input_image_path_utility,
        resolution_scaling_factor=5,
    )  # Create an instance of ContourZonesProcessor

    # Compute and save sample prompts for all ground truth masks
    processor.save_point_prompts()
    # Save ground truth masks to the output directory
    processor.save_ground_truth_masks()
    # Save images with contour zones overlaid on top of the input image
    processor.save_overlay_images()
