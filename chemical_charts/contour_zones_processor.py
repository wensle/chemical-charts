import csv
from typing import List

import cv2
import numpy as np

from chemical_charts.ground_truth_prompt_pipeline import GroundTruthPromptPipeline
from chemical_charts.input_image_path_utility import InputImagePathUtility


class ContourZonesProcessor:
    def __init__(
        self,
        input_image_path_utility: InputImagePathUtility,
        prompt_pipeline: GroundTruthPromptPipeline,
        num_prompt_points: int = 10,
        resolution_scaling_factor=5,
    ) -> None:
        """
        Initialize the processor with the input image path utility, processing
        parameters, and sampling strategy.
        """
        self.path_util: InputImagePathUtility = input_image_path_utility
        self.prompt_pipeline: GroundTruthPromptPipeline = prompt_pipeline
        self.num_prompt_points: int = num_prompt_points
        self.resolution_scaling_factor: int = resolution_scaling_factor
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

                processed_mask = self.prompt_pipeline.process(ground_truth_mask)

                sample_prompts = self.prompt_pipeline.execute_strategies(
                    ground_truth_mask=processed_mask,
                    num_points=self.num_prompt_points,
                )
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

    # def _process_contour_layers(self) -> List[np.ndarray]:
    #     """
    #     Processes contour layers to generate masks. Overlapping layers are processed
    #     into a single mask representing the change, while non-overlapping layers each
    #     generate an individual mask. The first mask layer is always preserved.

    #     Returns:
    #         masks: A list of processed mask layers in numpy.ndarray format.
    #     """

    #     # Initialize list to store masks
    #     masks = []

    #     # Loop through each contour layer
    #     for i in range(len(self.contour_layers)):
    #         # Get the current and previous contour layers
    #         contour_layer1 = self.contour_layers[i - 1]
    #         contour_layer2 = self.contour_layers[i]

    #         is_last_layer = i == len(self.contour_layers) - 1
    #         is_2nd_to_last_layer = i == len(self.contour_layers) - 2

    #         # Check if current and previous layers overlap
    #         if self.get_overlap(contour_layer1, contour_layer2):
    #             # If overlap, calculate the change mask
    #             change_mask = self.get_change_mask(contour_layer1, contour_layer2)
    #             # If change mask is not all zeros, or it's the last contour layer, add
    #             # it to masks
    #             is_empty_mask = np.count_nonzero(change_mask) == 0
    #             if not is_empty_mask or is_last_layer:
    #                 masks.append(change_mask)
    #         else:
    #             if is_last_layer:
    #                 # If it's the last contour layer and no overlap with previous, add
    #                 # it to masks
    #                 masks.append(contour_layer2)
    #             else:
    #                 # If no overlap, add both layers as separate masks
    #                 masks.extend([contour_layer1, contour_layer2])

    #     return masks

    def _process_contour_layers(self) -> List[np.ndarray]:
        """
        Processes contour layers to generate masks. Overlapping layers are processed
        into a single mask representing the change, while non-overlapping layers each
        generate an individual mask. The first mask layer is always preserved.

        Returns:
            masks: A list of processed mask layers in numpy.ndarray format.
        """

        filtered_layer = []

        # Loop through each contour layer
        for contour_layer in self.contour_layers:
            # Check if the contour layer is already encountered
            if not any(
                np.array_equal(contour_layer, layer) for layer in filtered_layer
            ):
                # Add the contour layer to the encountered list
                filtered_layer.append(contour_layer)

        # Initialize list to store masks
        masks = []

        # Placeholder for the change mask between overlapping layers
        change_mask = None

        # Loop through each contour layer
        for i in range(len(filtered_layer)):
            # Get the current and previous contour layers
            contour_layer1 = filtered_layer[i - 1] if i > 0 else None
            contour_layer2 = filtered_layer[i]

            is_last_layer = i == len(filtered_layer) - 1

            if change_mask is not None:
                # If a change mask from previous overlapping layers exists
                if not self.get_overlap(change_mask, contour_layer2):
                    # If current layer doesn't overlap with the change mask,
                    # append the change mask to the list and reset it
                    masks.append(change_mask)
                    change_mask = None

            # Check if current and previous layers overlap
            if i == 0:
                pass  # Skip the first layer
            elif i > 0 and self.get_overlap(contour_layer1, contour_layer2):
                # If overlap, calculate the change mask
                change_mask = self.get_change_mask(contour_layer1, contour_layer2)
                if np.count_nonzero(change_mask) == 0:
                    # If change mask is all zeros, reset it
                    change_mask = None
            else:
                # If no overlap with previous
                if change_mask is None:
                    # If no change mask exists, append the previous layer
                    masks.append(contour_layer1)
                masks.append(contour_layer2)

            if is_last_layer:
                if change_mask is not None:
                    # If it's the last contour layer and a change mask exists, append it
                    masks.append(change_mask)
                    masks.append(contour_layer2)

        return masks

    def _save_masks(self, masks: List[np.ndarray]) -> None:
        """
        Saves given masks as image files.

        Args:
            masks: List of masks in numpy.ndarray format to be saved.
        """
        mask_index = 1
        # Loop through each mask
        for mask in masks:
            # If mask is not all zeros (empty), save it as an image file
            if np.count_nonzero(mask) != 0:
                mask_path = self.path_util.get_ground_truth_mask_path(mask_index)
                cv2.imwrite(str(mask_path), mask)
                mask_index += 1

    def save_ground_truth_masks(self) -> None:
        """
        Processes the contour layers into masks, checks for overlap between the first
        and last masks, calculates a change mask if overlap exists, and finally saves
        all masks as image files.
        """
        # Process contour layers into masks
        masks = self._process_contour_layers()

        # After all masks are processed
        if len(masks) > 1:
            # Get the first and last masks
            first_mask = masks[0]
            last_mask = masks[-1]

            # Check if the last mask is a region within the first mask (i.e. they
            # overlap)
            if self.get_overlap(first_mask, last_mask):
                # Calculate the change mask and replace the first mask with the change
                # mask
                change_mask = self.get_change_mask(first_mask, last_mask)
                masks[0] = change_mask

        # Now save the masks
        self._save_masks(masks)


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
