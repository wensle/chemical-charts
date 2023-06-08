from pathlib import Path

import scipy as sp


class InputImageSetBuilder:
    def __init__(self, dataset_dir: str):
        """
        Constructs an instance of the InputImageSetBuilder. Prepares for the creation of
        new an input image set by creating necessary directory structures.

        Args:
            dataset_dir (str): Root directory path where the new input image set will
            be built.
        """
        self.dataset_dir = Path(dataset_dir).resolve()
        self.image_count = self._get_next_image_count()

        self._create_folders()

    @property
    def input_image_dir(self) -> Path:
        """The directory for the current input image set."""
        return self.dataset_dir / f"input_image{self.image_count}"

    @property
    def ground_truth_masks_dir(self) -> Path:
        """The directory for ground truth masks related to the current input image
        set."""
        return self.input_image_dir / "ground_truth_masks"

    @property
    def contour_layers_dir(self) -> Path:
        """The directory for the contour layers related to the current input image
        set."""
        return self.input_image_dir / "contour_layers"

    @property
    def prompts_dir(self) -> Path:
        """The directory for prompts related to the current input image set."""
        return self.input_image_dir / "prompts"

    @property
    def point_prompts_dir(self) -> Path:
        """The directory for point prompts within the prompts directory."""
        return self.prompts_dir / "point_prompts"

    @property
    def input_with_contour_overlay_dir(self) -> Path:
        """The directory for images that overlay contour zones on the input image."""
        return self.input_image_dir / "input_with_contour_overlay"

    def get_ground_truth_mask_path(self, index: int) -> Path:
        """
        Generates the file path for a specific ground truth mask.

        Args:
            index (int): The index of the ground truth mask.

        Returns:
            Path: Full path to the corresponding ground truth mask file.
        """
        return self.ground_truth_masks_dir / f"ground_truth_mask{index}.png"

    def get_contour_layer_path(self, index: int) -> Path:
        """
        Generates the file path for a specific contour layer.

        Args:
            index (int): The index of the contour layer.

        Returns:
            Path: Full path to the corresponding contour layer file.
        """
        return self.contour_layers_dir / f"contour_layer{index}.png"

    def get_prompt_placement_path(self, index: int) -> Path:
        """
        Generates the file path for a specific prompt placement image.

        Args:
            index (int): The index of the prompt placement image.

        Returns:
            Path: Full path to the corresponding prompt placement file.
        """
        return self.point_prompts_dir / f"prompts_placement{index}.png"

    def _get_next_image_count(self) -> int:
        """
        Finds the next image index by examining existing directories in the dataset. The
        method expects the directory names to follow the format "input_imageX", where X
        represents the image index. Any directory name that does not adhere to this
        format will be ignored.

        Returns:
            int: The next image index to be used.
        """
        existing_dirs = [
            dir_.name for dir_ in self.dataset_dir.iterdir() if dir_.is_dir()
        ]
        image_counts = []
        for name in existing_dirs:
            split_name = name.split("input_image")
            if len(split_name) == 2 and split_name[1].isdigit():
                image_counts.append(int(split_name[1]))
        return max(image_counts, default=0) + 1

    def _create_folders(self):
        """
        Creates the necessary directory structure for the dataset, if they do not
        already exist.
        """
        self.input_image_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_masks_dir.mkdir(parents=True, exist_ok=True)
        self.contour_layers_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.point_prompts_dir.mkdir(parents=True, exist_ok=True)
        self.input_with_contour_overlay_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    dataset_dir = "/home/wensley/chemical-charts/test_synthetic_data"
    builder = InputImageSetBuilder(dataset_dir)

    # Print the directory paths
    print(f"Input Image Directory: {builder.input_image_dir}")
    print(f"Ground Truth Masks Directory: {builder.ground_truth_masks_dir}")
    print(f"Contour Layers Directory: {builder.contour_layers_dir}")
    print(f"Prompts Directory: {builder.prompts_dir}")
    print(f"Point Prompts Directory: {builder.point_prompts_dir}")
    print(
        f"Input with Contour Overlay Directory: "
        f"{builder.input_with_contour_overlay_dir}"
    )

    # Generate file paths
    ground_truth_mask_path = builder.get_ground_truth_mask_path(1)
    contour_layer_path = builder.get_contour_layer_path(1)
    prompt_placement_path = builder.get_prompt_placement_path(1)

    print(f"Ground Truth Mask Path: {ground_truth_mask_path}")
    print(f"Contour Layer Path: {contour_layer_path}")
    print(f"Prompt Placement Path: {prompt_placement_path}")
