from pathlib import Path


class InputImagePathUtility:
    def __init__(self, input_image_dir: str):
        """
        Constructs an instance of the InputImagePathUtility. The input image directory
        must already exist before creating an instance of this class.

        Args:
            input_image_dir (str): Directory path where the input image set resides.

        Raises:
            FileNotFoundError: If input_image_dir does not exist.
        """
        self.input_image_dir = Path(input_image_dir).resolve()

        if not self.input_image_dir.exists():
            raise FileNotFoundError(f"No directory found at {self.input_image_dir}")

        self.input_image_file_name = self.input_image_dir.name

        self._create_folders()

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
    def input_with_ground_truth_overlay_dir(self) -> Path:
        """The directory for images that overlay ground truth masks on the input
        image."""
        return self.input_image_dir / "input_with_ground_truth_overlay"

    def get_input_image_path(self) -> Path:
        """
        Generates the file path for the input image.

        Returns:
            Path: Full path to the corresponding input image file.

        Raises:
            ValueError: If multiple input image files are found in the directory.
        """
        image_files = list(self.input_image_dir.glob("input_image*.png"))

        if not image_files:
            # Assume that the consumer still needs to generate the input image and will
            # use the returned path to do so. We return the input image directory path
            # plus the input image file name with .png extension so that the consumer
            # can generate the input
            return self.input_image_dir / f"{self.input_image_file_name}.png"
        elif len(image_files) > 1:
            raise ValueError("Multiple input image files found in the directory.")

        return image_files[0]

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

    def get_point_placement_path(self, index: int) -> Path:
        """
        Generates the file path for a specific point placement image.

        Args:
            index (int): The index of the point placement image.

        Returns:
            Path: Full path to the corresponding point placement file.
        """
        return self.point_prompts_dir / f"point_placement{index}.png"

    def get_point_prompts_csv_path(self) -> Path:
        """
        Generates the file path for the point prompts CSV file.

        Returns:
            Path: Full path to the point prompts CSV file.
        """
        return self.point_prompts_dir / "point_prompts.csv"

    def get_input_with_ground_truth_overlay_path(self, index: int) -> Path:
        """
        Generates the file path for the current input image with a specific ground truth
        mask overlay.

        Args:
            index (int): The index of the input image with ground truth mask overlay.

        Returns:
            Path: Full path to the corresponding input image with ground truth mask
            overlay file.
        """
        return (
            self.input_with_ground_truth_overlay_dir
            / f"input_with_ground_truth_overlay{index}.png"
        )

    def _create_folders(self):
        """
        Creates the necessary directory structure for the dataset if they do not already
        exist.
        """
        self.ground_truth_masks_dir.mkdir(parents=True, exist_ok=True)
        self.contour_layers_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.point_prompts_dir.mkdir(parents=True, exist_ok=True)
        self.input_with_ground_truth_overlay_dir.mkdir(parents=True, exist_ok=True)
