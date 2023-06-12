import json
import logging
import os

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

from chemical_charts.contour_plot import TernaryContourPlot
from chemical_charts.contour_zones_processor import ContourZonesProcessor
from chemical_charts.ground_truth_prompt_pipeline import GroundTruthPromptPipeline
from chemical_charts.input_image_path_utility import InputImagePathUtility
from chemical_charts.prompts.ground_truth_processors import ErosionProcessor
from chemical_charts.prompts.point_selection_strategies import (
    CentroidPointSelectionStrategy,
    RandomPointSelectionStrategy,
)
from chemical_charts.thermo_calculator import ThermoCalculator
from chemical_charts.config import DOTENV_PATH


def main():
    load_dotenv(DOTENV_PATH)
    db_fname = "Al-Cu-Y.tdb"
    db_path = os.path.join(os.getenv("TDB_DIR"), db_fname)
    data_output_dir = os.getenv("DATA_OUTPUT_DIR")
    os.makedirs(data_output_dir, exist_ok=True)

    # Setup the logger for exceptions
    exceptions_logger = logging.getLogger("exceptions")
    exceptions_logger.setLevel(logging.ERROR)
    exceptions_log_file_path = os.path.join(data_output_dir, "exceptions.log")
    exceptions_handler = logging.FileHandler(exceptions_log_file_path)
    exceptions_formatter = logging.Formatter("%(message)s")
    exceptions_handler.setFormatter(exceptions_formatter)
    exceptions_logger.addHandler(exceptions_handler)

    components = ["AL", "CU", "Y", "VA"]
    pressures = [101325]  # Atmospheric pressure in Pa

    # We use the following temperatures for our contour plots because they represent key
    # points in the thermal behavior of the Al-Cu-Y system. 300 K approximates room
    # temperature conditions, 550 K and 750 K represent intermediate temperatures where
    # significant phase transformations might occur. 1000 K is a high temperature that
    # may induce additional phase changes, while 1300 K is above the typical melting
    # point of the system to observe the behavior of the liquid phase. The choice of
    # these specific temperatures should be guided by specific properties of the Al-Cu-Y
    # system and the research questions being investigated.
    temperatures = [300, 550, 750, 1000, 1300]  # in K
    output_properties = ["GM_MIX", "SM_MIX", "HM_MIX", "HM"]
    phases = ["LIQUID", "FCC_A1", "BCC_A2", "HCP_A3"]

    calculator = ThermoCalculator(
        db_path, components, pressures, temperatures, output_properties
    )

    for i, plot_data in enumerate(
        calculator.calculate_properties(phases=phases), start=1
    ):
        # Setup individual logger
        current_dir = os.path.join(data_output_dir, f"input_image{i}")
        os.makedirs(current_dir, exist_ok=True)
        path_util = InputImagePathUtility(current_dir)
        logger = logging.getLogger(f"input_image{i}")
        logger.setLevel(logging.ERROR)
        log_file_path = os.path.join(current_dir, f"input_image{i}.log")
        handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        try:
            xs = plot_data["xs"]
            ys = plot_data["ys"]
            zs = plot_data["zs"]

            coordinates, values = np.array([xs, ys, 1 - xs - ys]), zs

            title = (
                f'{plot_data["pressure"]} Pa - '
                f'{", ".join(plot_data["composition"])} - '
                f'{plot_data["phase"]} - '
                f'{plot_data["output_property"]} - '
                f'{plot_data["temperature"]:.2f} K'
            )

            # Create plots
            plotter = TernaryContourPlot(
                coordinates=coordinates, values=values, plot_title=title
            )

            line_traces = plotter.create_line_trace_dicts()
            line_layout = plotter.generate_layout()
            line_fig = go.Figure(data=line_traces, layout=line_layout)

            input_img_path = path_util.get_input_image_path()
            line_fig.write_image(input_img_path, width=1024, height=1024)

            fill_traces = plotter.create_contour_fill_traces()
            fill_layout = plotter.generate_layout(
                show_lines=False,
                show_ticks=False,
                show_title=False,
                show_pole_labels=False,
                paper_bgcolor="rgb(0,0,0)",
            )

            for i, fill_trace in enumerate(fill_traces, start=1):
                fig = go.Figure(data=fill_trace, layout=fill_layout)
                # fig.show()
                contour_layer_path = path_util.get_contour_layer_path(i)
                fig.write_image(contour_layer_path, width=1024, height=1024, scale=5)

            # Create the ground truth prompt pipeline
            pipeline = GroundTruthPromptPipeline()

            # Add the erosion processor to the pipeline
            erosion_processor = ErosionProcessor(kernel_size=(5, 5), iterations=1)
            pipeline.add_processor(erosion_processor)

            # Add the point selection strategies to the pipeline
            pipeline.add_strategy(RandomPointSelectionStrategy())
            pipeline.add_strategy(CentroidPointSelectionStrategy())

            # Process the images
            processor = ContourZonesProcessor(
                path_util,
                pipeline,
                resolution_scaling_factor=5,
                num_prompt_points=30,
            )  # Create an instance of ContourZonesProcessor

            # Save ground truth masks to the output directory
            processor.save_ground_truth_masks()
            # Compute and save sample prompts for all ground truth masks
            processor.save_point_prompts()
            # Save images with contour zones overlaid on top of the input image
            processor.save_overlay_images()

        except Exception:
            plot_data.pop("xs")
            plot_data.pop("ys")
            plot_data.pop("zs")
            logger.error(
                f"Exception occurred, data: {json.dumps(plot_data, indent=4)}",
                exc_info=True,
            )
            exceptions_logger.error(
                f"An exception occurred while creating: input_image{i}"
            )


if __name__ == "__main__":
    main()
