import json
import logging
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

from chemical_charts.contour_plot import TernaryContourPlot
from chemical_charts.thermo_calculator import ThermoCalculator


def optimal_contours(xs, ys, zs):
    std_dev = np.std(zs)

    if std_dev > 5:
        # If the standard deviation is greater than 10, return 10
        return 5
    elif std_dev < 5:
        # If the standard deviation is less than 5, return 5
        return 5
    else:
        # If the standard deviation falls between 5 and 20, return it as an integer
        return int(std_dev)


def main():
    logger = logging.getLogger("chemical_charts")
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler("error.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    load_dotenv()
    db_fname = "Al-Cu-Y.tdb"
    db_path = os.path.join(os.getenv("TDB_DIR"), db_fname)

    # use the class
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
    temperatures = [300, 550, 750, 1000, 1300]  # K
    output_properties = ["GM_MIX", "SM_MIX", "HM_MIX", "HM"]
    calculator = ThermoCalculator(
        db_path, components, pressures, temperatures, output_properties
    )

    contour_dir = Path(os.getcwd()) / "synthetic_data" / "contour_plots"
    mask_dir = Path(os.getcwd()) / "synthetic_data" / "mask_plots"

    contour_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for plot_data in calculator.calculate_properties():
        xs = plot_data["xs"]
        ys = plot_data["ys"]
        zs = plot_data["zs"]

        coordinates, values = np.array([xs, ys, 1 - xs - ys]), zs

        try:
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

            line_traces = plotter.create_contour_line_traces()
            line_layout = plotter.generate_layout()
            line_fig = go.Figure(data=line_traces, layout=line_layout)

            # Create mask
            fill_traces = plotter.create_contour_fill_traces()
            fill_layout = plotter.generate_layout(
                show_lines=False,
                show_ticks=False,
                show_title=False,
                show_pole_labels=False,
            )

            fill_fig = go.Figure(data=fill_traces, layout=fill_layout)

            line_fig.write_image(str(contour_dir / f"{title}.png"))
            fill_fig.write_image(str(mask_dir / f"{title}.png"))

        except Exception as e:
            plot_data.pop("xs")
            plot_data.pop("ys")
            plot_data.pop("zs")
            logger.error(
                f"Exception occurred, data: {json.dumps(plot_data, indent=4)}",
                exc_info=True,
            )


if __name__ == "__main__":
    main()
