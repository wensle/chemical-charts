import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from chemical_charts.contour_plot import ContourPlot
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
    temperatures = np.linspace(573.15, 10573.15, 101)
    output_properties = ["GM_MIX", "SM_MIX", "HM_MIX"]
    calculator = ThermoCalculator(
        db_path, components, pressures, temperatures, output_properties
    )

    contour_dir = Path(os.getcwd()) / "synthetic_data" / "contour_plots"
    mask_dir = Path(os.getcwd()) / "synthetic_data" / "mask_plots"

    contour_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for plot_data in calculator.calculate_properties():
        # Create a ContourPlot instance
        contour_plot = ContourPlot()

        # Prepare data
        coordinates, values = contour_plot.prepare_data(
            plot_data["xs"], plot_data["ys"], plot_data["zs"]
        )

        try:
            # Create contour plot
            title = (
                f'{plot_data["pressure"]} Pa - '
                f'{", ".join(plot_data["composition"])} - '
                f'{plot_data["phase"]} - '
                f'{plot_data["output_property"]} - '
                f'{plot_data["temperature"]:.2f} K'
            )

            # Use the function
            num_contours = optimal_contours(
                plot_data["xs"], plot_data["ys"], plot_data["zs"]
            )

            contour_plot_fig = contour_plot.create_contour_plot(
                coordinates, values, title, num_contours
            )

            # Create mask
            mask_fig = contour_plot.create_mask(coordinates, values, num_contours)

            contour_plot_fig.write_image(str(contour_dir / f"{title}.png"))
            mask_fig.write_image(str(mask_dir / f"{title}.png"))

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
