from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory._ternary_contour import (
    _compute_grid,
    _prepare_barycentric_coord,
    _colors,
    _extract_contours,
    _add_outer_contour,
    _transform_barycentric_cartesian,
    _ilr_inverse,
    _is_invalid_contour,
)

from chemical_charts.input_image_path_utility import InputImagePathUtility


class TernaryContourPlot:
    """A class used to generate a ternary contour line plot via Plotly.

    Compared to the 'create_ternary_contour' factory function in Plotly, this class
    provides enhanced flexibility and additional customization options.

    Attributes:
        coordinates (np.ndarray): The coordinates in the ternary space.
        values (np.ndarray): The function values at the coordinates.
        pole_labels (List[str]): Labels for the poles of the ternary plot.
        plot_title (str): The title of the plot.
        plot_width (int): The width of the plot.
        plot_height (int): The height of the plot.
        num_contours (int): The number of contour levels.
        line_color (str): The color of the contour lines.
        line_width (int): The width of the contour lines.
        axis_ticks (str): Position of the axis ticks ("inside", "outside", or "both").
        interp_mode (str): The interpolation mode for grid computation.
        barycentric_coordinates (np.ndarray): Barycentric coordinates derived from the
            input coordinates.
        min_value (float): The minimum value among the input values.
        max_value (float): The maximum value among the input values.
        grid_values (np.ndarray): Values on the grid derived from input values and
            coordinates.
        grid_coord_x (np.ndarray): x-coordinates on the grid derived from input
            coordinates.
        grid_coord_y (np.ndarray): y-coordinates on the grid derived from input
            coordinates.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        plot_title: str,
        plot_width: int = 500,
        plot_height: int = 500,
        plot_colorscale: str = "Rainbow",
        plot_bgcolor: str = "rgb(255,255,255)",
        pole_labels: list = ["a", "b", "c"],
        num_contours: int = 5,
        line_color: str = "rgb(0,0,0)",
        line_width: int = 1,
        axis_ticks: str = "outside",
        interp_mode: str = "cartesian",
    ) -> None:
        """Initializes the TernaryContourPlot with the given parameters."""
        self.coordinates = coordinates
        self.values = values
        self.pole_labels = pole_labels
        self.plot_title = plot_title
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.plot_colorscale = plot_colorscale
        self.plot_bgcolor = plot_bgcolor
        self.num_contours = num_contours
        self.line_color = line_color
        self.line_width = line_width
        self.axis_ticks = axis_ticks
        self.interp_mode = interp_mode

        self.barycentric_coordinates = _prepare_barycentric_coord(coordinates)
        self.min_value, self.max_value = values.min(), values.max()
        self.grid_coord_z, self.grid_coord_x, self.grid_coord_y = _compute_grid(
            self.barycentric_coordinates, values, interp_mode=self.interp_mode
        )

        # Initialize lists for the coordinates of the contour lines
        self.a_coords = []
        self.b_coords = []
        self.c_coords = []

        # Initialize lists for the coordinates in
        self.x_coords = []
        self.y_coords = []

        # Initialize list for the values in the contour lines
        self.z_coords = []

        self._contour_colors: list

        self.generate_contour_data()

    def get_contour_color_by_index(self, index: int):
        """Retrieves the color of a contour based on its index.

        Args:
            index (int): The index of the contour.

        Returns:
            list: The color of the contour at the given index.

        Raises:
            IndexError: If the index is out of the range of available contours.
        """
        if index < 0 or index >= len(self._contour_colors):
            raise IndexError("Contour index out of range.")
        return self._contour_colors[index]

    def create_line_trace_dicts(self) -> List[go.Scatterternary]:
        line_trace_dicts = []
        for i, (a, b, c) in enumerate(zip(self.a_coords, self.b_coords, self.c_coords)):
            line_trace_dicts.append(
                self._create_trace_dict(a, b, c, i, coloring="lines")
            )
        line_trace_dicts.append(self._create_border_dict(coloring="lines"))
        return line_trace_dicts

    def create_contour_fill_traces(self) -> List[go.Scatterternary]:
        fill_trace_dicts = []
        for i, (a, b, c) in enumerate(zip(self.a_coords, self.b_coords, self.c_coords)):
            fill_trace_dicts.append(self._create_trace_dict(a, b, c, i))
        return fill_trace_dicts

    def _create_border_dict(self, coloring=None):
        if coloring == "lines":
            line_color = self.line_color
        elif coloring is None:
            line_color = "white"
        else:
            raise ValueError("Invalid coloring mode. Must be 'lines' or None.")

        # Set the line width to an arbitrary value that looks good. The line width of
        # the contour lines is 1. We can't use a line width of 1 because it will not be
        # visible. Therefore, we need to use a larger line width.
        BORDER_LINE_WIDTH = 18

        return dict(
            type="scatterternary",
            a=[1, 0, 0, 1],
            b=[0, 1, 0, 0],
            c=[0, 0, 1, 0],
            line=dict(
                color=line_color,
                width=BORDER_LINE_WIDTH,
                shape="linear",
            ),
            fill="none",
            mode="lines",
            showlegend=False,
        )

    def generate_layout(
        self,
        show_title: bool = True,
        show_pole_labels: bool = True,
        show_lines: bool = True,
        show_ticks: bool = True,
        paper_bgcolor: str = "rgb(255,255,255)",
    ) -> Dict[str, Any]:
        """Generates a layout dictionary for the ternary plot.

        Args:
            include_title (bool): Flag to include plot title. Default is True.
            include_aaxis_title (bool): Flag to include 'a' axis title. Default is True.
            include_baxis_title (bool): Flag to include 'b' axis title. Default is True.
            include_caxis_title (bool): Flag to include 'c' axis title. Default is True.

        Returns:
            dict: A dictionary specifying the layout of the ternary plot.
        """

        def generate_axis_dicts() -> tuple[dict]:
            dicts = ()
            for pole_label in self.pole_labels:
                d = dict(
                    title=dict(text=pole_label) if show_pole_labels else "",
                    min=0.01,
                    linewidth=self.line_width if show_lines else 0,
                    ticks=self.axis_ticks if show_ticks else "",
                    showgrid=False,
                )
                # Ugly way to remove tick values, but it works. It uses the update
                # operator for dictionaries which was added in Python 3.9.
                if not show_ticks:
                    d |= dict(tickvals=[], tickmode="array")
                dicts += (d,)
            return dicts

        aaxis, baxis, caxis = generate_axis_dicts()

        return go.Layout(
            dict(
                title=self.plot_title if show_title else "",
                width=self.plot_width,
                height=self.plot_height,
                ternary=dict(
                    sum=1,
                    aaxis=aaxis,
                    baxis=baxis,
                    caxis=caxis,
                    bgcolor=paper_bgcolor,
                ),
                showlegend=False,
                paper_bgcolor=paper_bgcolor,
            )
        )

    def generate_contour_data(self):
        """
        This function generates a set of contour data from a provided grid of
        z-coordinates.

        The grid of z-coordinates represents a 3-dimensional surface in
        the form of a 2D array, where each element in the array corresponds to a point
        (x, y) on the surface, and the value of the element represents the 'z' value or
        the height at that point.

        This method extracts and orders the contours from a given grid based on their
        area, transforming them into Cartesian coordinates if necessary. Additionally,
        it determines the colors of the contours based on their relative value within
        the data range.

        Note: Assumes the presence of a grid of z-coordinates, along with relevant
        parameters for contours and color mapping.
        """

        # Define color mapping based on number of contours. Colors are taken from the
        # midpoint of the colormap, excluding extrema.
        color_scale = _colors(self.num_contours + 2, self.plot_colorscale)

        # Define contour values between the minimum and maximum, excluding extrema. For
        # a binary array [0, 1], the value of the contour for number_of_contours=1 is
        # 0.5.
        contour_values = np.linspace(
            self.min_value, self.max_value, self.num_contours + 2
        )

        # Exclude the color scale extrema for contours
        color_min, color_max = color_scale[0], color_scale[-1]
        color_scale = color_scale[1:-1]
        contour_values = contour_values[1:-1]

        # Extract all contours from the z-coordinate grid
        contours, contour_values, contour_areas, contour_colors = _extract_contours(
            self.grid_coord_z, contour_values, color_scale
        )

        # Order the contours by decreasing area
        ordered_indices = np.argsort(contour_areas)[::-1]

        # Add an outer contour to fill gaps outside of computed contours
        (
            contours,
            contour_values,
            contour_areas,
            contour_colors,
            _,
        ) = _add_outer_contour(
            contours,
            contour_values,
            contour_areas,
            contour_colors,
            contour_values,
            contour_values[ordered_indices[0]],
            self.min_value,
            self.max_value,
            color_scale,
            color_min,
            color_max,
        )

        # Adjust the ordered_indices to account for the addition of the outer contour.
        # The outer contour is always placed first (at index 0) due to its role in
        # filling gaps between computed contours. The remaining indices are incremented
        # by one to correctly align with the updated contour list (which now includes
        # the outer contour at the start).
        ordered_indices = np.concatenate(([0], ordered_indices + 1))

        # Compute transformation from barycentric coordinates to Cartesian
        _, invM = _transform_barycentric_cartesian()

        # Compute the grid step sizes in x and y directions
        dx = (
            self.grid_coord_x.max() - self.grid_coord_x.min()
        ) / self.grid_coord_x.size
        dy = (
            self.grid_coord_y.max() - self.grid_coord_y.min()
        ) / self.grid_coord_y.size

        # Iterate over contours in order of decreasing area
        for index in ordered_indices:
            # Extract the x and y coordinates of the contour
            y_contour, x_contour = contours[index].T
            z_contour = contour_values[index]

            # Depending on the interpolation mode, transform the contour coordinates
            # back to barycentric coordinates.
            if self.interp_mode == "cartesian":
                barycentric_coords = np.dot(
                    invM,
                    np.stack(
                        (dx * x_contour, dy * y_contour, np.ones(x_contour.shape))
                    ),
                )
            elif self.interp_mode == "ilr":
                barycentric_coords = _ilr_inverse(
                    np.stack(
                        (
                            dx * x_contour + self.grid_coord_x.min(),
                            dy * y_contour + self.grid_coord_y.min(),
                        )
                    )
                )

            # If the contour is the outer triangle, set its barycentric coordinates
            # manually. For other contours, use the calculated barycentric coordinates.
            if index == 0:
                a = np.array([1, 0, 0])
                b = np.array([0, 1, 0])
                c = np.array([0, 0, 1])
            else:
                a, b, c = barycentric_coords

            # If the contour is invalid, skip it.
            if _is_invalid_contour(x_contour, y_contour):
                continue

            # Save the contour colors for later use.
            self._contour_colors = contour_colors

            # Append the calculated coordinates and contour values to the class
            # variables.
            self.a_coords.append(a)
            self.b_coords.append(b)
            self.c_coords.append(c)
            self.x_coords.append(x_contour)
            self.y_coords.append(y_contour)
            self.z_coords.append(z_contour)

    def _create_trace_dict(self, a, b, c, contour_index, coloring=None):
        if coloring == "lines":
            line_color = self.line_color
            fillcolor = self.plot_bgcolor
            fill = "none"
            mode = "lines"
            width = self.line_width
            shape = "spline"
        elif coloring is None:
            line_color = self.plot_bgcolor
            fillcolor = "rgb(255,255,255)"
            fill = "toself"
            mode = "none"
            width = 1
            shape = "linear"
        else:
            raise ValueError("Invalid coloring mode. Must be 'lines' or None.")

        trace_dict = dict(
            type="scatterternary",
            a=a,
            b=b,
            c=c,
            mode=mode,
            line=dict(color=line_color, shape=shape, width=width),
            fill=fill,
            fillcolor=fillcolor,
            showlegend=False,
            hoverinfo="skip",
        )

        return trace_dict


if __name__ == "__main__":
    import os

    Al = np.array(
        [0.0, 0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 1.0]
    )
    Cu = np.array(
        [0.0, 1.0 / 3, 2.0 / 3, 1.0, 0.0, 1.0 / 3, 2.0 / 3, 0.0, 1.0 / 3, 0.0]
    )
    Y = 1 - Al - Cu
    zs = (Al - 0.01) * Cu * (Al - 0.52) * (Cu - 0.48) * (Y - 1) ** 2

    coordinates, values = np.array([Al, Cu, Y]), zs

    # Create a ContourPlot instance
    plotter = TernaryContourPlot(
        coordinates=coordinates, values=values, plot_title="Hello World!"
    )

    os.makedirs("input_image1", exist_ok=True)
    path_util = InputImagePathUtility("input_image1")

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
