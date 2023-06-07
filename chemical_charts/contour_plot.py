from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory._ternary_contour import (
    _compute_grid,
    _contour_trace,
    _prepare_barycentric_coord,
    create_ternary_contour,
)


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
        pole_labels: list = ["a", "b", "c"],
        num_contours: int = 5,
        line_color: str = "rgb(0,0,0)",
        line_width: int = 2,
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
        self.num_contours = num_contours
        self.line_color = line_color
        self.line_width = line_width
        self.axis_ticks = axis_ticks
        self.interp_mode = interp_mode
        self.barycentric_coordinates = _prepare_barycentric_coord(coordinates)
        self.min_value, self.max_value = values.min(), values.max()
        self.grid_values, self.grid_coord_x, self.grid_coord_y = _compute_grid(
            self.barycentric_coordinates, values, interp_mode=self.interp_mode
        )

    def create_contour_line_traces(self) -> List[go.Scatterternary]:
        """Creates the contour line traces for a ternary contour plot.

        This method prepares the contour traces based on the grid values and
        coordinates defined for the instance. It also applies a workaround to
        show the outer border of the ternary plot.

        Returns:
            List[go.Scatterternary]: A list of Plotly Scatterternary objects that
            represent the contour lines in a ternary plot.
        """

        contour_traces, _ = _contour_trace(
            self.grid_coord_x,
            self.grid_coord_y,
            self.grid_values,
            ncontours=self.num_contours,
            colorscale=self.plot_colorscale,
            linecolor=self.line_color,
            interp_mode=self.interp_mode,
            coloring="lines",
            v_min=self.min_value,
            v_max=self.max_value,
        )

        # Magic number, because the line width of the contour lines is 1 and not
        # customizable. We can't use the line width of the contour lines because it will
        # not be visible.
        BORDER_LINE_WIDTH = 4
        contour_traces[0].update(
            dict(
                a=[1, 0, 0, 1],
                b=[0, 1, 0, 0],
                c=[0, 0, 1, 0],
                line=dict(
                    color=self.line_color,
                    width=BORDER_LINE_WIDTH,
                    shape="linear",
                ),
                fill="none",
            )
        )

        return contour_traces

    def create_contour_fill_traces(self) -> List[go.Scatterternary]:
        contour_traces, discrete_cm = _contour_trace(
            self.grid_coord_x,
            self.grid_coord_y,
            self.grid_values,
            ncontours=self.num_contours,
            colorscale=self.plot_colorscale,
            # A workaround due to the implementation of _contour_trace, the color of the
            # area is always the same as the line color. When linecolor=None, we set the
            # fill color correctly.
            linecolor=None,
            interp_mode=self.interp_mode,
            v_min=self.min_value,
            v_max=self.max_value,
        )

        # Magic number, because the line width of the contour lines is 1 and not
        # customizable. We can't use the line width of the contour lines because it will
        # not be visible.
        BORDER_LINE_WIDTH = 4

        # Workaround to set the outer border of the ternary plot, which is not
        # visible but still there. This is necessary to create pixel-perfect plots that
        # match the contour line plot. When creating the contour line plot, we can
        # simply update some properties of the first trace. However, this is not
        # possible for the contour fill plot, because the first trace is also part of
        # the contour fill plot. Therefore, we need to insert a new trace at the
        # beginning of the list.
        contour_traces.insert(
            0,
            go.Scatterternary(
                dict(
                    type="scatterternary",
                    a=[1, 0, 0, 1],
                    b=[0, 1, 0, 0],
                    c=[0, 0, 1, 0],
                    line=dict(
                        color="rgba(0,0,0,0)",
                        width=BORDER_LINE_WIDTH,
                        shape="linear",
                    ),
                    fill="none",
                )
            ),
        )

        # Workaround to set the line color of the contour lines to transparent. There is
        # no option to set the line color to transparent in the contour trace using the
        # _contour_trace function. Therefore, we need to update the line color of each
        # trace.
        for trace in contour_traces:
            trace.update(dict(line=dict(color="rgba(0,0,0,0)")))

        return contour_traces

    def create_ternary_contour(self) -> go.Figure:
        return create_ternary_contour(
            coordinates=self.coordinates,
            values=self.values,
            interp_mode=self.interp_mode,
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
                    min=0.00,
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
                    bgcolor="rgb(255,255,255)",
                ),
                showlegend=False,
                paper_bgcolor=paper_bgcolor,
            )
        )


if __name__ == "__main__":
    # import plotly.io as pio

    # pio.renderers.default = "browser"

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

    line_traces = plotter.create_contour_line_traces()
    line_layout = plotter.generate_layout()

    go.Figure(data=line_traces, layout=line_layout).show()

    fill_traces = plotter.create_contour_fill_traces()
    fill_layout = plotter.generate_layout(
        show_lines=False, show_ticks=False, show_title=False, show_pole_labels=False
    )

    go.Figure(data=fill_traces, layout=fill_layout).show()
