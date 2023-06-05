import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory._ternary_contour import create_ternary_contour


class ContourPlot:
    """Class for creating ternary contour plots.

    Attributes:
        width (int): Width of the plot.
        height (int): Height of the plot.
        border_line_width (int): Width of the border line.
        pole_labels (list): Labels for the poles.
    """

    def __init__(
        self, width=700, height=700, border_line_width=14, pole_labels=["Al", "Y", "Cu"]
    ):
        """Initializes ContourPlot with the given parameters.

        Args:
            width (int): Width of the plot.
            height (int): Height of the plot.
            border_line_width (int): Width of the border line.
            pole_labels (list): Labels for the poles.
        """
        self.width = width
        self.height = height
        self.border_line_width = border_line_width
        self.pole_labels = pole_labels

    def prepare_data(self, xs, ys, zs):
        """Prepares data for contour plot.

        Args:
            xs (np.ndarray): x coordinates.
            ys (np.ndarray): y coordinates.
            zs (np.ndarray): z coordinates (values for the contour).

        Returns:
            np.ndarray: Prepared data for contour plot.
        """
        return np.array([xs, ys, 1 - xs - ys]), zs

    def create_contour_plot(self, coordinates, values, plot_title, ncontours):
        """Creates a contour plot.

        Args:
            coordinates (np.ndarray): x, y coordinates and their sum (which should
                equal 1).
            values (np.ndarray): z values for the contour.
            plot_title (str): Title for the plot.

        Returns:
            plotly.graph_objs._figure.Figure: The contour plot.
        """
        contour_fig = create_ternary_contour(
            coordinates,
            values,
            pole_labels=self.pole_labels,
            interp_mode="cartesian",
            coloring="lines",
            linecolor="black",
            width=self.width,
            height=self.height,
            title=plot_title,
            ncontours=ncontours,
        )

        contour_fig.update_ternaries(
            dict(
                aaxis_showgrid=False,
                baxis_showgrid=False,
                caxis_showgrid=False,
                bgcolor="rgba(0,0,0,0)",
            )
        )

        contour_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")

        contour_fig.add_trace(
            go.Scatterternary(
                a=[1, 0, 0, 1],
                b=[0, 1, 0, 0],
                c=[0, 0, 1, 0],
                mode="lines",
                line=dict(color="black", width=self.border_line_width),
                fill="toself",
                fillcolor="rgba(0,0,0,0)",
                name="border",
            )
        )

        return contour_fig

    def create_mask(self, coordinates, values, ncontours):
        """Creates a mask for the contour plot.

        Args:
            coordinates (np.ndarray): x, y coordinates and their sum (which should
                equal 1).
            values (np.ndarray): z values for the contour.

        Returns:
            plotly.graph_objs._figure.Figure: The mask for the contour plot.
        """
        mask_fig = create_ternary_contour(
            coordinates,
            values,
            pole_labels=["", "", ""],
            interp_mode="cartesian",
            width=self.width,
            height=self.height,
            ncontours=ncontours,
        )

        mask_fig.update_traces(dict(line=dict(color="rgba(0,0,0,0)", width=0)))

        axis_dict = dict(
            showgrid=False,
            ticks="",
            tickmode="array",
            tickvals=[],
            linewidth=0,
            color="rgba(0,0,0,0)",
        )

        layout_dict = dict(
            ternary=dict(
                aaxis=axis_dict,
                baxis=axis_dict,
                caxis=axis_dict,
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        mask_fig.update_layout(layout_dict)

        return mask_fig


if __name__ == "__main__":
    # Example usage:
    xs = np.array([])  # placeholder
    ys = np.array([])  # placeholder
    zs = np.array([])  # placeholder

    plot_data = {
        "phase": "Phase",
        "temperature": "Temperature",
        "pressure": "Pressure",
        "composition": ["AL", "CU", "Y", "VA"],
        "output_property": "Output Property",
        "xs": xs,
        "ys": ys,
        "zs": zs,
    }

    # Create a ContourPlot instance
    contour_plot = ContourPlot()

    # Prepare data
    coordinates, values = contour_plot.prepare_data(
        plot_data["xs"], plot_data["ys"], plot_data["zs"]
    )

    # Create contour plot
    contour_plot_fig = contour_plot.create_contour_plot(
        coordinates,
        values,
        (
            f'{plot_data["phase"]} - {plot_data["temperature"]} - '
            f'{plot_data["pressure"]} -'
            f'{", ".join(plot_data["composition"])} - {plot_data["output_property"]}'
        ),
    )

    # Create mask
    mask_fig = contour_plot.create_mask(coordinates, values)

    # Display the figures
    contour_plot_fig.show()
    mask_fig.show()
