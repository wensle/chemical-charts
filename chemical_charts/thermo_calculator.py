import itertools
from typing import List
import numpy as np
from pycalphad import Database, calculate
from pycalphad.core.utils import filter_phases, unpack_components


class ThermoCalculator:
    """A class used to calculate thermodynamic properties for all combinations of given
    components.

    Attributes:
        dbf (Database): pycalphad Database object representing the database of
        thermodynamic data. components (list of str): List of component names. pressures
        (np.ndarray): Numpy array of pressure values. temperatures (np.ndarray): Numpy
        array of temperature values. output_properties (list of str): List of output
        property names. component_combos (list of list of str): All combinations of
        given components.
    """

    def __init__(
        self,
        db_name: str,
        components: List[str],
        pressures: np.ndarray,
        temperatures: np.ndarray,
        output_properties: List[str],
    ):
        """Initializes ThermoCalculator with given parameters.

        Args:
            db_name (str): Name of the thermodynamic database file.
            components (List[str]): List of components.
            pressures (np.ndarray): Numpy array of pressure values.
            temperatures (np.ndarray): Numpy array of temperature values.
            output_properties (List[str]): List of output property names.
        """
        self.dbf = Database(db_name)
        self.components = components
        self.pressures = pressures
        self.temperatures = temperatures
        self.output_properties = output_properties
        self.component_combos = [self.components]

    @staticmethod
    def generate_self_cartesian(elements: List[str]) -> List[List[str]]:
        """Generates all combinations of given elements.

        Args:
            elements (List[str]): List of elements.

        Returns:
            List[List[str]]: List of all combinations of given elements.
        """
        combinations = []
        for r in range(1, len(elements) + 1):
            combinations.extend(itertools.combinations(elements, r))
        return [list(comb) for comb in combinations]

    def calculate_properties(self, phases: List[str] = None):
        """Calculates thermodynamic properties for all combinations of components.

        Yields:
            dict: Dictionary containing the calculated properties for each condition
                and phase.
        """
        for combo in self.component_combos:
            species_obj_set = unpack_components(self.dbf, combo)
            list_of_possible_phases = filter_phases(self.dbf, species_obj_set)
            if phases is not None:
                list_of_possible_phases = [
                    phase for phase in list_of_possible_phases if phase in phases
                ]
            for phase in list_of_possible_phases:
                yield from self.process_phase(species_obj_set, phase)

    def process_phase(self, species_obj_set, phase):
        """Processes a given phase for all combinations of temperature and pressure.

        Yields:
            dict: Dictionary containing the calculated properties for each condition.
        """
        for temp in self.temperatures:
            for pressure in self.pressures:
                yield from self.process_conditions(
                    species_obj_set, phase, temp, pressure
                )

    def process_conditions(self, species_obj_set, phase, temp, pressure):
        """Processes a given condition (temperature and pressure) for all output
            properties.

        Yields:
            dict: Dictionary containing the calculated properties for each output
                property.
        """
        for output_property in self.output_properties:
            condition_result = self.calculate_condition(
                species_obj_set, phase, temp, pressure, output_property
            )
            if condition_result is not None:
                yield condition_result

    def calculate_condition(
        self, species_obj_set, phase, temp, pressure, output_property
    ) -> dict:
        """Calculates a specific output property for a given condition and phase.

        Args:
            species_obj_set: Set of species objects for the current component
                combination.
            phase (str): Phase to process.
            temp (float): Temperature value.
            pressure (float): Pressure value.
            output_property (str): Output property name.

        Returns:
            dict: Dictionary containing the calculated properties, or None if the
                conditions do not yield a valid calculation.
        """
        c = calculate(
            self.dbf,
            species_obj_set,
            phase,
            output=output_property,
            T=temp,
            P=pressure,
            pdens=500,
        )
        if len(c.X.points) in [0, 1]:
            return None

        xs = c.X.values[0, 0, 0, :, 0]
        ys = c.X.values[0, 0, 0, :, 1]
        zs = c[output_property].values[0, 0, 0, :]

        # Check if barycentric coordinates are positive
        if np.any(np.stack((xs, ys, 1 - xs - ys)) < 0):
            return None

        result = {
            "phase": phase,
            "temperature": temp,
            "pressure": pressure,
            "composition": self.components,
            "output_property": output_property,
            "xs": xs,
            "ys": ys,
            "zs": zs,
        }
        return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    db_fname = "Al-Cu-Y.tdb"
    db_path = os.path.join(os.getenv("TDB_DIR"), db_fname)

    # use the class
    components = ["AL", "CU", "Y", "VA"]
    pressures = [101325]  # Atmospheric pressure in Pa
    temperatures = np.linspace(573.15, 1573.15, 11)
    output_properties = ["GM_MIX", "SM_MIX", "HM_MIX"]
    calculator = ThermoCalculator(
        db_path, components, pressures, temperatures, output_properties
    )
    # results = calculator.calculate_properties()
    # print(results)
