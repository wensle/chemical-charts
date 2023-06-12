from chemical_charts.prompts.point_selection_strategies import (
    CombinedPointSelectionStrategy,
    PointSelectionStrategy,
)


class GroundTruthPromptPipeline:
    """
    A pipeline for creating prompts from ground truth masks.
    """

    def __init__(self):
        self.processors = []
        self.strategies = []

    def add_processor(self, processor):
        self.processors.append(processor)

    def add_strategy(self, strategy: PointSelectionStrategy):
        self.strategies.append(strategy)

    def process(self, mask):
        processed_mask = mask.copy()
        for processor in self.processors:
            processed_mask = processor.process(processed_mask)
        return processed_mask

    def execute_strategies(self, ground_truth_mask, num_points):
        processed_image = self.process(ground_truth_mask)

        combined_strategy = CombinedPointSelectionStrategy(*self.strategies)
        points = combined_strategy.get_points(processed_image, num_points)
        return points
