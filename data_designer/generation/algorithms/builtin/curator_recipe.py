from data_designer.contracts.algorithms import AlgorithmResult
from .descriptors import CURATOR_RECIPES


class CuratorRecipeAlgorithm:
    recipe = ""

    def describe(self):
        return CURATOR_RECIPES[self.recipe]

    def run(self, request, context):
        return AlgorithmResult(context.provider("curator.recipe").execute(
            self.recipe, request.records, dict(request.config)))


class DenseCaptionAlgorithm(CuratorRecipeAlgorithm):
    recipe = "dense_caption"


class VideoTemporalQaAlgorithm(CuratorRecipeAlgorithm):
    recipe = "video_temporal_qa"
