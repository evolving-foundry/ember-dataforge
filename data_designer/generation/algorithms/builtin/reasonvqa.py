from pathlib import Path

from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from .descriptors import REASONVQA


class ReasonVqaAlgorithm:
    def describe(self):
        return REASONVQA

    def run(self, request, context):
        from .reasonvqa_paths import (
            _path_is_question_usable, generate_reasonvqa_row_from_wikidata_path,
        )
        from .reasonvqa_inaturalist_templates import with_inaturalist_reasonvqa_template_defaults
        cfg = with_inaturalist_reasonvqa_template_defaults(dict(request.config))
        usable = [r for r in request.records if _path_is_question_usable(r.payload, cfg)]
        paths = [r.payload for r in usable]
        output, rejected = [], len(request.records) - len(usable)
        for index, record in enumerate(usable):
            context.check_cancelled()
            try:
                row = generate_reasonvqa_row_from_wikidata_path(
                    path_record=record.payload, all_path_records=paths, row_index=index,
                    config=cfg, recipe="knowledge_reasoning_vqa", generator_prompt_version="v1",
                    schema_version=REASONVQA.output_schema, params={})
            except ValueError as error:
                if cfg.get("difficulty_policy", {}).get("drop_rows_without_hard_negatives") and "Not enough typed distractors" in str(error):
                    rejected += 1
                    continue
                raise
            output.append(RecordEnvelope("qa:" + record.id, row, REASONVQA.output_schema, (record.id,)))
        self._write_exports(cfg, [record.payload for record in output])
        return AlgorithmResult(output, {"unusable_path": rejected} if rejected else {})

    @staticmethod
    def _write_exports(cfg, rows):
        # When image_dir is not configured, default_image_dir falls back to export_dir/images,
        # so export_dir anchors both the image directory and the exported artifacts.
        if not rows or not cfg.get("export_enabled", True) or not cfg.get("export_dir"):
            return
        from data_designer.storage.reasonvqa import write_reasonvqa_exports
        write_reasonvqa_exports(
            rows=rows, export_dir=Path(cfg["export_dir"]),
            split=str(cfg.get("split") or "train"),
            categories=list(cfg.get("categories") or []),
        )
