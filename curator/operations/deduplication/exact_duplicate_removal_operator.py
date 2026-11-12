from typing import Any, Dict, Optional
import ray
from .deduplication_operator import DeduplicationOperation
from base.dataset import MultimodalDataset


class ExactDuplicateRemovalOperation(DeduplicationOperation):
    """Exact duplicate removal operation."""


    PARAMS_SCHEMA = {
        "category":    "deduplication",
        "description": "Exact deduplication: remove identical data records (calls Ray Dataset.distinct())",
        "params":      {},
        "example":     {},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.deduplicate(dataset)

    def deduplicate(self, dataset):
        """
        Implement exact duplicate removal.

        Use Ray Dataset groupby and aggregation to deduplicate rows.
        """
        import json
        
        # Approach: deduplicate with map_batches.
        # Ray 2.5+ requires returning a dict rather than a list.
        def deduplicate_batch(batch):
            """Deduplicate a batch and return a dict."""
            seen = set()
            unique_rows = []
            
            # The batch is a dict mapping column names to lists.
            # Convert it to a list of rows.
            if isinstance(batch, dict):
                # Get the batch size
                batch_size = len(list(batch.values())[0]) if batch else 0
                rows = []
                for i in range(batch_size):
                    row = {k: v[i] for k, v in batch.items()}
                    rows.append(row)
            else:
                rows = batch
            
            # Deduplicate
            for row in rows:
                row_key = json.dumps(row, sort_keys=True, default=str)
                if row_key not in seen:
                    seen.add(row_key)
                    unique_rows.append(row)
            
            # Convert back to dict format
            if unique_rows:
                result = {k: [row[k] for row in unique_rows] for k in unique_rows[0].keys()}
            else:
                result = {}
            
            return result
        
        # Deduplicate with map_batches
        return dataset.map_batches(deduplicate_batch, batch_size=1000)
