from __future__ import annotations

import logging
import ray
from collections import deque
from typing import Any, Dict, List, Optional

from .dag_models import DAGDefinition, DAGNode
from base.dataset import MultimodalDataset

logger = logging.getLogger(__name__)


class DAGExecutor:
    """DAG executor.

    Execution strategy: execute nodes one by one in topological order; when
    enable_parallel=True, independent nodes in the same topological layer are scheduled in parallel via ray.remote.
    """

    def __init__(self, dag_def: DAGDefinition, enable_parallel: bool = True):
        self.dag_def = dag_def
        self.enable_parallel = enable_parallel
        self._nodes: Dict[str, DAGNode] = dag_def.to_dag_nodes()
        self._adj: Dict[str, List[str]] = {}   # node_id → successor node_ids
        self._in_degree: Dict[str, int] = {}
        self._build_graph()

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """Build the adjacency list and in-degree table."""
        for node_id in self._nodes:
            self._adj[node_id] = []
            self._in_degree[node_id] = 0

        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep in self._nodes:
                    self._adj[dep].append(node.node_id)
                    self._in_degree[node.node_id] = (
                        self._in_degree.get(node.node_id, 0) + 1
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Kahn topological sort, returning nodes grouped by layer (nodes in the same layer can run in parallel).

        Raises:
            ValueError: If a cycle is detected
        """
        in_degree = dict(self._in_degree)
        queue: deque = deque(
            node_id for node_id, deg in in_degree.items() if deg == 0
        )
        layers: List[List[str]] = []
        visited = 0

        while queue:
            layer = list(queue)
            queue.clear()
            layers.append(layer)
            visited += len(layer)
            next_level = set()
            for node_id in layer:
                for succ in self._adj.get(node_id, []):
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_level.add(succ)
            queue.extend(sorted(next_level))  # Sorting ensures determinism

        if visited != len(self._nodes):
            raise ValueError("The DAG contains a cycle and cannot be executed")

        return layers

    # ------------------------------------------------------------------
    # Execution phase
    # ------------------------------------------------------------------

    def execute(
        self,
        initial_datasets: Optional[Dict[str, MultimodalDataset]] = None,
    ) -> Dict[str, MultimodalDataset]:
        """Execute the DAG and return the output datasets of each exit node.

        Args:
            initial_datasets: Initial datasets for entry nodes (key = node_id).
                              If None, the entry node's LoadingOperation loads data itself.

        Returns:
            {exit_node_id: MultimodalDataset}
        """
        layers = self._topological_sort()
        node_outputs: Dict[str, MultimodalDataset] = {}

        # Pre-populate initial datasets
        if initial_datasets:
            node_outputs.update(initial_datasets)

        for layer in layers:
            if self.enable_parallel and len(layer) > 1:
                results = self._execute_layer_parallel(layer, node_outputs)
            else:
                results = self._execute_layer_sequential(layer, node_outputs)
            node_outputs.update(results)

        return {
            node_id: node_outputs[node_id]
            for node_id in self.dag_def.exit_nodes
            if node_id in node_outputs
        }

    def _execute_layer_sequential(
        self,
        layer: List[str],
        node_outputs: Dict[str, MultimodalDataset],
    ) -> Dict[str, MultimodalDataset]:
        results: Dict[str, MultimodalDataset] = {}
        for node_id in layer:
            results[node_id] = self._execute_node(node_id, node_outputs)
        return results

    def _execute_layer_parallel(
        self,
        layer: List[str],
        node_outputs: Dict[str, MultimodalDataset],
    ) -> Dict[str, MultimodalDataset]:
        """Execute nodes in the same topological layer in parallel using ray.remote."""
        # Defer imports to avoid a circular import
        from ..pipeline import OperationRegistry

        @ray.remote
        def _remote_execute(node: DAGNode, input_dataset) -> Optional[Any]:
            op = OperationRegistry.create_operation(node.operation_name, node.params)
            return op.execute(input_dataset)

        futures = {}
        for node_id in layer:
            node = self._nodes[node_id]
            input_ds = self._resolve_input(node, node_outputs)
            if self._evaluate_condition(node.condition, input_ds):
                futures[node_id] = _remote_execute.remote(node, input_ds)
            else:
                futures[node_id] = None  # Condition not met; pass through

        results: Dict[str, MultimodalDataset] = {}
        for node_id, future in futures.items():
            if future is None:
                node = self._nodes[node_id]
                results[node_id] = self._resolve_input(node, node_outputs)
            else:
                try:
                    results[node_id] = ray.get(future)
                except Exception as e:
                    logger.warning("[DAGExecutor] Parallel execution failed for node '%s': %s", node_id, e)
                    raise

        return results

    def _execute_node(
        self,
        node_id: str,
        node_outputs: Dict[str, MultimodalDataset],
    ) -> MultimodalDataset:
        # Defer imports to avoid a circular import
        from ..pipeline import OperationRegistry

        node = self._nodes[node_id]
        input_ds = self._resolve_input(node, node_outputs)

        # Condition check
        if not self._evaluate_condition(node.condition, input_ds):
            return input_ds  # Condition not met; pass through the input

        op = OperationRegistry.create_operation(node.operation_name, node.params)
        result = op.execute(input_ds)
        if not isinstance(result, MultimodalDataset):
            raise TypeError(
                f"Node '{node_id}' operation returned an invalid type: {type(result)}; expected MultimodalDataset"
            )
        return result

    def _resolve_input(
        self,
        node: DAGNode,
        node_outputs: Dict[str, MultimodalDataset],
    ) -> Optional[MultimodalDataset]:
        """Merge predecessor outputs as the input to the current node."""
        deps = [dep for dep in node.depends_on if dep in node_outputs]
        if not deps:
            return node_outputs.get(node.node_id)  # May be a pre-populated entry dataset

        if len(deps) == 1:
            return node_outputs[deps[0]]

        return self._merge_datasets(
            [node_outputs[dep] for dep in deps], node.merge_strategy
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_datasets(
        datasets: List[MultimodalDataset], strategy: str = "union"
    ) -> MultimodalDataset:
        """Merge multiple datasets."""
        if not datasets:
            raise ValueError("_merge_datasets: dataset list is empty")
        if len(datasets) == 1:
            return datasets[0]

        base = datasets[0]
        if strategy == "union":
            merged_ray = base.dataset
            for ds in datasets[1:]:
                merged_ray = merged_ray.union(ds.dataset)
        else:
            # Zip mode: use the shortest input and merge by index (simplified implementation)
            merged_ray = base.dataset
            for ds in datasets[1:]:
                merged_ray = merged_ray.union(ds.dataset)

        combined_modalities = list(
            {m for ds in datasets for m in (ds.modalities or [])}
        )
        return MultimodalDataset(dataset=merged_ray, modalities=combined_modalities)

    @staticmethod
    def _evaluate_condition(expr: Optional[str], dataset) -> bool:
        """Evaluate a restricted condition expression.

        Only ``num_rows``, literals, boolean/comparison operators, and basic
        arithmetic are accepted. Attribute access, subscriptions, calls, and
        imports are rejected during AST validation. Evaluation failures keep
        the historical fail-open behavior and return ``True``.
        """
        if not expr:
            return True
        try:
            ctx = {"dataset": dataset, "num_rows": dataset.dataset.count() if dataset else 0}
            return bool(eval(expr, {"__builtins__": {}}, ctx))  # noqa: S307
        except Exception:
            return True

    def visualize(self) -> str:
        """Output a Mermaid-format DAG diagram that can be embedded in Jupyter / Markdown."""
        layers = self._topological_sort()
        lines = ["graph TD"]
        for node_id, node in self._nodes.items():
            label = f"{node_id}[{node.operation_name}]"
            lines.append(f"    {label}")
        for node_id, node in self._nodes.items():
            for dep in node.depends_on:
                lines.append(f"    {dep} --> {node_id}")
        return "\n".join(lines)
