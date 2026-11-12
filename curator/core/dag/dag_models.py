from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel


class EdgeType(str, Enum):
    DATA_FLOW = "data_flow"      # Standard data flow
    CONDITIONAL = "conditional"  # Conditional routing
    MERGE = "merge"              # Data merge
    BROADCAST = "broadcast"      # Broadcast (one data stream fans out to multiple nodes)


@dataclass
class DAGNode:
    node_id: str
    operation_name: str               # Registration name in OperationRegistry
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # List of predecessor node IDs
    condition: Optional[str] = None   # Restricted expressions (num_rows/constants/comparisons/booleans/arithmetic); False passes through the input
    merge_strategy: str = "union"     # Multi-input merge strategy (currently only union is supported)


@dataclass
class DAGEdge:
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType = EdgeType.DATA_FLOW
    condition_expr: Optional[str] = None


class DAGDefinition(BaseModel):
    """DAG definition model with YAML loading support."""

    nodes: List[Dict[str, Any]]
    entry_nodes: List[str]    # Entry node IDs with no predecessor dependencies
    exit_nodes: List[str]     # Exit node IDs with no successor dependencies

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DAGDefinition":
        """Load a DAG definition from a YAML file.

        YAML format example (same level as pipeline_example.yaml):
            dag:
              entry_nodes: [load_text]
              exit_nodes: [save_merged]
              nodes:
                - id: load_text
                  operation: text_loading
                  params: {}
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dag_data = data.get("dag", data)
        return cls(
            nodes=dag_data.get("nodes", []),
            entry_nodes=dag_data.get("entry_nodes", []),
            exit_nodes=dag_data.get("exit_nodes", []),
        )

    def to_dag_nodes(self) -> Dict[str, DAGNode]:
        """Convert the nodes list to a {node_id: DAGNode} dictionary."""
        result: Dict[str, DAGNode] = {}
        for n in self.nodes:
            node_id = n.get("id") or n.get("node_id")
            if not node_id:
                continue
            result[node_id] = DAGNode(
                node_id=node_id,
                operation_name=n.get("operation") or n.get("operation_name", ""),
                params=n.get("params", {}),
                depends_on=n.get("depends_on", []),
                condition=n.get("condition"),
                merge_strategy=n.get("merge_strategy", "union"),
            )
        return result
