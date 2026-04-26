"""Save and load .midas session files (JSON format)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from midas_gui.session import GraphModel, NodeModel, Edge, NODE_TYPES

_VERSION = "1.0"

# Property keys that should not be serialized
_SKIP_PROPS = {"_class", "_unresolved"}


def _clean_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of a node's properties dict."""
    clean: dict[str, Any] = {}
    for key, val in props.items():
        if key in _SKIP_PROPS:
            continue
        if isinstance(val, np.ndarray):
            continue
        clean[key] = val
    return clean


def save_session(
    graph: GraphModel,
    path: str | Path,
    imported_modules: list[str] | None = None,
) -> None:
    """Serialize the graph to a .midas JSON file."""
    nodes = []
    for node in graph.nodes.values():
        nodes.append({
            "id": node.id,
            "type_id": node.type_id,
            "x": node.x,
            "y": node.y,
            "properties": _clean_properties(node.properties),
        })

    edges = []
    for edge in graph.edges:
        edges.append({
            "source_node_id": edge.source_node_id,
            "source_port_name": edge.source_port_name,
            "target_node_id": edge.target_node_id,
            "target_port_name": edge.target_port_name,
        })

    data = {
        "version": _VERSION,
        "name_counters": dict(graph._name_counters),
        "imported_modules": imported_modules or [],
        "nodes": nodes,
        "edges": edges,
    }

    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_imported_modules(path: str | Path) -> list[str]:
    """Read just the imported_modules list from a .midas file without loading the graph."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return data.get("imported_modules", [])


def load_session(path: str | Path) -> tuple[GraphModel, list[str]]:
    """Deserialize a .midas JSON file into a GraphModel.

    Returns
    -------
    (graph, imported_modules)
        The reconstructed graph and list of user module paths to re-import.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)

    graph = GraphModel()

    # Restore name counters
    for type_id, count in data.get("name_counters", {}).items():
        graph._name_counters[type_id] = count

    # Restore nodes
    for node_data in data.get("nodes", []):
        type_id = node_data["type_id"]
        if type_id not in NODE_TYPES:
            continue  # skip nodes whose type isn't registered

        spec = NODE_TYPES[type_id]
        # Start from the spec defaults, then overlay saved properties
        props = dict(spec.default_properties)
        saved_props = node_data.get("properties", {})
        props.update(saved_props)

        node = NodeModel(
            id=node_data["id"],
            type_id=type_id,
            x=node_data.get("x", 0.0),
            y=node_data.get("y", 0.0),
            properties=props,
        )
        graph.nodes[node.id] = node

    # Restore edges (skip edges referencing missing nodes/ports)
    for edge_data in data.get("edges", []):
        src_id = edge_data["source_node_id"]
        tgt_id = edge_data["target_node_id"]
        if src_id not in graph.nodes or tgt_id not in graph.nodes:
            continue
        edge = Edge(
            source_node_id=src_id,
            source_port_name=edge_data["source_port_name"],
            target_node_id=tgt_id,
            target_port_name=edge_data["target_port_name"],
        )
        graph.edges.append(edge)

    imported_modules = data.get("imported_modules", [])
    return graph, imported_modules
