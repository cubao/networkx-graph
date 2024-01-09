from __future__ import annotations

from typing import Optional

from ._core import DiGraph as DiGraphImpl
from ._core import (
    Edge,
    Node,
    Route,
    ShortestPathGenerator,
    __doc__,
    __version__,
    add,
    rapidjson,
    subtract,
)


class DiGraph(DiGraphImpl):
    def __init__(self, round_n: Optional[int] = 3):  # noqa: UP007
        super().__init__(round_n=round_n)

    def add_node(self, node: str, length: float = 1.0, **attr):
        assert isinstance(node, str), "we only supports str key!"
        node = super().add_node(node, length=length)
        for k, v in attr.items():
            node[k] = v
        return node

    def add_edge(self, node0: str, node1: str, **attr):
        assert isinstance(node0, str), "we only supports str key!"
        assert isinstance(node1, str), "we only supports str key!"
        edge = super().add_edge(node0, node1)
        for k, v in attr.items():
            edge[k] = v
        return edge


__all__ = [
    "__doc__",
    "__version__",
    "add",
    "subtract",
    "rapidjson",
    "Node",
    "Edge",
    "Route",
    "DiGraph",
    "ShortestPathGenerator",
]
