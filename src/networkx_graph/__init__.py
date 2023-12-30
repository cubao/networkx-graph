from __future__ import annotations

from ._core import DiGraph as DiGraphImpl
from ._core import Edge, Node, __doc__, __version__, add, rapidjson, subtract


class DiGraph(DiGraphImpl):
    def __init__(self):
        super().__init__()

    @property
    def nodes(self):
        return super().nodes()

    @property
    def edges(self):
        return super().edges()

    def add_node(self, node: str, length: float = 1.0, **attr):
        if not isinstance(node, str):
            node = str(node)
        node = super().add_node(node, length=length)
        for k, v in attr.items():
            setattr(node, k, v)
        return node

    def add_edge(self, node0: str, node1: str, **attr):
        if not isinstance(node0, str):
            node0 = str(node0)
        if not isinstance(node1, str):
            node1 = str(node1)
        edge = super().add_edge(node0, node1)
        for k, v in attr.items():
            setattr(edge, k, v)
        return edge


__all__ = [
    "__doc__",
    "__version__",
    "add",
    "subtract",
    "rapidjson",
    "Node",
    "Edge",
    "DiGraph",
]
