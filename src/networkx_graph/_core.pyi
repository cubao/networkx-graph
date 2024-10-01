"""

        Pybind11 example plugin
        -----------------------

        .. currentmodule:: network-graph

        .. autosummary::
           :toctree: _generate

           add
           subtract

"""
from __future__ import annotations

import typing

import pybind11_stubgen.typing_ext

__all__ = [
    "Bindings",
    "DiGraph",
    "Edge",
    "Endpoints",
    "Indexer",
    "Node",
    "Path",
    "Sequences",
    "ShortestPathGenerator",
    "ShortestPathWithUbodt",
    "Sinks",
    "UbodtRecord",
    "ZigzagPath",
    "ZigzagPathGenerator",
    "add",
    "subtract",
]

class Bindings:
    def __call__(self) -> dict[str, list[tuple[float, float, typing.Any]]]:
        """
        Get the map of node IDs to their bindings
        """
    @property
    def graph(self) -> ...:
        """
        Get the DiGraph object associated with this Bindings object
        """

class DiGraph:
    def __init__(self, round_n: int | None = 3) -> None: ...
    def add_edge(self, node0: str, node1: str) -> Edge:
        """
        Add an edge between two nodes in the graph
        """
    def add_node(self, id: str, *, length: float) -> Node:
        """
        Add a node to the graph
        """
    def all_paths(
        self,
        source: str,
        target: str,
        *,
        cutoff: float,
        source_offset: float | None = None,
        target_offset: float | None = None,
        sinks: Sinks = None,
    ) -> list[Path]:
        """
        Find all paths between two nodes
        """
    def all_paths_from(
        self,
        source: str,
        *,
        cutoff: float,
        offset: float | None = None,
        sinks: Sinks = None,
    ) -> list[Path]:
        """
        Find all paths from a source node
        """
    def all_paths_to(
        self,
        target: str,
        *,
        cutoff: float,
        offset: float | None = None,
        sinks: Sinks = None,
    ) -> list[Path]:
        """
        Find all paths to a target node
        """
    def all_paths_to_bindings(
        self,
        source: str,
        *,
        cutoff: float,
        bindings: Bindings,
        offset: float | None = None,
        direction: int = 0,
        sinks: Sinks = None,
        with_endings: bool = False,
    ) -> tuple[list[Path], list[Path]]:
        """
        Find all paths to bindings
        """
    @typing.overload
    def build_ubodt(
        self, thresh: float, *, pool_size: int = 1, nodes_thresh: int = 100
    ) -> list[UbodtRecord]:
        """
        Build UBODT (Upper Bounded Origin Destination Table)
        """
    @typing.overload
    def build_ubodt(self, source: int, thresh: float) -> list[UbodtRecord]:
        """
        Build UBODT (Upper Bounded Origin Destination Table) from a specific source
        """
    def distance_to_bindings(
        self,
        source: str,
        *,
        cutoff: float,
        bindings: Bindings,
        offset: float | None = None,
        direction: int = 0,
        sinks: Sinks = None,
    ) -> tuple[float | None, float | None]:
        """
        Calculate distances to bindings
        """
    def encode_bindings(
        self, bindings: dict[str, list[tuple[float, float, typing.Any]]]
    ) -> Bindings:
        """
        Encode bindings
        """
    def encode_endpoints(
        self,
        endpoints: dict[
            str,
            tuple[
                typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)],
                typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)],
            ],
        ],
        *,
        is_wgs84: bool = True,
    ) -> Endpoints:
        """
        Encode endpoints
        """
    def encode_sequences(self, sequences: list[list[str]]) -> Sequences:
        """
        Encode sequences
        """
    def encode_sinks(self, sinks: set[str]) -> Sinks:
        """
        Encode sink nodes
        """
    def encode_ubodt(
        self,
        source_road: str,
        target_road: str,
        source_next: str,
        target_prev: str,
        cost: float,
    ) -> UbodtRecord | None:
        """
        Encode UBODT (Upper Bounded Origin Destination Table)
        """
    def predecessors(self, id: str) -> list[str]:
        """
        Get predecessors of a node
        """
    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        cutoff: float,
        source_offset: float | None = None,
        target_offset: float | None = None,
        sinks: Sinks = None,
        endpoints: Endpoints = None,
    ) -> Path | None:
        """
        Find the shortest path between two nodes
        """
    def shortest_path_to_bindings(
        self,
        source: str,
        *,
        cutoff: float,
        bindings: Bindings,
        offset: float | None = None,
        direction: int = 0,
        sinks: Sinks = None,
    ) -> tuple[Path | None, Path | None]:
        """
        Find the shortest path to bindings
        """
    def shortest_paths_from(
        self,
        source: str,
        *,
        cutoff: float,
        offset: float | None = None,
        sinks: Sinks = None,
    ) -> ShortestPathGenerator:
        """
        Find shortest paths from a source node to all reachable nodes
        """
    def shortest_paths_to(
        self,
        target: str,
        *,
        cutoff: float,
        offset: float | None = None,
        sinks: Sinks = None,
    ) -> ShortestPathGenerator:
        """
        Find shortest paths to a target node from all reachable nodes
        """
    @typing.overload
    def shortest_zigzag_path(
        self, source: str, target: str, *, cutoff: float, direction: int = 0
    ) -> ZigzagPath | None:
        """
        Find the shortest zigzag path between two nodes
        """
    @typing.overload
    def shortest_zigzag_path(
        self, source: str, *, cutoff: float, direction: int = 0
    ) -> ZigzagPathGenerator:
        """
        Find the shortest zigzag paths from a source node
        """
    def successors(self, id: str) -> list[str]:
        """
        Get successors of a node
        """
    @property
    def edges(self) -> dict[tuple[str, str], Edge]:
        """
        Get all edges in the graph
        """
    @property
    def indexer(self) -> Indexer:
        """
        Get the indexer for the graph
        """
    @property
    def nodes(self) -> dict[str, Node]:
        """
        Get all nodes in the graph
        """
    @property
    def round_n(self) -> int | None:
        """
        Get the number of decimal places to round to
        """
    @property
    def round_scale(self) -> float | None:
        """
        Get the scale factor for rounding
        """
    @property
    def sibs_under_next(self) -> dict[str, set[str]]:
        """
        Get siblings under the next node
        """
    @property
    def sibs_under_prev(self) -> dict[str, set[str]]:
        """
        Get siblings under the previous node
        """

class Edge:
    def __getitem__(self, attr_name: str) -> typing.Any:
        """
        Get an attribute of the Edge by name
        """
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: str, arg1: typing.Any) -> typing.Any:
        """
        Set an attribute of the Edge by name
        """
    def to_dict(self) -> dict:
        """
        Convert the Edge to a dictionary
        """

class Endpoints:
    @property
    def graph(self) -> ...:
        """
        Get the DiGraph object associated with this Endpoints object
        """
    @property
    def is_wgs84(self) -> bool:
        """
        Check if the coordinates are in WGS84 format
        """

class Indexer:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, index: dict[str, int]) -> None: ...
    @typing.overload
    def contains(self, id: int) -> bool:
        """
        Check if the Indexer contains the given integer ID
        """
    @typing.overload
    def contains(self, id: str) -> bool:
        """
        Check if the Indexer contains the given string ID
        """
    @typing.overload
    def get_id(self, id: int) -> str | None:
        """
        Get the string ID corresponding to the given integer ID
        """
    @typing.overload
    def get_id(self, id: str) -> int | None:
        """
        Get the integer ID corresponding to the given string ID
        """
    @typing.overload
    def id(self, id: int) -> str:
        """
        Get or create an integer ID for the given integer
        """
    @typing.overload
    def id(self, id: str) -> int:
        """
        Get or create an integer ID for the given string
        """
    @typing.overload
    def index(self, str_id: str, int_id: int) -> bool:
        """
        Add a new string-integer ID pair to the Indexer
        """
    @typing.overload
    def index(self, index: dict[str, int]) -> bool:
        """
        Add multiple string-integer ID pairs to the Indexer
        """
    @typing.overload
    def index(self) -> dict[str, int]:
        """
        Get the current index mapping
        """

class Node:
    def __getitem__(self, attr_name: str) -> typing.Any:
        """
        Get an attribute of the Node by name
        """
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: str, arg1: typing.Any) -> typing.Any:
        """
        Set an attribute of the Node by name
        """
    def to_dict(self) -> dict:
        """
        Convert the Node to a dictionary
        """
    @property
    def length(self) -> float:
        """
        Get the length of the node
        """

class Path:
    @staticmethod
    def Build(
        graph: ...,
        nodes: list[str],
        *,
        start_offset: float | None = None,
        end_offset: float | None = None,
        binding: tuple[str, tuple[float, float, typing.Any]] | None = None,
    ) -> Path:
        """
        Build a Path object from a list of nodes and optional parameters
        """
    def __getitem__(self, attr_name: str) -> typing.Any:
        """
        Get an attribute of the Path by name
        """
    def __setitem__(self, arg0: str, arg1: typing.Any) -> typing.Any:
        """
        Set an attribute of the Path by name
        """
    def along(self, offset: float) -> tuple[str, float]:
        """
        Get the node and offset along the Path at a given distance
        """
    def locate(self, ref: tuple[str, float], eps: float = 0.01) -> float | None:
        """
        Locate ref:=(node_id, offset) in the Path, return offset along path
        """
    def offsets(self) -> list[float]:
        """
        Get offsets of each node in the Path
        """
    def search_for_seqs(
        self, sequences: ..., quick_return: bool = True
    ) -> dict[int, list[Path]]:
        """
        Search for sequences within the Path
        """
    def slice(self, start: float, end: float) -> Path:
        """
        Create a new Path that is a slice of the current Path
        """
    def to_dict(self) -> dict:
        """
        Convert the Path to a dictionary
        """
    @property
    def _signature(self) -> tuple:
        """
        Get the internal signature of the Path
        """
    @property
    def binding(self) -> tuple[str, tuple[float, float, typing.Any]] | None:
        """
        Get the binding information of the Path, if any
        """
    @property
    def dist(self) -> float:
        """
        Get the total distance of the Path
        """
    @property
    def end(self) -> tuple[str, float | None]:
        """
        Get the end node and offset of the Path
        """
    @property
    def graph(self) -> ...:
        """
        Get the DiGraph object associated with this Path
        """
    @property
    def nodes(self) -> list[str]:
        """
        Get the list of node IDs in the Path
        """
    @property
    def start(self) -> tuple[str, float | None]:
        """
        Get the start node and offset of the Path
        """

class Sequences:
    @property
    def graph(self) -> ...:
        """
        Get the DiGraph object associated with this Sequences object
        """

class ShortestPathGenerator:
    def __init__(self) -> None: ...
    def cutoff(self) -> float:
        """
        Get the cutoff distance for the shortest path
        """
    def destinations(self) -> list[tuple[float, str]]:
        """
        Get the sorted list of destinations and their distances
        """
    def dists(self) -> dict[str, float]:
        """
        Get the map of distances to each node in the shortest path
        """
    def path(self, node: str) -> Path | None:
        """
        Get the shortest path to a specific node
        """
    def paths(self) -> list[Path]:
        """
        Get all shortest paths from the source to all reachable nodes within the cutoff distance
        """
    def prevs(self) -> dict[str, str]:
        """
        Get the map of previous nodes in the shortest path
        """
    def source(self) -> tuple[str, float | None] | None:
        """
        Get the source node and offset for the shortest path
        """
    def target(self) -> tuple[str, float | None] | None:
        """
        Get the target node and offset for the shortest path
        """
    def to_dict(self) -> dict:
        """
        Convert the ShortestPathGenerator object to a dictionary
        """

class ShortestPathWithUbodt:
    @staticmethod
    def Dump_Ubodt(ubodt: list[UbodtRecord], path: str) -> bool:
        """
        Dump UBODT to a file (static method)
        """
    @staticmethod
    def Load_Ubodt(path: str) -> list[UbodtRecord]:
        """
        Load UBODT from a file path (static method)
        """
    @typing.overload
    def __init__(self, graph: DiGraph, ubodt: list[UbodtRecord]) -> None:
        """
        Initialize ShortestPathWithUbodt with a DiGraph and UBODT records
        """
    @typing.overload
    def __init__(
        self,
        graph: DiGraph,
        thresh: float,
        *,
        pool_size: int = 1,
        nodes_thresh: int = 100,
    ) -> None:
        """
        Initialize ShortestPathWithUbodt with a DiGraph and build UBODT
        """
    @typing.overload
    def __init__(self, graph: DiGraph, path: str) -> None:
        """
        Initialize ShortestPathWithUbodt with a DiGraph and UBODT file path
        """
    def by_source(
        self, source: str, cutoff: float | None = None
    ) -> list[tuple[float, str]]:
        """
        Get paths from a source node
        """
    def by_target(
        self, target: str, cutoff: float | None = None
    ) -> list[tuple[float, str]]:
        """
        Get paths to a target node
        """
    def dist(self, source: str, target: str) -> float | None:
        """
        Get the distance between source and target nodes
        """
    @typing.overload
    def dump_ubodt(self) -> list[UbodtRecord]:
        """
        Dump UBODT to a vector of UbodtRecord
        """
    @typing.overload
    def dump_ubodt(self, path: str) -> bool:
        """
        Dump UBODT to a file
        """
    @typing.overload
    def load_ubodt(self, path: str) -> None:
        """
        Load UBODT from a file path
        """
    @typing.overload
    def load_ubodt(self, rows: list[UbodtRecord]) -> None:
        """
        Load UBODT from a vector of UbodtRecord
        """
    def path(self, source: str, target: str) -> Path | None:
        """
        Get the shortest path between source and target nodes
        """
    def size(self) -> int:
        """
        Get the size of the UBODT
        """

class Sinks:
    def __call__(self) -> set[str]:
        """
        Get the set of sink nodes
        """
    @property
    def graph(self) -> ...:
        """
        Get the DiGraph object associated with this Sinks object
        """

class UbodtRecord:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: UbodtRecord) -> bool: ...
    def __init__(
        self,
        source_road: int,
        target_road: int,
        source_next: int,
        target_prev: int,
        cost: float,
    ) -> None:
        """
        Initialize a UbodtRecord with source road, target road, source next, target previous, and cost
        """
    def __lt__(self, arg0: UbodtRecord) -> bool: ...
    @property
    def cost(self) -> float:
        """
        Get the cost associated with this record
        """
    @property
    def source_next(self) -> int:
        """
        Get the next source road ID
        """
    @property
    def source_road(self) -> int:
        """
        Get the source road ID
        """
    @property
    def target_prev(self) -> int:
        """
        Get the previous target road ID
        """
    @property
    def target_road(self) -> int:
        """
        Get the target road ID
        """

class ZigzagPath(Path):
    def to_dict(self) -> dict:
        """
        Convert the ZigzagPath to a dictionary
        """
    @property
    def directions(self) -> list[int]:
        """
        Get the list of directions for each node in the ZigzagPath
        """
    @property
    def dist(self) -> float:
        """
        Get the total distance of the ZigzagPath
        """
    @property
    def graph(self) -> ...:
        """
        Get the DiGraph object associated with this ZigzagPath
        """
    @property
    def nodes(self) -> list[str]:
        """
        Get the list of node IDs in the ZigzagPath
        """

class ZigzagPathGenerator:
    def __init__(self) -> None: ...
    def cutoff(self) -> float:
        """
        Get the cutoff distance for the zigzag path
        """
    def destinations(self) -> list[tuple[float, str]]:
        """
        Get the destinations and their distances in the zigzag path
        """
    def dists(self) -> dict[tuple[str, int], float]:
        """
        Get the distances to each node in the zigzag path
        """
    def path(self, node: str) -> ZigzagPath | None:
        """
        Get the zigzag path to a specific node
        """
    def paths(self) -> list[ZigzagPath]:
        """
        Get all zigzag paths
        """
    def prevs(self) -> dict[tuple[str, int], tuple[str, int]]:
        """
        Get the previous nodes in the zigzag path
        """
    def source(self) -> str | None:
        """
        Get the source node for the zigzag path
        """
    def to_dict(self) -> dict:
        """
        Convert the ZigzagPathGenerator object to a dictionary
        """

def add(arg0: int, arg1: int) -> int:
    """
    Add two numbers

    Some other explanation about the add function.
    """

def subtract(arg0: int, arg1: int) -> int:
    """
    Subtract two numbers

    Some other explanation about the subtract function.
    """

__version__: str = "0.2.5"
