from __future__ import annotations

import pytest

import networkx_graph as m
from networkx_graph import DiGraph, Node, Path


def test_version():
    assert m.__version__ == "0.0.8"


def test_add():
    assert m.add(1, 2) == 3


def test_sub():
    assert m.subtract(1, 2) == -1


def test_networkx():
    return
    """
    G = nx.DiGraph()
    G.add_node(node, length=100.0, key="value")
    G.add_edge(prev, curr)
    nx.freeze(G)

    prevs = list(G.predecessors(node))
    nexts = list(G.successors(node))
    # G.nodes.keys()
    # G.edges.keys()
    # G.edges[k]['path]
    print()
    """


def test_digraph_networkx():
    try:
        import networkx as nx
    except ImportError:
        return
    G0 = nx.DiGraph()
    way = G0.add_node("way1", length=15.0)
    assert way is None
    G0.add_node("way2", length=5.0, text="text", number=42, list=[4, 2])
    G0.add_node("way3", length=25.0)
    assert G0.nodes["way2"]["text"] == "text"
    for way in G0.nodes:
        assert isinstance(way, str)
    for way in G0.nodes.keys():  # noqa: SIM118
        assert isinstance(way, str)
    for way in G0.nodes.values():
        assert isinstance(way, dict)
        assert way["length"] in (15.0, 5.0, 25.0)
    for key, way in G0.nodes.items():
        assert isinstance(key, str)
        assert isinstance(way, dict)
    edge = G0.add_edge("way1", "way2")
    assert edge is None
    G0.add_edge("way1", "way3")
    for key, edge in G0.edges.items():
        assert isinstance(key, tuple)
        assert isinstance(edge, dict)


def test_digraph():
    test_digraph_networkx()

    node = Node()
    assert node.length == 1.0
    node.key = 777
    assert node.__dict__ == {"key": 777}
    assert node.to_dict() == {"length": 1.0, "key": 777}
    node.key = [1, 2, 3]
    assert node["key"] == [1, 2, 3]
    node.key.append(5)
    assert node["key"] == [1, 2, 3, 5]
    assert node.to_dict() == {"length": 1.0, "key": [1, 2, 3, 5]}
    node.to_dict()["key"].extend([7, 9])
    assert node.to_dict() == {"length": 1.0, "key": [1, 2, 3, 5, 7, 9]}

    node.to_dict()["new_key"] = "value"
    assert list(node.to_dict().keys()) == ["length", "key"]

    node["key"] = "value"
    node["num"] = 42
    assert node.key == "value"
    assert node.num == 42
    node.key = 3.14
    assert node["key"] == 3.14
    node.num = 123
    assert node["num"] == 123

    with pytest.raises(AttributeError):
        node.length = 5
    with pytest.raises(KeyError):
        node["length"] = 5

    G1 = DiGraph()
    way1 = G1.add_node("way1", length=15.0)
    way2 = G1.add_node("way2", length=5.0, text="text", number=42, list=[4, 2])
    assert way1.length == 15.0
    assert way2.length == 5.0
    assert way2.text == "text"
    assert way2.number == 42
    assert way2.list == [4, 2]
    assert G1.nodes["way1"] is way1

    assert not G1.edges
    edge = G1.add_edge("way1", "way2")
    assert ("way1", "way2") in G1.edges
    assert G1.edges[("way1", "way2")] is edge

    edge["key"] = "value"
    assert edge.to_dict() == {"key": "value"}
    assert edge.key == "value"
    edge.to_dict()["new_key"] = "value"
    assert edge.__dict__ == {"key": "value"}


def graph1(G=None):
    """
                             --------w3:10m---------o------------------w4:20m-------------o
                            /                                                             | w6:3m
    o---------w1:10m-------o-----------------w2:15m---------o---------------w5:15m--------o------------w7:10m--o

    """
    if G is None:
        G = DiGraph()
    G.add_node("w1", length=10.0)
    G.add_node("w2", length=15.0)
    G.add_node("w5", length=15.0)
    G.add_node("w3", length=10.0)
    G.add_node("w4", length=20.0)
    G.add_node("w6", length=3.0)
    G.add_node("w7", length=10.0)
    G.add_edge("w1", "w2")
    G.add_edge("w1", "w3")
    G.add_edge("w2", "w5")
    G.add_edge("w3", "w4")
    G.add_edge("w4", "w6")
    G.add_edge("w6", "w7")
    G.add_edge("w5", "w7")
    return G


def graph2(G=None):
    """
                             --------w3:10m---------o------------------w4:20m-------------+
                            /                                                             |
    o---------w1:10m-------o-----------------w2:15m---------o---------------w5:15m--------o------------w7:10m--o

    """
    if G is None:
        G = DiGraph()
    G.add_node("w1", length=10.0)
    G.add_node("w2", length=15.0)
    G.add_node("w5", length=15.0)
    G.add_node("w3", length=10.0)
    G.add_node("w4", length=20.0)
    G.add_node("w7", length=10.0)
    G.add_edge("w1", "w2")
    G.add_edge("w1", "w3")
    G.add_edge("w2", "w5")
    G.add_edge("w3", "w4")
    G.add_edge("w4", "w7")
    G.add_edge("w5", "w7")
    return G


def test_digraph_shortest_paths():
    G = graph1()

    assert set(G.successors("w1")) == {"w2", "w3"}
    assert set(G.predecessors("w7")) == {"w5", "w6"}

    shorts = G.shortest_paths_from("w1", cutoff=200.0)
    assert shorts.destinations() == [
        (0.0, "w2"),
        (0.0, "w3"),
        (10.0, "w4"),
        (15.0, "w5"),
        (30.0, "w6"),
        (30.0, "w7"),
    ]
    shorts = G.shortest_paths_from("w1", cutoff=200.0, offset=-1)
    assert shorts.destinations() == [
        (10.0, "w2"),
        (10.0, "w3"),
        (20.0, "w4"),
        (25.0, "w5"),
        (40.0, "w6"),
        (40.0, "w7"),
    ]
    shorts = G.shortest_paths_from("w1", cutoff=200.0, offset=3.0)
    assert shorts.destinations() == [
        (7.0, "w2"),
        (7.0, "w3"),
        (17.0, "w4"),
        (22.0, "w5"),
        (37.0, "w6"),
        (37.0, "w7"),
    ]
    shorts1 = G.shortest_paths_from("w1", cutoff=200.0, offset=10.0)
    shorts2 = G.shortest_paths_from("w1", cutoff=200.0, offset=13.0)
    assert shorts1.destinations() == shorts2.destinations()

    shorts = G.shortest_paths_to("w7", cutoff=20.0, offset=3.0)
    dists = shorts.destinations()
    assert dists == [(3.0, "w5"), (3.0, "w6"), (6.0, "w4"), (18.0, "w2")]
    assert (
        dists
        == G.shortest_paths_to(
            "w7",
            cutoff=18.0,
            offset=3.0,
        ).destinations()
    )
    assert (
        dists[:-1] == G.shortest_paths_to("w7", cutoff=17.0, offset=3.0).destinations()
    )

    path = G.shortest_path("w1", "w7", cutoff=37.0, source_offset=3.0)
    assert path is not None
    assert path.to_dict() == {
        "dist": 37.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", 3.0),
        "end": ("w7", None),
    }
    path = G.shortest_path("w1", "w7", cutoff=37.0 - 1e-3, source_offset=3.0)
    assert path is None
    path = G.shortest_path("w1", "w7", cutoff=30.0)
    assert path is not None
    assert path.to_dict() == {
        "dist": 30.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", None),
        "end": ("w7", None),
    }
    path = G.shortest_path("w1", "w7", cutoff=30.0 - 1e-3)
    assert path is None

    path = G.shortest_path("w1", "w7", cutoff=33, source_offset=9, target_offset=1)
    assert path.to_dict() == {
        "dist": 32.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", 9.0),
        "end": ("w7", 1.0),
    }

    path = G.shortest_path("w1", "w7", cutoff=40.0)
    assert path.nodes == ["w1", "w2", "w5", "w7"]
    assert path.to_dict() == {
        "dist": 30.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", None),
        "end": ("w7", None),
    }
    # take a detour
    path = G.shortest_path("w1", "w7", cutoff=40.0, sinks=G.encode_sinks({"w5"}))
    assert path.to_dict() == {
        "dist": 33.0,
        "nodes": ["w1", "w3", "w4", "w6", "w7"],
        "start": ("w1", None),
        "end": ("w7", None),
    }


def all_paths_from(G, start, cutoff):
    """python implementation"""
    assert start is not None
    assert cutoff >= 0
    output = []

    def backtrace(path, length):
        if length > cutoff:
            return
        nexts = list(G.successors(path[-1]))
        if not nexts:
            output.append((length, path))
            return
        if len(path) > 1:
            new_length = length + G.nodes[path[-1]]["length"]
            if new_length > cutoff:
                output.append((length, path))
                return
            length = new_length

        N = len(output)
        for nid in nexts:
            if nid in path:
                continue
            backtrace([*path, nid], length)
        if len(output) == N:
            output.append((length, path))

    backtrace([start], 0.0)
    output = sorted(output, key=lambda x: x[0])
    return [{"dist": round(d, 3), "nodes": p} for d, p in output]


def test_all_paths_from():
    try:
        import networkx as nx

        G = graph1(nx.DiGraph())
        paths = all_paths_from(G, "w1", 10.0)
        assert paths == [
            {"dist": 0.0, "nodes": ["w1", "w2"]},
            {"dist": 10.0, "nodes": ["w1", "w3", "w4"]},
        ]
    except ImportError:
        pass

    G = graph1()
    paths = G.all_paths_from("w1", cutoff=10.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {
            "dist": 10.0,
            "nodes": ["w1", "w2"],
            "start": ("w1", None),
            "end": ("w2", 10.0),
        },
        {
            "dist": 10.0,
            "nodes": ["w1", "w3", "w4"],
            "start": ("w1", None),
            "end": ("w4", 0.0),
        },
    ]
    paths = G.all_paths_from("w1", cutoff=10.0, sinks=G.encode_sinks({"w3"}))
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {
            "dist": 10.0,
            "nodes": ["w1", "w2"],
            "start": ("w1", None),
            "end": ("w2", 10.0),
        },
        {
            "dist": 10.0,
            "nodes": ["w1", "w3"],
            "start": ("w1", None),
            "end": ("w3", 10.0),
        },
    ]

    G = graph1()
    paths = G.all_paths_from("w1", cutoff=5.0, offset=2.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {
            "dist": 5.0,
            "nodes": ["w1"],
            "start": ("w1", 2.0),
            "end": ("w1", 7.0),
        }
    ]
    paths = G.all_paths_from("w1", cutoff=15.0, offset=2.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {"dist": 15.0, "nodes": ["w1", "w2"], "start": ("w1", 2.0), "end": ("w2", 7.0)},
        {"dist": 15.0, "nodes": ["w1", "w3"], "start": ("w1", 2.0), "end": ("w3", 7.0)},
    ]

    paths = G.all_paths_from("w1", cutoff=25.0, offset=5.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {
            "dist": 25.0,
            "nodes": ["w1", "w2", "w5"],
            "start": ("w1", 5.0),
            "end": ("w5", 5.0),
        },
        {
            "dist": 25.0,
            "nodes": ["w1", "w3", "w4"],
            "start": ("w1", 5.0),
            "end": ("w4", 10.0),
        },
    ]

    paths = G.all_paths_from("w1", cutoff=5.12345, offset=2.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {"dist": 5.123, "nodes": ["w1"], "start": ("w1", 2.0), "end": ("w1", 7.123)}
    ]

    assert G.round_n == 3
    assert G.round_scale == 1e3

    G = DiGraph(round_n=None)
    assert G.round_n is None
    assert G.round_scale is None
    G = graph1(G)
    paths = G.all_paths_from("w1", cutoff=5.12345, offset=2.0)
    paths = [r.to_dict() for r in paths]
    assert [
        {"dist": 5.12345, "nodes": ["w1"], "start": ("w1", 2.0), "end": ("w1", 7.12345)}
    ]

    G = DiGraph(round_n=-1)
    assert G.round_n == -1
    assert G.round_scale == 0.1
    G = graph1(G)
    paths = G.all_paths_from("w1", cutoff=5.12345, offset=2.0)
    paths = [r.to_dict() for r in paths]
    assert [{"dist": 10.0, "nodes": ["w1"], "start": ("w1", 0.0), "end": ("w1", 10.0)}]


def test_all_paths_to():
    G = graph1()
    paths = G.all_paths_to("w7", cutoff=30.0, offset=4.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {
            "dist": 30.0,
            "nodes": ["w3", "w4", "w6", "w7"],
            "start": ("w3", 7.0),
            "end": ("w7", 4.0),
        },
        {
            "dist": 30.0,
            "nodes": ["w2", "w5", "w7"],
            "start": ("w2", 4.0),
            "end": ("w7", 4.0),
        },
    ]
    paths = G.all_paths_to("w7", cutoff=30.0)
    paths = [r.to_dict() for r in paths]
    assert paths == [
        {
            "dist": 30.0,
            "nodes": ["w3", "w4", "w6", "w7"],
            "start": ("w3", 3.0),
            "end": ("w7", None),
        },
        {
            "dist": 30.0,
            "nodes": ["w1", "w2", "w5", "w7"],
            "start": ("w1", 10.0),
            "end": ("w7", None),
        },
    ]


def test_all_paths():
    G = graph2()

    paths = G.all_paths("w1", "w1", cutoff=20)
    assert not paths  # skip trivial
    paths = G.all_paths("w1", "w1", cutoff=20, source_offset=3.0, target_offset=4.0)
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 1.0,
        "nodes": ["w1"],
        "start": ("w1", 3.0),
        "end": ("w1", 4.0),
    }
    paths = G.all_paths("w1", "w1", cutoff=20, source_offset=13.0, target_offset=14.0)
    assert not paths

    paths = G.all_paths("w1", "w3", cutoff=10)
    assert {
        "dist": 0.0,
        "nodes": ["w1", "w3"],
        "start": ("w1", None),
        "end": ("w3", None),
    }
    paths = G.all_paths("w1", "w4", cutoff=10)
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 10.0,
        "nodes": ["w1", "w3", "w4"],
        "start": ("w1", None),
        "end": ("w4", None),
    }
    paths = G.all_paths("w1", "w4", cutoff=9)
    assert not paths

    paths = G.all_paths("w1", "w4", cutoff=20, target_offset=5)
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 15.0,
        "nodes": ["w1", "w3", "w4"],
        "start": ("w1", None),
        "end": ("w4", 5.0),
    }
    paths = G.all_paths("w1", "w4", cutoff=14, target_offset=5)
    assert not paths
    paths = G.all_paths("w1", "w4", cutoff=20, source_offset=8, target_offset=5)
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 17.0,
        "nodes": ["w1", "w3", "w4"],
        "start": ("w1", 8.0),
        "end": ("w4", 5.0),
    }

    paths = G.all_paths("w1", "w7", cutoff=80)
    assert len(paths) == 2
    paths = [r.to_dict() for r in paths]
    r1 = {
        "dist": 30.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", None),
        "end": ("w7", None),
    }
    r2 = {
        "dist": 30.0,
        "nodes": ["w1", "w3", "w4", "w7"],
        "start": ("w1", None),
        "end": ("w7", None),
    }
    assert paths in ([r1, r2], [r2, r1])

    paths = G.all_paths("w1", "w7", cutoff=80, source_offset=3.0, target_offset=4.0)
    assert len(paths) == 2
    paths = [r.to_dict() for r in paths]
    r1 = {
        "dist": 41.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", 3.0),
        "end": ("w7", 4.0),
    }
    r2 = {
        "dist": 41.0,
        "nodes": ["w1", "w3", "w4", "w7"],
        "start": ("w1", 3.0),
        "end": ("w7", 4.0),
    }
    assert paths in ([r1, r2], [r2, r1])

    paths = G.all_paths(
        "w1",
        "w7",
        cutoff=80,
        source_offset=3.0,
        target_offset=4.0,
        sinks=G.encode_sinks({"w4"}),
    )
    assert len(paths) == 1
    assert paths[0].to_dict() == r1


def test_routing():
    with pytest.raises(TypeError) as excinfo:
        Path()
    assert "No constructor defined" in repr(excinfo.value)
    G = graph1()

    obj = {"key": "value"}
    bindings = G.encode_bindings({"w3": [], "w2": [(3.0, 4, "text"), (8, 10, obj)]})
    decoded = bindings()
    assert decoded == {"w2": [(3.0, 4.0, "text"), (8.0, 10.0, obj)], "w3": []}
    assert decoded["w2"][-1][-1] is obj
    decoded["w2"][-1][-1]["num"] = 42
    assert obj["num"] == 42

    dists = G.shortest_paths_from("w1", cutoff=20.0).destinations()
    assert dists == [(0.0, "w2"), (0.0, "w3"), (10.0, "w4"), (15.0, "w5")]

    sinks = G.encode_sinks({"w2", "w3"})
    assert sinks() == {"w2", "w3"}

    dists = G.shortest_paths_from("w1", cutoff=20.0, sinks=sinks).destinations()
    assert dists == [(0.0, "w2"), (0.0, "w3")]

    path_generator = G.shortest_paths_from("w1", cutoff=20.0, sinks=sinks)
    assert path_generator.destinations() == [(0.0, "w2"), (0.0, "w3")]
    assert path_generator.to_dict() == {"cutoff": 20.0, "source": ("w1", None)}

    sinks = G.encode_sinks({"w6"})
    path_generator = G.shortest_paths_from(
        "w1",
        cutoff=20.0,
        offset=5.0,
        sinks=sinks,
    )
    assert path_generator.destinations() == [
        (5.0, "w2"),
        (5.0, "w3"),
        (15.0, "w4"),
        (20.0, "w5"),
    ]
    assert path_generator.to_dict() == {"cutoff": 20.0, "source": ("w1", 5.0)}

    path_generator = G.shortest_paths_from(
        "w1",
        cutoff=80.0,
        offset=5.0,
        sinks=sinks,
    )
    assert path_generator.destinations() == [
        (5.0, "w2"),
        (5.0, "w3"),
        (15.0, "w4"),
        (20.0, "w5"),
        (35.0, "w6"),
        (35.0, "w7"),
    ]
    assert path_generator.prevs() == {
        "w2": "w1",
        "w3": "w1",
        "w4": "w3",
        "w5": "w2",
        "w6": "w4",
        "w7": "w5",
    }
    assert path_generator.dists() == {
        "w2": 5.0,
        "w3": 5.0,
        "w4": 15.0,
        "w5": 20.0,
        "w6": 35.0,
        "w7": 35.0,
    }
    assert path_generator.source() == ("w1", 5.0)
    assert path_generator.target() is None

    paths = path_generator.paths()
    assert len(paths) == 2
    assert path_generator.cutoff() == 80.0

    assert paths[0].to_dict() == {
        "dist": 45.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", 5.0),
        "end": ("w7", 10.0),
    }
    assert paths[1].to_dict() == {
        "dist": 38.0,
        "nodes": ["w1", "w3", "w4", "w6"],
        "start": ("w1", 5.0),
        "end": ("w6", 3.0),
    }

    path_generator = G.shortest_paths_from(
        "w1",
        cutoff=2.0,
        offset=6.0,
    )
    paths = path_generator.paths()
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 2.0,
        "nodes": ["w1"],
        "start": ("w1", 6.0),
        "end": ("w1", 8.0),
    }
    path_generator = G.shortest_paths_to(
        "w1",
        cutoff=2.0,
        offset=6.0,
    )
    paths = path_generator.paths()
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 2.0,
        "nodes": ["w1"],
        "start": ("w1", 4.0),
        "end": ("w1", 6.0),
    }

    path_generator = G.shortest_paths_from(
        "w1",
        cutoff=40.0,
        offset=6.000001,
        sinks=sinks,
    )
    paths = path_generator.paths()
    assert len(paths) == 2
    assert paths[0].to_dict() == {
        "dist": 40.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", 6.0),
        "end": ("w7", 6.0),
    }
    assert paths[1].to_dict() == {
        "dist": 37.0,
        "nodes": ["w1", "w3", "w4", "w6"],
        "start": ("w1", 6.0),
        "end": ("w6", 3.0),
    }

    path_generator = G.shortest_paths_from(
        "w7",
        cutoff=20.0,
        offset=3.0,
    )
    paths = path_generator.paths()
    assert len(paths) == 1
    assert paths[0].to_dict() == {
        "dist": 7.0,
        "nodes": ["w7"],
        "start": ("w7", 3.0),
        "end": ("w7", 10.0),
    }

    path_generator = G.shortest_paths_to(
        "w7",
        cutoff=20.0,
        offset=3.0,
    )
    paths = [r.to_dict() for r in path_generator.paths()]
    assert len(paths) == 2
    path1 = {
        "dist": 20.0,
        "nodes": ["w2", "w5", "w7"],
        "start": ("w2", 13.0),
        "end": ("w7", 3.0),
    }
    path2 = {
        "dist": 20.0,
        "nodes": ["w4", "w6", "w7"],
        "start": ("w4", 6.0),
        "end": ("w7", 3.0),
    }
    assert paths in [[path1, path2], [path2, path1]]
    assert path_generator.to_dict() == {"cutoff": 20.0, "target": ("w7", 3.0)}

    assert path_generator.path("w5").to_dict() == {
        "dist": 18.0,
        "nodes": ["w5", "w7"],
        "start": ("w5", 0.0),
        "end": ("w7", 3.0),
    }
    assert path_generator.path("w6").to_dict() == {
        "dist": 6.0,
        "nodes": ["w6", "w7"],
        "start": ("w6", 0.0),
        "end": ("w7", 3.0),
    }
    assert path_generator.prevs() == {"w2": "w5", "w4": "w6", "w5": "w7", "w6": "w7"}
    assert path_generator.dists() == {"w2": 18.0, "w4": 6.0, "w5": 3.0, "w6": 3.0}
    assert path_generator.path("w7") is None

    path_generator = G.shortest_paths_from(
        "w1",
        cutoff=80.0,
        offset=6.0,
    )
    assert len(path_generator.paths()) == 2

    G = graph2()
    path_generator = G.shortest_paths_from(
        "w1",
        cutoff=80.0,
        offset=6.0,
    )
    assert len(path_generator.paths()) == 2
    destinations = [r.end for r in path_generator.paths()]
    assert ("w7", 10.0) in destinations
    assert ("w5", 15.0) in destinations or ("w4", 20.0) in destinations


def test_shortest_path_to_bindings():
    G = graph1()
    obj1 = {}
    bindings = G.encode_bindings(
        {
            "w3": [(1, 3, obj1)],
            "w7": [(3, 4, "obj2")],
        }
    )
    backwards, forwards = G.shortest_path_to_bindings(
        "w1",
        cutoff=50.0,
        bindings=bindings,
    )
    assert backwards is None
    forwards = forwards.to_dict()
    assert forwards == {
        "dist": 1.0,
        "nodes": ["w1", "w3"],
        "start": ("w1", None),
        "end": ("w3", 1.0),
        "binding": ("w3", (1.0, 3.0, obj1)),
    }
    forwards["binding"][-1][-1]["key"] = "value"
    assert obj1 == {"key": "value"}

    _, forwards2 = G.shortest_path_to_bindings(
        "w1", cutoff=50.0, bindings=bindings, sinks=G.encode_sinks({"w3"})
    )
    assert forwards2.to_dict() == forwards
    _, forwards = G.shortest_path_to_bindings(
        "w3",
        cutoff=50.0,
        bindings=bindings,
    )
    assert forwards.to_dict() == {
        "dist": 26.0,
        "nodes": ["w3", "w4", "w6", "w7"],
        "start": ("w3", None),
        "end": ("w7", 3.0),
        "binding": ("w7", (3.0, 4.0, "obj2")),
    }

    _, forwards = G.shortest_path_to_bindings(
        "w3", cutoff=50.0, bindings=bindings, offset=1.0
    )
    assert forwards.to_dict() == {
        "dist": 0.0,
        "nodes": ["w3"],
        "start": ("w3", 1.0),
        "end": ("w3", 1.0),
        "binding": ("w3", (1.0, 3.0, {"key": "value"})),
    }

    _, forwards = G.shortest_path_to_bindings(
        "w3", cutoff=50.0, bindings=bindings, offset=1.0 + 1e-8
    )
    assert forwards.to_dict() == {
        "dist": 35.0,
        "nodes": ["w3", "w4", "w6", "w7"],
        "start": ("w3", 1.0),
        "end": ("w7", 3.0),
        "binding": ("w7", (3.0, 4.0, "obj2")),
    }

    backwards, forwards = G.shortest_path_to_bindings(
        "w3",
        cutoff=50.0,
        bindings=bindings,
        offset=5.0,
    )
    assert backwards.to_dict() == {
        "dist": 2.0,
        "nodes": ["w3"],
        "start": ("w3", 3.0),
        "end": ("w3", 5.0),
        "binding": ("w3", (1.0, 3.0, {"key": "value"})),
    }
    assert forwards.to_dict() == {
        "dist": 31.0,
        "nodes": ["w3", "w4", "w6", "w7"],
        "start": ("w3", 5.0),
        "end": ("w7", 3.0),
        "binding": ("w7", (3.0, 4.0, "obj2")),
    }

    backwards, forwards = G.shortest_path_to_bindings(
        "w3",
        cutoff=50.0,
        bindings=bindings,
        offset=5.0,
        direction=1,  # forwards
    )
    assert backwards is None
    assert forwards is not None

    backwards, forwards = G.shortest_path_to_bindings(
        "w3",
        cutoff=50.0,
        bindings=bindings,
        offset=5.0,
        direction=-1,  # backwards
    )
    assert backwards is not None
    assert forwards is None

    backwards, forwards = G.shortest_path_to_bindings(
        "w3",
        cutoff=2.0,
        bindings=bindings,
        offset=5.0,
    )
    assert backwards is not None
    assert forwards is None
    backwards, forwards = G.shortest_path_to_bindings(
        "w3",
        cutoff=2.0 - 1e-3,
        bindings=bindings,
        offset=5.0,
    )
    assert backwards is None
    assert forwards is None

    backwards, forwards = G.shortest_path_to_bindings(
        "w4",
        cutoff=30,
        bindings=bindings,
    )
    assert forwards.to_dict() == {
        "dist": 6.0,
        "nodes": ["w4", "w6", "w7"],
        "start": ("w4", None),
        "end": ("w7", 3.0),
        "binding": ("w7", (3.0, 4.0, "obj2")),
    }
    assert backwards.to_dict() == {
        "dist": 7.0,
        "nodes": ["w3", "w4"],
        "start": ("w3", 3.0),
        "end": ("w4", None),
        "binding": ("w3", (1.0, 3.0, {"key": "value"})),
    }

    backwards, forwards = G.shortest_path_to_bindings(
        "w7",
        cutoff=30,
        bindings=G.encode_bindings(
            {
                "w3": [(3, 8, "obj3")],
                "w2": [(2, 3, "obj4")],
                "w5": [(8, 8, "obj5")],
            }
        ),
    )
    assert forwards is None
    assert backwards.to_dict() == {
        "dist": 7.0,
        "nodes": ["w5", "w7"],
        "start": ("w5", 8.0),
        "end": ("w7", None),
        "binding": ("w5", (8.0, 8.0, "obj5")),
    }


def test_all_paths_to_bindings():
    G = graph1()
    bindings = G.encode_bindings(
        {
            "w1": [(4, 4, "obj1")],
            "w3": [(1, 3, "obj31"), (5, 6, "obj32"), (9, 10, "obj33")],
            "w7": [(3, 4, "obj7")],
        }
    )
    backwards, forwards = G.all_paths_to_bindings(
        "w3",
        cutoff=30,
        offset=5.5,
        bindings=bindings,
    )
    assert len(backwards) == 1
    assert len(forwards) == 1
    assert backwards[0].to_dict() == {
        "dist": 2.5,
        "nodes": ["w3"],
        "start": ("w3", 3.0),
        "end": ("w3", 5.5),
        "binding": ("w3", (1.0, 3.0, "obj31")),
    }
    assert forwards[0].to_dict() == {
        "dist": 3.5,
        "nodes": ["w3"],
        "start": ("w3", 5.5),
        "end": ("w3", 9.0),
        "binding": ("w3", (9.0, 10.0, "obj33")),
    }

    backwards, forwards = G.all_paths_to_bindings(
        "w4",
        cutoff=30,
        bindings=bindings,
    )
    assert len(forwards) == 1
    assert forwards[0].to_dict() == {
        "dist": 6.0,
        "nodes": ["w4", "w6", "w7"],
        "start": ("w4", None),
        "end": ("w7", 3.0),
        "binding": ("w7", (3.0, 4.0, "obj7")),
    }
    assert len(backwards) == 1
    assert backwards[0].to_dict() == {
        "dist": 0.0,
        "nodes": ["w3", "w4"],
        "start": ("w3", 10.0),
        "end": ("w4", None),
        "binding": ("w3", (9.0, 10.0, "obj33")),
    }

    backwards, forwards = G.all_paths_to_bindings(
        "w7",
        cutoff=80,
        offset=1.0,
        bindings=bindings,
    )
    assert len(forwards) == 1
    assert forwards[0].to_dict() == {
        "dist": 2.0,
        "nodes": ["w7"],
        "start": ("w7", 1.0),
        "end": ("w7", 3.0),
        "binding": ("w7", (3.0, 4.0, "obj7")),
    }
    assert len(backwards) == 2
    assert backwards[0].to_dict() == {
        "dist": 24.0,
        "nodes": ["w3", "w4", "w6", "w7"],
        "start": ("w3", 10.0),
        "end": ("w7", 1.0),
        "binding": ("w3", (9.0, 10.0, "obj33")),
    }
    assert backwards[1].to_dict() == {
        "dist": 37.0,
        "nodes": ["w1", "w2", "w5", "w7"],
        "start": ("w1", 4.0),
        "end": ("w7", 1.0),
        "binding": ("w1", (4.0, 4.0, "obj1")),
    }

    backwards, forwards = G.all_paths_to_bindings(
        "w7",
        cutoff=80,
        offset=1.0,
        bindings=bindings,
        direction=1,
    )
    assert len(backwards) == 0
    assert len(forwards) == 1
    backwards, forwards = G.all_paths_to_bindings(
        "w7",
        cutoff=80,
        offset=1.0,
        bindings=bindings,
        direction=-1,
    )
    assert len(backwards) == 2
    assert len(forwards) == 0
