from __future__ import annotations

import json

import pytest

import networkx_graph as m
from networkx_graph import DiGraph, Node, rapidjson


def test_version():
    assert m.__version__ == "0.0.5"


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


def test_rapidjson():
    j = rapidjson()
    assert j.dumps() == "null"
    assert json.dumps(None) == "null"
    j = rapidjson({})
    assert j.dumps() == "{}"
    j = rapidjson([])
    assert j.dumps() == "[]"
    assert rapidjson(5).dumps() == "5"
    assert rapidjson(3.14).dumps() == "3.14"
    assert rapidjson("text").dumps() == '"text"'
    for text in [
        "3.14",
        "5",
        '"text"',
        '{"key": "value"}',
        '["list", "items"]',
    ]:
        assert rapidjson().loads(text)() == json.loads(text)


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


def test_digraph_dijkstra():
    G = graph1()

    assert set(G.successors("w1")) == {"w2", "w3"}
    assert set(G.predecessors("w7")) == {"w5", "w6"}

    dists = G.single_source_dijkstra("w1", cutoff=200.0)
    assert dists == [
        (0.0, "w2"),
        (0.0, "w3"),
        (10.0, "w4"),
        (15.0, "w5"),
        (30.0, "w6"),
        (30.0, "w7"),
    ]
    dists = G.single_source_dijkstra("w1", cutoff=200.0, offset=-1)
    assert dists == [
        (10.0, "w2"),
        (10.0, "w3"),
        (20.0, "w4"),
        (25.0, "w5"),
        (40.0, "w6"),
        (40.0, "w7"),
    ]
    dists = G.single_source_dijkstra("w1", cutoff=200.0, offset=3.0)
    assert dists == [
        (7.0, "w2"),
        (7.0, "w3"),
        (17.0, "w4"),
        (22.0, "w5"),
        (37.0, "w6"),
        (37.0, "w7"),
    ]
    dists1 = G.single_source_dijkstra("w1", cutoff=200.0, offset=10.0)
    dists2 = G.single_source_dijkstra("w1", cutoff=200.0, offset=13.0)
    assert dists1 == dists2

    dists = G.single_source_dijkstra("w7", cutoff=20.0, offset=3.0, reverse=True)
    assert dists == [(3.0, "w5"), (3.0, "w6"), (6.0, "w4"), (18.0, "w2")]
    assert dists == G.single_source_dijkstra(
        "w7", cutoff=18.0, offset=3.0, reverse=True
    )
    assert dists[:-1] == G.single_source_dijkstra(
        "w7", cutoff=17.0, offset=3.0, reverse=True
    )


def all_routes_from(G, start, cutoff):
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
    return [{"dist": round(d, 3), "path": p} for d, p in output]


def test_all_routes_from():
    try:
        import networkx as nx

        G = graph1(nx.DiGraph())
        routes = all_routes_from(G, "w1", 10.0)
        assert routes == [
            {"dist": 0.0, "path": ["w1", "w2"]},
            {"dist": 10.0, "path": ["w1", "w3", "w4"]},
        ]
    except ImportError:
        pass

    G = graph1()
    routes = G.all_routes_from("w1", cutoff=10.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {
            "dist": 10.0,
            "path": ["w1", "w2"],
            "start": ("w1", None),
            "end": ("w2", 10.0),
        },
        {
            "dist": 10.0,
            "path": ["w1", "w3", "w4"],
            "start": ("w1", None),
            "end": ("w4", 0.0),
        },
    ]

    G = graph1()
    routes = G.all_routes_from("w1", cutoff=5.0, offset=2.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {
            "dist": 5.0,
            "path": ["w1"],
            "start": ("w1", 2.0),
            "end": ("w1", 7.0),
        }
    ]
    routes = G.all_routes_from("w1", cutoff=15.0, offset=2.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {"dist": 15.0, "path": ["w1", "w2"], "start": ("w1", 2.0), "end": ("w2", 7.0)},
        {"dist": 15.0, "path": ["w1", "w3"], "start": ("w1", 2.0), "end": ("w3", 7.0)},
    ]

    routes = G.all_routes_from("w1", cutoff=25.0, offset=5.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {
            "dist": 25.0,
            "path": ["w1", "w2", "w5"],
            "start": ("w1", 5.0),
            "end": ("w5", 5.0),
        },
        {
            "dist": 25.0,
            "path": ["w1", "w3", "w4"],
            "start": ("w1", 5.0),
            "end": ("w4", 10.0),
        },
    ]

    routes = G.all_routes_from("w1", cutoff=5.12345, offset=2.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {"dist": 5.123, "path": ["w1"], "start": ("w1", 2.0), "end": ("w1", 7.123)}
    ]

    assert G.round_n == 3
    assert G.round_scale == 1e3

    G = DiGraph(round_n=None)
    assert G.round_n is None
    assert G.round_scale is None
    G = graph1(G)
    routes = G.all_routes_from("w1", cutoff=5.12345, offset=2.0)
    routes = [r.to_dict() for r in routes]
    assert [
        {"dist": 5.12345, "path": ["w1"], "start": ("w1", 2.0), "end": ("w1", 7.12345)}
    ]

    G = DiGraph(round_n=-1)
    assert G.round_n == -1
    assert G.round_scale == 0.1
    G = graph1(G)
    routes = G.all_routes_from("w1", cutoff=5.12345, offset=2.0)
    routes = [r.to_dict() for r in routes]
    assert [{"dist": 10.0, "path": ["w1"], "start": ("w1", 0.0), "end": ("w1", 10.0)}]


def test_all_routes_to():
    G = graph1()
    routes = G.all_routes_to("w7", cutoff=30.0, offset=4.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {
            "dist": 30.0,
            "path": ["w3", "w4", "w6", "w7"],
            "start": ("w3", 7.0),
            "end": ("w7", 4.0),
        },
        {
            "dist": 30.0,
            "path": ["w2", "w5", "w7"],
            "start": ("w2", 4.0),
            "end": ("w7", 4.0),
        },
    ]
    routes = G.all_routes_to("w7", cutoff=30.0)
    routes = [r.to_dict() for r in routes]
    assert routes == [
        {
            "dist": 30.0,
            "path": ["w3", "w4", "w6", "w7"],
            "start": ("w3", 3.0),
            "end": ("w7", None),
        },
        {
            "dist": 30.0,
            "path": ["w1", "w2", "w5", "w7"],
            "start": ("w1", 10.0),
            "end": ("w7", None),
        },
    ]
