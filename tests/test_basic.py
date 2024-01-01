from __future__ import annotations

import json

import pytest

import networkx_graph as m
from networkx_graph import DiGraph, Node, rapidjson


def test_version():
    assert m.__version__ == "0.0.2"


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
    way1 = G0.add_node("way1", length=15.0)
    way2 = G0.add_node("way2", length=5.0, text="text", number=42, list=[4, 2])
    way3 = G0.add_node("way3", length=25.0)
    for way in G0.nodes:
        assert isinstance(way, str)
    for way in G0.nodes.keys():
        assert isinstance(way, str)
    for way in G0.nodes.values():
        assert isinstance(way, dict)
        assert way['length'] in (15.0, 5.0, 25.0)
    for key, way in G0.nodes.items():
        assert isinstance(key, str)
        assert isinstance(way, dict)
    edge1 = G0.add_edge("way1", "way2")
    edge2 = G0.add_edge("way1", "way3")
    for key, edge in G0.edges.items():
        assert isinstance(key, tuple)
        assert isinstance(edge, dict)
    print()

def test_digraph():
    test_digraph_networkx()

    node = Node()
    assert node.length == 1.0
    node.key = 777
    assert node.__dict__ == {"key": 777}
    node.key = [1, 2, 3]
    assert node["key"] == [1, 2, 3]
    node.key.append(5)
    assert node["key"] == [1, 2, 3, 5]

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

    print()

def test_digraph_dijkstra():
    print()

test_digraph()
print()