#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <fstream>
#include <iostream>
#include <set>

#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#define DBG_MACRO_NO_WARNING
#include "dbg.h"

#include "heap.hpp"
#include "indexer.hpp"
#include "types.hpp"

#include "spdlog/spdlog.h"
// fix exposed macro 'GetObject' from wingdi.h (included by spdlog.h) under
// windows, see https://github.com/Tencent/rapidjson/issues/1448
#ifdef GetObject
#undef GetObject
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using rvp = py::return_value_policy;
using namespace pybind11::literals;

template <class T> inline void hash_combine(std::size_t &seed, const T &value)
{
    seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
template <typename... T> struct hash<tuple<T...>>
{
    size_t operator()(const tuple<T...> &t) const
    {
        size_t seed = 0;
        hash_tuple(seed, t);
        return seed;
    }

  private:
    template <std::size_t I = 0, typename... Ts>
    inline typename enable_if<I == sizeof...(Ts), void>::type
    hash_tuple(size_t &seed, const tuple<Ts...> &t) const
    {
    }

    template <std::size_t I = 0, typename... Ts>
        inline typename enable_if <
        I<sizeof...(Ts), void>::type hash_tuple(size_t &seed,
                                                const tuple<Ts...> &t) const
    {
        hash_combine(seed, get<I>(t));
        hash_tuple<I + 1, Ts...>(seed, t);
    }
};
} // namespace std

namespace nano_fmm
{

struct Node
{
    double length{1.0};
};

struct Edge
{
};

struct DiGraph;

inline double ROUND(double v, double s)
{
    return std::floor(v * s + 0.5) / s; // not same std::round(v * s) / s;
}

struct Route
{
    Route() = default;
    Route(const DiGraph *graph = nullptr, double dist = 0.0,
          const std::vector<int64_t> &path = {},
          std::optional<double> start_offset = {},
          std::optional<double> end_offset = {})
        : graph(graph), dist(dist), path(path), start_offset(start_offset),
          end_offset(end_offset)
    {
    }
    const DiGraph *graph{nullptr};
    double dist{0.0};
    std::vector<int64_t> path;
    std::optional<double> start_offset;
    std::optional<double> end_offset;

    void round(double scale)
    {
        dist = ROUND(dist, scale);
        if (start_offset) {
            start_offset = ROUND(*start_offset, scale);
        }
        if (end_offset) {
            end_offset = ROUND(*end_offset, scale);
        }
    }
};

struct DiGraph
{
    DiGraph(std::optional<int8_t> round_n = 3)
    {
        if (round_n) {
            round_scale_ = std::pow(10.0, *round_n);
        }
    }
    std::optional<int8_t> round_n() const
    {
        if (!round_scale_) {
            return {};
        }
        return static_cast<int8_t>(ROUND(std::log10(*round_scale_), 1.0));
    }
    std::optional<double> round_scale() const { return round_scale_; }

    Node &add_node(const std::string &id, double length = 1.0)
    {
        reset();
        if (round_scale_) {
            length = ROUND(length, *round_scale_);
        }
        auto idx = indexer_.id(id);
        auto &node = nodes_[idx];
        node.length = length;
        lengths_[idx] = length;
        return node;
    }
    Edge &add_edge(const std::string &node0, const std::string &node1)
    {
        reset();
        auto idx0 = indexer_.id(node0);
        auto idx1 = indexer_.id(node1);
        nexts_[idx0].insert(idx1);
        prevs_[idx1].insert(idx0);
        lengths_[idx0] = nodes_[idx0].length;
        lengths_[idx1] = nodes_[idx1].length;
        auto &edge = edges_[std::make_tuple(idx0, idx1)];
        return edge;
    }

    const std::unordered_map<std::string, Node *> &nodes() const
    {
        return cache().nodes;
    }
    const std::unordered_map<std::tuple<std::string, std::string>, Edge *> &
    edges() const
    {
        return cache().edges;
    }

    std::vector<std::string> predecessors(const std::string &id) const
    {
        return __nexts(id, prevs_);
    }
    std::vector<std::string> successors(const std::string &id) const
    {
        return __nexts(id, nexts_);
    }

    std::string __node_id(int64_t node) const { return indexer_.id(node); }
    std::vector<std::string> __node_ids(const std::vector<int64_t> &nodes) const
    {
        std::vector<std::string> ids;
        ids.reserve(nodes.size());
        for (auto node : nodes) {
            ids.push_back(indexer_.id(node));
        }
        return ids;
    }

    std::vector<std::tuple<double, std::string>> single_source_dijkstra(
        const std::string &start, double cutoff,
        std::optional<double> offset = {},
        const std::unordered_set<std::string> *sinks = nullptr,
        std::unordered_map<std::string, std::string> *prevs = nullptr,
        bool reverse = false) const
    {
        if (cutoff < 0) {
            return {};
        }
        auto start_idx = indexer_.get_id(start);
        if (!start_idx) {
            return {};
        }
        auto length = lengths_.find(*start_idx);
        if (length == lengths_.end()) {
            return {};
        }
        if (!offset) {
            offset = 0.0;
        } else {
            if (reverse) {
                offset = std::max(0.0, *offset);
            } else {
                offset = std::max(0.0, length->second - *offset);
            }
        }
        std::unique_ptr<unordered_set<int64_t>> sinks_ptr;
        if (sinks) {
            sinks_ptr = std::make_unique<unordered_set<int64_t>>();
            for (auto &node : *sinks) {
                auto nid = indexer_.get_id(node);
                if (nid) {
                    sinks_ptr->insert(std::move(*nid));
                }
            }
            if (sinks_ptr->empty()) {
                sinks_ptr.reset();
            }
        }
        unordered_map<int64_t, int64_t> pmap;
        unordered_map<int64_t, double> dmap;
        single_source_dijkstra(*start_idx, cutoff, reverse ? prevs_ : nexts_,
                               pmap, dmap, sinks_ptr.get(), *offset);
        if (prevs) {
            for (const auto &pair : pmap) {
                (*prevs)[indexer_.id(pair.first)] = indexer_.id(pair.second);
            }
        }
        auto ret = std::vector<std::tuple<double, std::string>>{};
        ret.reserve(dmap.size());
        for (auto &pair : dmap) {
            ret.emplace_back(
                std::make_tuple(pair.second, indexer_.id(pair.first)));
        }
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    std::vector<Route> all_routes_from(const std::string &source, double cutoff,
                                       std::optional<double> offset = {}) const
    {
        if (cutoff < 0) {
            return {};
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        auto routes = __all_routes(*src_idx, cutoff, offset, lengths_, nexts_);
        if (round_scale_) {
            for (auto &r : routes) {
                r.round(*round_scale_);
            }
        }
        return routes;
    }

    std::vector<Route> all_routes_to(const std::string &target, double cutoff,
                                     std::optional<double> offset = {}) const
    {
        if (cutoff < 0) {
            return {};
        }
        auto dst_idx = indexer_.get_id(target);
        if (!dst_idx) {
            return {};
        }
        auto length = lengths_.find(*dst_idx);
        if (length == lengths_.end()) {
            return {};
        }
        if (offset) {
            offset = std::max(0.0, std::min(*offset, length->second));
            offset = length->second - *offset;
        }
        auto routes = __all_routes(*dst_idx, cutoff, offset, lengths_, prevs_);
        for (auto &r : routes) {
            if (r.start_offset) {
                r.start_offset = lengths_.at(r.path.front()) - *r.start_offset;
            }
            if (r.end_offset) {
                r.end_offset = lengths_.at(r.path.back()) - *r.end_offset;
            }
            std::reverse(r.path.begin(), r.path.end());
            std::swap(r.start_offset, r.end_offset);
        }
        if (round_scale_) {
            for (auto &r : routes) {
                r.round(*round_scale_);
            }
        }
        return routes;
    }

    DiGraph &from_rapidjson(const RapidjsonValue &json) { return *this; }
    RapidjsonValue to_rapidjson(RapidjsonAllocator &allocator) const
    {
        RapidjsonValue json(rapidjson::kObjectType);
        return json;
    }
    RapidjsonValue to_rapidjson() const
    {
        RapidjsonAllocator allocator;
        return to_rapidjson(allocator);
    }

    void freeze() { freezed_ = true; }
    void build() const {}
    void reset() const
    {
        if (freezed_) {
            throw std::logic_error("can't reset when freezed");
        }
        cache_.reset();
    }

    Indexer &indexer() { return indexer_; }

  private:
    bool freezed_{false};
    std::optional<double> round_scale_;
    std::unordered_map<int64_t, Node> nodes_;
    std::unordered_map<std::tuple<int64_t, int64_t>, Edge> edges_;
    unordered_map<int64_t, double> lengths_;
    unordered_map<int64_t, unordered_set<int64_t>> nexts_, prevs_;
    mutable Indexer indexer_;
    struct Cache
    {
        std::unordered_map<std::string, Node *> nodes;
        std::unordered_map<std::tuple<std::string, std::string>, Edge *> edges;
    };
    mutable std::optional<Cache> cache_;
    Cache &cache() const
    {
        if (cache_) {
            return *cache_;
        }
        cache_ = Cache();
        for (auto &pair : nodes_) {
            cache_->nodes.emplace(indexer_.id(pair.first),
                                  const_cast<Node *>(&pair.second));
        }
        for (auto &pair : edges_) {
            cache_->edges.emplace(
                std::make_tuple(indexer_.id(std::get<0>(pair.first)),
                                indexer_.id(std::get<1>(pair.first))),
                const_cast<Edge *>(&pair.second));
        }
        return *cache_;
    }

    void round(Route &r) const
    {
        r.dist = ROUND(r.dist, *round_scale_);
        if (r.start_offset) {
            r.start_offset = ROUND(*r.start_offset, *round_scale_);
        }
        if (r.end_offset) {
            r.end_offset = ROUND(*r.end_offset, *round_scale_);
        }
    }

    std::vector<std::string>
    __nexts(const std::string &id,
            const unordered_map<int64_t, unordered_set<int64_t>> &jumps) const
    {
        auto idx = indexer_.get_id(id);
        if (!idx) {
            return {};
        }
        auto itr = jumps.find(*idx);
        if (itr == jumps.end()) {
            return {};
        }
        auto nodes = std::vector<std::string>{};
        nodes.reserve(itr->second.size());
        for (auto &prev : itr->second) {
            nodes.push_back(indexer_.id(prev));
        }
        return nodes;
    }

    void single_source_dijkstra(
        int64_t start, double cutoff, //
        const unordered_map<int64_t, unordered_set<int64_t>> &jumps,
        unordered_map<int64_t, int64_t> &pmap,
        unordered_map<int64_t, double> &dmap,
        const unordered_set<int64_t> *sinks = nullptr,
        double init_offset = 0.0) const
    {
        // https://github.com/cubao/nano-fmm/blob/37d2979503f03d0a2759fc5f110e2b812d963014/src/nano_fmm/network.cpp#L449C67-L449C72
        auto itr = jumps.find(start);
        if (itr == jumps.end()) {
            return;
        }
        Heap Q;
        Q.push(start, 0.0);
        dmap.insert({start, 0.0});
        for (auto next : itr->second) {
            Q.push(next, init_offset);
            pmap.insert({next, start});
            dmap.insert({next, init_offset});
        }
        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            if (node.value > cutoff) {
                break;
            }
            auto u = node.index;
            if (sinks && sinks->count(u)) {
                continue;
            }
            auto itr = jumps.find(u);
            if (itr == jumps.end()) {
                continue;
            }
            double u_cost = lengths_.at(u);
            for (auto v : itr->second) {
                auto c = node.value + u_cost;
                auto iter = dmap.find(v);
                if (iter != dmap.end()) {
                    if (c < iter->second) {
                        pmap[v] = u;
                        dmap[v] = c;
                        Q.decrease_key(v, c);
                    }
                } else {
                    if (c <= cutoff) {
                        pmap.insert({v, u});
                        dmap.insert({v, c});
                        Q.push(v, c);
                    }
                }
            }
        }
        dmap.erase(start);
    }

    std::vector<Route> __all_routes(
        int64_t source, double cutoff, std::optional<double> offset,
        const unordered_map<int64_t, double> &lengths,
        const unordered_map<int64_t, unordered_set<int64_t>> &jumps) const
    {
        auto length = lengths.find(source);
        if (length == lengths.end()) {
            return {};
        }

        if (offset) {
            offset = std::max(0.0, std::min(*offset, length->second));
            double delta = length->second - *offset;
            if (cutoff <= delta) {
                return {
                    Route(this, cutoff, {source}, *offset, *offset + cutoff)};
            }
            cutoff -= delta;
        }
        std::vector<Route> routes;
        std::function<void(std::vector<int64_t> &, double)> backtrace;
        backtrace = [&routes, cutoff, &lengths, &jumps, this,
                     &backtrace](std::vector<int64_t> &path, double length) {
            if (length > cutoff) {
                return;
            }
            auto tail = path.back();
            if (path.size() > 1) {
                double new_length = length + this->lengths_.at(tail);
                if (new_length > cutoff) {
                    routes.push_back(
                        Route(this, cutoff, path, {}, cutoff - length));
                    return;
                }
                length = new_length;
            }
            auto itr = jumps.find(tail);
            if (itr == jumps.end() || itr->second.empty()) {
                routes.push_back(
                    Route(this, length, path, {}, this->lengths_.at(tail)));
                return;
            }
            const auto N = routes.size();
            for (auto next : itr->second) {
                if (std::find(path.begin(), path.end(), next) != path.end()) {
                    continue;
                }
                path.push_back(next);
                backtrace(path, length);
                path.pop_back();
            }
            if (N == routes.size()) {
                routes.push_back(
                    Route(this, length, path, {}, this->lengths_.at(tail)));
            }
        };
        auto path = std::vector<int64_t>{source};
        backtrace(path, 0.0);

        if (offset) {
            double delta = length->second - *offset;
            for (auto &route : routes) {
                route.dist += delta;
                route.start_offset = *offset;
            }
        }
        std::sort(
            routes.begin(), routes.end(),
            [](const auto &r1, const auto &r2) { return r1.dist < r2.dist; });
        return routes;
    }
};
} // namespace nano_fmm

using namespace nano_fmm;

void bind_rapidjson(py::module &m);

PYBIND11_MODULE(_core, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: network-graph

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    bind_rapidjson(m);

    py::class_<Indexer>(m, "Indexer", py::module_local()) //
        .def(py::init<>())
        .def("contains",
             py::overload_cast<int64_t>(&Indexer::contains, py::const_), "id"_a)
        .def("contains",
             py::overload_cast<const std::string &>(&Indexer::contains,
                                                    py::const_),
             "id"_a)
        .def("get_id", py::overload_cast<int64_t>(&Indexer::get_id, py::const_),
             "id"_a)
        .def("get_id",
             py::overload_cast<const std::string &>(&Indexer::get_id,
                                                    py::const_),
             "id"_a)
        //
        .def("id", py::overload_cast<int64_t>(&Indexer::id), "id"_a)
        .def("id", py::overload_cast<const std::string &>(&Indexer::id), "id"_a)
        .def("index",
             py::overload_cast<const std::string &, int64_t>(&Indexer::index),
             "str_id"_a, "int_id"_a)
        .def("index", py::overload_cast<>(&Indexer::index, py::const_))
        //
        .def("from_rapidjson", &Indexer::from_rapidjson, "json"_a)
        .def("to_rapidjson",
             py::overload_cast<>(&Indexer::to_rapidjson, py::const_))
        //
        ;

    py::class_<Node>(m, "Node", py::module_local(), py::dynamic_attr()) //
        .def(py::init<>())
        .def_property_readonly("length",
                               [](const Node &self) { return self.length; })
        .def(
            "__getitem__",
            [](Node &self, const std::string &attr_name) -> py::object {
                if (attr_name == "length") {
                    return py::float_(self.length);
                }
                auto py_obj = py::cast(self);
                if (!py::hasattr(py_obj, attr_name.c_str())) {
                    throw py::key_error(
                        fmt::format("attribute:{} not found", attr_name));
                }
                return py_obj.attr(attr_name.c_str());
            },
            "attr_name"_a)
        .def("__setitem__",
             [](Node &self, const std::string &attr_name,
                py::object obj) -> py::object {
                 if (attr_name == "length") {
                     throw py::key_error("length is readonly");
                 }
                 py::cast(self).attr(attr_name.c_str()) = obj;
                 return obj;
             })
        .def("to_dict",
             [](Node &self) {
                 py::dict ret;
                 ret["length"] = self.length;
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<Edge>(m, "Edge", py::module_local(), py::dynamic_attr()) //
        .def(py::init<>())
        .def(
            "__getitem__",
            [](Edge &self, const std::string &attr_name) -> py::object {
                auto py_obj = py::cast(self);
                if (!py::hasattr(py_obj, attr_name.c_str())) {
                    throw py::key_error(
                        fmt::format("attribute:{} not found", attr_name));
                }
                return py_obj.attr(attr_name.c_str());
            },
            "attr_name"_a)
        .def("__setitem__",
             [](Edge &self, const std::string &attr_name,
                py::object obj) -> py::object {
                 py::cast(self).attr(attr_name.c_str()) = obj;
                 return obj;
             })
        .def("to_dict",
             [](Edge &self) {
                 py::dict ret;
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<Route>(m, "Route", py::module_local(), py::dynamic_attr()) //
        .def_property_readonly(
            "graph", [](Route &self) { return self.graph; },
            rvp::reference_internal)
        .def_property_readonly("dist",
                               [](const Route &self) { return self.dist; })
        .def_property_readonly(
            "path",
            [](const Route &self) { return self.graph->__node_ids(self.path); })
        .def_property_readonly(
            "start",
            [](const Route &self)
                -> std::tuple<std::string, std::optional<double>> {
                return std::make_tuple(self.graph->__node_id(self.path.front()),
                                       self.start_offset);
            })
        .def_property_readonly(
            "end",
            [](const Route &self)
                -> std::tuple<std::string, std::optional<double>> {
                return std::make_tuple(self.graph->__node_id(self.path.back()),
                                       self.end_offset);
            })
        .def(
            "__getitem__",
            [](Route &self, const std::string &attr_name) -> py::object {
                if (attr_name == "dist") {
                    return py::float_(self.dist);
                } else if (attr_name == "path") {
                    auto path = self.graph->__node_ids(self.path);
                    py::list ret;
                    for (auto &node : path) {
                        ret.append(node);
                    }
                    return ret;
                } else if (attr_name == "start") {
                    auto start = self.graph->__node_id(self.path.front());
                    return py::make_tuple(start, self.start_offset);
                } else if (attr_name == "end") {
                    auto end = self.graph->__node_id(self.path.back());
                    return py::make_tuple(end, self.end_offset);
                }
                auto py_obj = py::cast(self);
                if (!py::hasattr(py_obj, attr_name.c_str())) {
                    throw py::key_error(
                        fmt::format("attribute:{} not found", attr_name));
                }
                return py_obj.attr(attr_name.c_str());
            },
            "attr_name"_a)
        .def("__setitem__",
             [](Route &self, const std::string &attr_name,
                py::object obj) -> py::object {
                 if (attr_name == "graph" || attr_name == "dist" ||
                     attr_name == "path" || attr_name == "start" ||
                     attr_name == "end") {
                     throw py::key_error(
                         fmt::format("{} is readonly", attr_name));
                 }
                 py::cast(self).attr(attr_name.c_str()) = obj;
                 return obj;
             })
        .def("to_dict",
             [](Route &self) {
                 py::dict ret;
                 ret["dist"] = self.dist;
                 py::list path;
                 for (auto &node : self.graph->__node_ids(self.path)) {
                     path.append(node);
                 }
                 ret["path"] = path;
                 auto start = self.graph->__node_id(self.path.front());
                 ret["start"] = py::make_tuple(start, self.start_offset);
                 auto end = self.graph->__node_id(self.path.back());
                 ret["end"] = py::make_tuple(end, self.end_offset);
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<DiGraph>(m, "DiGraph", py::module_local(), py::dynamic_attr()) //
        .def(py::init<std::optional<int8_t>>(), "round_n"_a = 3)
        //
        .def_property_readonly("round_n", &DiGraph::round_n)
        .def_property_readonly("round_scale", &DiGraph::round_scale)
        //
        .def("add_node", &DiGraph::add_node, "id"_a, py::kw_only(), "length"_a,
             rvp::reference_internal)
        .def("add_edge", &DiGraph::add_edge, "node0"_a, "node1"_a,
             rvp::reference_internal)
        //
        .def_property_readonly("nodes", &DiGraph::nodes,
                               rvp::reference_internal)
        .def_property_readonly("edges", &DiGraph::edges,
                               rvp::reference_internal)
        //
        .def("predecessors", &DiGraph::predecessors, "id"_a)
        .def("successors", &DiGraph::successors, "id"_a)
        //
        .def(
            "single_source_dijkstra",
            [](const DiGraph &self, const std::string &id, double cutoff,
               std::optional<double> offset, bool reverse) {
                return self.single_source_dijkstra(id, cutoff, offset, nullptr,
                                                   nullptr, reverse);
            },
            "id"_a, py::kw_only(),     //
            "cutoff"_a,                //
            "offset"_a = std::nullopt, //
            "reverse"_a = false)
        .def("all_routes_from", &DiGraph::all_routes_from, "source"_a,
             py::kw_only(), "cutoff"_a, "offset"_a = std::nullopt)
        .def("all_routes_to", &DiGraph::all_routes_to, "target"_a,
             py::kw_only(), "cutoff"_a, "offset"_a = std::nullopt)
        //
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

constexpr const auto RJFLAGS = rapidjson::kParseDefaultFlags |      //
                               rapidjson::kParseCommentsFlag |      //
                               rapidjson::kParseFullPrecisionFlag | //
                               rapidjson::kParseTrailingCommasFlag;

inline RapidjsonValue deepcopy(const RapidjsonValue &json,
                               RapidjsonAllocator &allocator)
{
    RapidjsonValue copy;
    copy.CopyFrom(json, allocator);
    return copy;
}
inline RapidjsonValue deepcopy(const RapidjsonValue &json)
{
    RapidjsonAllocator allocator;
    return deepcopy(json, allocator);
}

inline RapidjsonValue __py_int_to_rapidjson(const py::handle &obj)
{
    try {
        auto num = obj.cast<int64_t>();
        if (py::int_(num).equal(obj)) {
            return RapidjsonValue(num);
        }
    } catch (...) {
    }
    try {
        auto num = obj.cast<uint64_t>();
        if (py::int_(num).equal(obj)) {
            return RapidjsonValue(num);
        }
    } catch (...) {
    }
    throw std::runtime_error(
        "failed to convert to rapidjson, invalid integer: " +
        py::repr(obj).cast<std::string>());
}

inline RapidjsonValue to_rapidjson(const py::handle &obj,
                                   RapidjsonAllocator &allocator)
{
    if (obj.ptr() == nullptr || obj.is_none()) {
        return {};
    }
    if (py::isinstance<py::bool_>(obj)) {
        return RapidjsonValue(obj.cast<bool>());
    }
    if (py::isinstance<py::int_>(obj)) {
        return __py_int_to_rapidjson(obj);
    }
    if (py::isinstance<py::float_>(obj)) {
        return RapidjsonValue(obj.cast<double>());
    }
    if (py::isinstance<py::bytes>(obj)) {
        // https://github.com/pybind/pybind11_json/blob/master/include/pybind11_json/pybind11_json.hpp#L112
        py::module base64 = py::module::import("base64");
        auto str = base64.attr("b64encode")(obj)
                       .attr("decode")("utf-8")
                       .cast<std::string>();
        return RapidjsonValue(str.data(), str.size(), allocator);
    }
    if (py::isinstance<py::str>(obj)) {
        auto str = obj.cast<std::string>();
        return RapidjsonValue(str.data(), str.size(), allocator);
    }
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        RapidjsonValue arr(rapidjson::kArrayType);
        for (const py::handle &value : obj) {
            arr.PushBack(to_rapidjson(value, allocator), allocator);
        }
        return arr;
    }
    if (py::isinstance<py::dict>(obj)) {
        RapidjsonValue kv(rapidjson::kObjectType);
        for (const py::handle &key : obj) {
            auto k = py::str(key).cast<std::string>();
            kv.AddMember(RapidjsonValue(k.data(), k.size(), allocator),
                         to_rapidjson(obj[key], allocator), allocator);
        }
        return kv;
    }
    if (py::isinstance<RapidjsonValue>(obj)) {
        auto ptr = py::cast<const RapidjsonValue *>(obj);
        return deepcopy(*ptr, allocator);
    }
    throw std::runtime_error(
        "to_rapidjson not implemented for this type of object: " +
        py::repr(obj).cast<std::string>());
}

inline RapidjsonValue to_rapidjson(const py::handle &obj)
{
    RapidjsonAllocator allocator;
    return to_rapidjson(obj, allocator);
}

inline py::object to_python(const RapidjsonValue &j)
{
    if (j.IsNull()) {
        return py::none();
    } else if (j.IsBool()) {
        return py::bool_(j.GetBool());
    } else if (j.IsNumber()) {
        if (j.IsUint64()) {
            return py::int_(j.GetUint64());
        } else if (j.IsInt64()) {
            return py::int_(j.GetInt64());
        } else {
            return py::float_(j.GetDouble());
        }
    } else if (j.IsString()) {
        return py::str(std::string{j.GetString(), j.GetStringLength()});
    } else if (j.IsArray()) {
        py::list ret;
        for (const auto &e : j.GetArray()) {
            ret.append(to_python(e));
        }
        return ret;
    } else {
        py::dict ret;
        for (auto &m : j.GetObject()) {
            ret[py::str(
                std::string{m.name.GetString(), m.name.GetStringLength()})] =
                to_python(m.value);
        }
        return ret;
    }
}

inline void sort_keys_inplace(RapidjsonValue &json)
{
    if (json.IsArray()) {
        for (auto &e : json.GetArray()) {
            sort_keys_inplace(e);
        }
    } else if (json.IsObject()) {
        auto obj = json.GetObject();
        // https://rapidjson.docsforge.com/master/sortkeys.cpp/
        std::sort(obj.MemberBegin(), obj.MemberEnd(), [](auto &lhs, auto &rhs) {
            return strcmp(lhs.name.GetString(), rhs.name.GetString()) < 0;
        });
        for (auto &kv : obj) {
            sort_keys_inplace(kv.value);
        }
    }
}

inline RapidjsonValue sort_keys(const RapidjsonValue &json)
{
    RapidjsonAllocator allocator;
    RapidjsonValue copy;
    copy.CopyFrom(json, allocator);
    sort_keys_inplace(copy);
    return copy;
}

inline RapidjsonValue load_json(const std::string &path)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("can't open for reading: " + path);
    }
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    RapidjsonDocument d;
    d.ParseStream<RJFLAGS>(is);
    fclose(fp);
    return RapidjsonValue{std::move(d.Move())};
}
inline bool dump_json(const std::string &path, const RapidjsonValue &json,
                      bool indent = false, bool _sort_keys = false)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) {
        std::cerr << "can't open for writing: " + path << std::endl;
        return false;
    }
    using namespace rapidjson;
    char writeBuffer[65536];
    bool succ = false;
    FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
    if (indent) {
        PrettyWriter<FileWriteStream> writer(os);
        if (_sort_keys) {
            succ = sort_keys(json).Accept(writer);
        } else {
            succ = json.Accept(writer);
        }
    } else {
        Writer<FileWriteStream> writer(os);
        if (_sort_keys) {
            succ = sort_keys(json).Accept(writer);
        } else {
            succ = json.Accept(writer);
        }
    }
    fclose(fp);
    return succ;
}

inline RapidjsonValue loads(const std::string &json)
{
    RapidjsonDocument d;
    rapidjson::StringStream ss(json.data());
    d.ParseStream<RJFLAGS>(ss);
    if (d.HasParseError()) {
        throw std::invalid_argument(
            "invalid json, offset: " + std::to_string(d.GetErrorOffset()) +
            ", error: " + rapidjson::GetParseError_En(d.GetParseError()));
    }
    return RapidjsonValue{std::move(d.Move())};
}
inline std::string dumps(const RapidjsonValue &json, bool indent = false,
                         bool _sort_keys = false)
{
    if (_sort_keys) {
        return dumps(sort_keys(json), indent, !sort_keys);
    }
    rapidjson::StringBuffer buffer;
    if (indent) {
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        json.Accept(writer);
    } else {
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        json.Accept(writer);
    }
    return buffer.GetString();
}

inline bool __bool__(const RapidjsonValue &self)
{
    if (self.IsArray()) {
        return !self.Empty();
    } else if (self.IsObject()) {
        return !self.ObjectEmpty();
    } else if (self.IsString()) {
        return self.GetStringLength() != 0u;
    } else if (self.IsBool()) {
        return self.GetBool();
    } else if (self.IsNumber()) {
        if (self.IsUint64()) {
            return self.GetUint64() != 0;
        } else if (self.IsInt64()) {
            return self.GetInt64() != 0;
        } else {
            return self.GetDouble() != 0.0;
        }
    }
    return !self.IsNull();
}

inline int __len__(const RapidjsonValue &self)
{
    if (self.IsArray()) {
        return self.Size();
    } else if (self.IsObject()) {
        return self.MemberCount();
    }
    return 0;
}

void bind_rapidjson(py::module &m)
{
    auto rj =
        py::class_<RapidjsonValue>(m, "rapidjson", py::module_local()) //
            .def(py::init<>())
            .def(py::init(
                [](const py::object &obj) { return to_rapidjson(obj); }))
            // type checks
            .def("GetType", &RapidjsonValue::GetType)   //
            .def("IsNull", &RapidjsonValue::IsNull)     //
            .def("IsFalse", &RapidjsonValue::IsFalse)   //
            .def("IsTrue", &RapidjsonValue::IsTrue)     //
            .def("IsBool", &RapidjsonValue::IsBool)     //
            .def("IsObject", &RapidjsonValue::IsObject) //
            .def("IsArray", &RapidjsonValue::IsArray)   //
            .def("IsNumber", &RapidjsonValue::IsNumber) //
            .def("IsInt", &RapidjsonValue::IsInt)       //
            .def("IsUint", &RapidjsonValue::IsUint)     //
            .def("IsInt64", &RapidjsonValue::IsInt64)   //
            .def("IsUint64", &RapidjsonValue::IsUint64) //
            .def("IsDouble", &RapidjsonValue::IsDouble) //
            .def("IsFloat", &RapidjsonValue::IsFloat)   //
            .def("IsString", &RapidjsonValue::IsString) //
            //
            .def("IsLosslessDouble", &RapidjsonValue::IsLosslessDouble) //
            .def("IsLosslessFloat", &RapidjsonValue::IsLosslessFloat)   //
            //
            .def("SetNull", &RapidjsonValue::SetNull)     //
            .def("SetObject", &RapidjsonValue::SetObject) //
            .def("SetArray", &RapidjsonValue::SetArray)   //
            .def("SetInt", &RapidjsonValue::SetInt)       //
            .def("SetUint", &RapidjsonValue::SetUint)     //
            .def("SetInt64", &RapidjsonValue::SetInt64)   //
            .def("SetUint64", &RapidjsonValue::SetUint64) //
            .def("SetDouble", &RapidjsonValue::SetDouble) //
            .def("SetFloat", &RapidjsonValue::SetFloat)   //
            // setstring
            // get string
            //
            .def("Empty",
                 [](const RapidjsonValue &self) { return !__bool__(self); })
            .def("__bool__",
                 [](const RapidjsonValue &self) { return __bool__(self); })
            .def(
                "Size",
                [](const RapidjsonValue &self) -> int { return __len__(self); })
            .def(
                "__len__",
                [](const RapidjsonValue &self) -> int { return __len__(self); })
            .def("HasMember",
                 [](const RapidjsonValue &self, const std::string &key) {
                     return self.HasMember(key.c_str());
                 })
            .def("__contains__",
                 [](const RapidjsonValue &self, const std::string &key) {
                     return self.HasMember(key.c_str());
                 })
            .def("keys",
                 [](const RapidjsonValue &self) {
                     std::vector<std::string> keys;
                     if (self.IsObject()) {
                         keys.reserve(self.MemberCount());
                         for (auto &m : self.GetObject()) {
                             keys.emplace_back(m.name.GetString(),
                                               m.name.GetStringLength());
                         }
                     }
                     return keys;
                 })
            .def(
                "values",
                [](RapidjsonValue &self) {
                    std::vector<RapidjsonValue *> values;
                    if (self.IsObject()) {
                        values.reserve(self.MemberCount());
                        for (auto &m : self.GetObject()) {
                            values.push_back(&m.value);
                        }
                    }
                    return values;
                },
                rvp::reference_internal)
            // load/dump file
            .def(
                "load",
                [](RapidjsonValue &self,
                   const std::string &path) -> RapidjsonValue & {
                    self = load_json(path);
                    return self;
                },
                rvp::reference_internal)
            .def(
                "dump",
                [](const RapidjsonValue &self, const std::string &path,
                   bool indent, bool sort_keys) -> bool {
                    return dump_json(path, self, indent, sort_keys);
                },
                "path"_a, py::kw_only(), "indent"_a = false, "sort_keys"_a = false)
            // loads/dumps string
            .def(
                "loads",
                [](RapidjsonValue &self,
                   const std::string &json) -> RapidjsonValue & {
                    self = loads(json);
                    return self;
                },
                rvp::reference_internal)
            .def(
                "dumps",
                [](const RapidjsonValue &self, bool indent, bool sort_keys) -> std::string {
                    return dumps(self, indent, sort_keys);
                },
                py::kw_only(), "indent"_a = false, "sort_keys"_a = false)
            .def(
                "get",
                [](RapidjsonValue &self,
                   const std::string &key) -> RapidjsonValue * {
                    auto itr = self.FindMember(key.c_str());
                    if (itr == self.MemberEnd()) {
                        return nullptr;
                    } else {
                        return &itr->value;
                    }
                },
                "key"_a, rvp::reference_internal)
            .def(
                "__getitem__",
                [](RapidjsonValue &self,
                   const std::string &key) -> RapidjsonValue * {
                    auto itr = self.FindMember(key.c_str());
                    if (itr == self.MemberEnd()) {
                        throw pybind11::key_error(key);
                    }
                    return &itr->value;
                },
                rvp::reference_internal)
            .def(
                "__getitem__",
                [](RapidjsonValue &self, int index) -> RapidjsonValue & {
                    return self[index >= 0 ? index : index + (int)self.Size()];
                },
                rvp::reference_internal)
            .def("__delitem__",
                 [](RapidjsonValue &self, const std::string &key) {
                     return self.EraseMember(key.c_str());
                 })
            .def("__delitem__",
                 [](RapidjsonValue &self, int index) {
                     self.Erase(
                         self.Begin() +
                         (index >= 0 ? index : index + (int)self.Size()));
                 })
            .def("clear",
                 [](RapidjsonValue &self) -> RapidjsonValue & {
                     if (self.IsObject()) {
                         self.RemoveAllMembers();
                     } else if (self.IsArray()) {
                         self.Clear();
                     }
                     return self;
                 }, rvp::reference_internal)
            // get (python copy)
            .def("GetBool", &RapidjsonValue::GetBool)
            .def("GetInt", &RapidjsonValue::GetInt)
            .def("GetUint", &RapidjsonValue::GetUint)
            .def("GetInt64", &RapidjsonValue::GetInt64)
            .def("GetUInt64", &RapidjsonValue::GetUint64)
            .def("GetFloat", &RapidjsonValue::GetFloat)
            .def("GetDouble", &RapidjsonValue::GetDouble)
            .def("GetString",
                 [](const RapidjsonValue &self) {
                     return std::string{self.GetString(),
                                        self.GetStringLength()};
                 })
            .def("GetStringLength", &RapidjsonValue::GetStringLength)
            // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=MemoryView#memory-view
            .def("GetRawString", [](const RapidjsonValue &self) {
                return py::memoryview::from_memory(
                    self.GetString(),
                    self.GetStringLength()
                );
            }, rvp::reference_internal)
            .def("Get",
                 [](const RapidjsonValue &self) { return to_python(self); })
            .def("__call__",
                 [](const RapidjsonValue &self) { return to_python(self); })
            // set
            .def(
                "set",
                [](RapidjsonValue &self,
                   const py::object &obj) -> RapidjsonValue & {
                    self = to_rapidjson(obj);
                    return self;
                },
                rvp::reference_internal)
            .def(
                "set",
                [](RapidjsonValue &self,
                   const RapidjsonValue &obj) -> RapidjsonValue & {
                    self = deepcopy(obj);
                    return self;
                },
                rvp::reference_internal)
            .def( // same as set
                "copy_from",
                [](RapidjsonValue &self,
                   const RapidjsonValue &obj) -> RapidjsonValue & {
                    self = deepcopy(obj);
                    return self;
                },
                rvp::reference_internal)
            .def(
                "__setitem__",
                [](RapidjsonValue &self, int index, const py::object &obj) {
                    self[index >= 0 ? index : index + (int)self.Size()] =
                        to_rapidjson(obj);
                    return obj;
                },
                "index"_a, "value"_a, rvp::reference_internal)
            .def(
                "__setitem__",
                [](RapidjsonValue &self, const std::string &key,
                   const py::object &obj) {
                    auto itr = self.FindMember(key.c_str());
                    if (itr == self.MemberEnd()) {
                        RapidjsonAllocator allocator;
                        self.AddMember(
                            RapidjsonValue(key.data(), key.size(), allocator),
                            to_rapidjson(obj, allocator), allocator);
                    } else {
                        RapidjsonAllocator allocator;
                        itr->value = to_rapidjson(obj, allocator);
                    }
                    return obj;
                },
                rvp::reference_internal)
            .def(
                "push_back",
                [](RapidjsonValue &self,
                   const py::object &obj) -> RapidjsonValue & {
                    RapidjsonAllocator allocator;
                    self.PushBack(to_rapidjson(obj), allocator);
                    return self;
                },
                rvp::reference_internal)
            //
            .def(
                "pop_back",
                [](RapidjsonValue &self) -> RapidjsonValue
                                             & {
                                                 self.PopBack();
                                                 return self;
                                             },
                rvp::reference_internal)
            // https://pybind11.readthedocs.io/en/stable/advanced/classes.html?highlight=__deepcopy__#deepcopy-support
            .def("__copy__",
                 [](const RapidjsonValue &self, py::dict) -> RapidjsonValue {
                     return deepcopy(self);
                 })
            .def(
                "__deepcopy__",
                [](const RapidjsonValue &self, py::dict) -> RapidjsonValue {
                    return deepcopy(self);
                },
                "memo"_a)
            .def("clone",
                 [](const RapidjsonValue &self) -> RapidjsonValue {
                     return deepcopy(self);
                 })
            // https://pybind11.readthedocs.io/en/stable/advanced/classes.html?highlight=pickle#pickling-support
            .def(py::pickle(
                [](const RapidjsonValue &self) { return to_python(self); },
                [](py::object o) -> RapidjsonValue { return to_rapidjson(o); }))
            .def(py::self == py::self)
            .def(py::self != py::self)
        //
        ;
    py::enum_<rapidjson::Type>(rj, "type")
        .value("kNullType", rapidjson::kNullType)
        .value("kFalseType", rapidjson::kFalseType)
        .value("kTrueType", rapidjson::kTrueType)
        .value("kObjectType", rapidjson::kObjectType)
        .value("kArrayType", rapidjson::kArrayType)
        .value("kStringType", rapidjson::kStringType)
        .value("kNumberType", rapidjson::kNumberType)
        .export_values();
}
