#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <fstream>
#include <iostream>
#include <set>

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

inline double CLIP(double low, double v, double high)
{
    // return std::max(low, std::min(v, high));
    return v < low ? low : (v > high ? high : v);
}

struct Sinks
{
    // you stop at (on) sinks, no passing through
    const DiGraph *graph{nullptr};
    unordered_set<int64_t> nodes;
};

using Binding = std::tuple<double, double, py::object>; // always sorted
struct Bindings
{
    // you stop at first hit of bindings
    const DiGraph *graph{nullptr};
    unordered_map<int64_t, std::vector<Binding>> node2bindings;
};

struct Path
{
    Path() = default;
    Path(const DiGraph *graph, double dist = 0.0,
         const std::vector<int64_t> &nodes = {},
         std::optional<double> start_offset = {},
         std::optional<double> end_offset = {})
        : graph(graph), dist(dist), nodes(nodes), start_offset(start_offset),
          end_offset(end_offset)
    {
    }
    const DiGraph *graph{nullptr};
    double dist{0.0};
    std::vector<int64_t> nodes;
    std::optional<double> start_offset;
    std::optional<double> end_offset;
    std::optional<std::tuple<int64_t, Binding>> binding;

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

    bool through_sinks(const Sinks &sinks) const
    {
        if (sinks.graph != this->graph) {
            return false;
        }
        for (const auto &p : nodes) {
            if (sinks.nodes.count(p)) {
                // TODO, accept p == nodes[-1]?
                return true;
            }
        }
        return false;
    }

    bool through_bindings(const Bindings &bindings) const
    {
        if (bindings.graph != this->graph || nodes.empty()) {
            return false;
        }
        auto &kv = bindings.node2bindings;
        if (start_offset) {
            auto itr = kv.find(nodes.front());
            if (itr != kv.end() && !itr->second.empty()) {
                if (*start_offset <= std::get<1>(itr->second.back())) {
                    return true;
                }
            }
        }
        if (end_offset) {
            auto itr = kv.find(nodes.back());
            if (itr != kv.end() && !itr->second.empty()) {
                if (*start_offset >= std::get<0>(itr->second.front())) {
                    return true;
                }
            }
        }
        for (int i = 1, N = nodes.size(); i < N - 1; ++i) {
            auto itr = kv.find(nodes[i]);
            if (itr != kv.end() && !itr->second.empty()) {
                return true;
            }
        }
        return false;
    }
};

struct ShortestPathGenerator
{
    const DiGraph *graph{nullptr};
    unordered_map<int64_t, int64_t> prevs;
    unordered_map<int64_t, double> dists;

    using Click = std::tuple<int64_t, std::optional<double>>;
    double cutoff{0.0};
    std::optional<Click> source;
    std::optional<Click> target;
    bool ready() const
    {
        return graph && cutoff > 0 && ((bool)source ^ (bool)target);
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
        if (freezed_) {
            throw std::logic_error("DiGraph already freezed!");
        }
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
        if (freezed_) {
            throw std::logic_error("DiGraph already freezed!");
        }
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

    Sinks encode_sinks(const std::unordered_set<std::string> &nodes)
    {
        Sinks ret;
        ret.graph = this;
        for (auto &n : nodes) {
            ret.nodes.insert(indexer_.id(n));
        }
        return ret;
    }
    Bindings encode_bindings(
        const std::unordered_map<std::string, std::vector<Binding>> &bindings)
    {
        Bindings ret;
        ret.graph = this;
        for (auto &pair : bindings) {
            auto [itr, _] =
                ret.node2bindings.emplace(indexer_.id(pair.first), pair.second);
            std::sort(itr->second.begin(), itr->second.end());
        }
        return ret;
    }

    std::optional<int64_t> __node_id(const std::string &node) const
    {
        return indexer_.get_id(node);
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
    double length(int64_t node) const { return lengths_.at(node); }

    std::optional<Path> shortest_path(const std::string &source,           //
                                      const std::string &target,           //
                                      double cutoff,                       //
                                      std::optional<double> source_offset, //
                                      std::optional<double> target_offset,
                                      const Sinks *sinks = nullptr) const
    {
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        auto src_length = lengths_.find(*src_idx);
        if (src_length == lengths_.end()) {
            return {};
        }
        auto dst_idx = indexer_.get_id(target);
        if (!dst_idx) {
            return {};
        }
        auto dst_length = lengths_.find(*dst_idx);
        if (dst_length == lengths_.end()) {
            return {};
        }
        if (source_offset) {
            source_offset = CLIP(0.0, *source_offset, src_length->second);
        }
        if (target_offset) {
            target_offset = CLIP(0.0, *target_offset, dst_length->second);
        }
        std::optional<Path> path;
        if (*src_idx == *dst_idx) {
            if (!source_offset || !target_offset) {
                return {};
            }
            if (*target_offset - *source_offset > cutoff) {
                return {};
            }
            double dist = *target_offset - *source_offset;
            if (dist <= 0) {
                return {};
            }
            path = Path(this, dist, std::vector<int64_t>{*src_idx},
                        source_offset, target_offset);
        } else {
            double delta = 0.0;
            if (source_offset) {
                delta += src_length->second - *source_offset;
            }
            if (target_offset) {
                delta += *target_offset;
            }
            cutoff -= delta;
            path = __dijkstra(*src_idx, *dst_idx, cutoff, sinks);
            if (path) {
                path->dist += delta;
                path->start_offset = source_offset;
                path->end_offset = target_offset;
            }
        }
        if (path && round_scale_) {
            path->round(*round_scale_);
        }
        return path;
    }

    ShortestPathGenerator shortest_paths(const std::string &start,          //
                                         double cutoff,                     //
                                         std::optional<double> offset = {}, //
                                         bool reverse = false,              //
                                         const Sinks *sinks = nullptr) const
    {
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
        }
        auto start_idx = indexer_.get_id(start);
        if (!start_idx) {
            return {};
        }
        auto length = lengths_.find(*start_idx);
        if (length == lengths_.end()) {
            return {};
        }
        ShortestPathGenerator shortest_path;
        shortest_path.graph = this;
        shortest_path.cutoff = cutoff;
        if (!reverse) {
            shortest_path.source = std::make_tuple(*start_idx, offset);
        } else {
            shortest_path.target = std::make_tuple(*start_idx, offset);
        }
        auto &pmap = shortest_path.prevs;
        auto &dmap = shortest_path.dists;
        double init_offset =
            !offset ? 0.0
                    : (reverse ? std::max(0.0, *offset)
                               : std::max(0.0, length->second - *offset));
        single_source_dijkstra(*start_idx, cutoff, reverse ? prevs_ : nexts_,
                               pmap, dmap, sinks, init_offset);
        return shortest_path;
    }

    std::vector<Path> all_paths_from(const std::string &source, double cutoff,
                                     std::optional<double> offset = {},
                                     const Sinks *sinks = nullptr) const
    {
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        auto paths =
            __all_paths(*src_idx, cutoff, offset, lengths_, nexts_, sinks);
        if (round_scale_) {
            for (auto &p : paths) {
                p.round(*round_scale_);
            }
        }
        return paths;
    }

    std::vector<Path> all_paths_to(const std::string &target, double cutoff,
                                   std::optional<double> offset = {},
                                   const Sinks *sinks = nullptr) const
    {
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
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
            offset = CLIP(0.0, *offset, length->second);
            offset = length->second - *offset;
        }
        auto paths =
            __all_paths(*dst_idx, cutoff, offset, lengths_, prevs_, sinks);
        for (auto &p : paths) {
            if (p.start_offset) {
                p.start_offset = lengths_.at(p.nodes.front()) - *p.start_offset;
            }
            if (p.end_offset) {
                p.end_offset = lengths_.at(p.nodes.back()) - *p.end_offset;
            }
            std::reverse(p.nodes.begin(), p.nodes.end());
            std::swap(p.start_offset, p.end_offset);
        }
        if (round_scale_) {
            for (auto &p : paths) {
                p.round(*round_scale_);
            }
        }
        return paths;
    }

    std::vector<Path> all_paths(const std::string &source,           //
                                const std::string &target,           //
                                double cutoff,                       //
                                std::optional<double> source_offset, //
                                std::optional<double> target_offset,
                                const Sinks *sinks = nullptr) const
    {
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        auto src_length = lengths_.find(*src_idx);
        if (src_length == lengths_.end()) {
            return {};
        }
        auto dst_idx = indexer_.get_id(target);
        if (!dst_idx) {
            return {};
        }
        auto dst_length = lengths_.find(*dst_idx);
        if (dst_length == lengths_.end()) {
            return {};
        }
        if (source_offset) {
            source_offset = CLIP(0.0, *source_offset, src_length->second);
        }
        if (target_offset) {
            target_offset = CLIP(0.0, *target_offset, dst_length->second);
        }
        std::vector<Path> paths;
        if (*src_idx == *dst_idx) {
            if (!source_offset || !target_offset) {
                return {};
            }
            if (*target_offset - *source_offset > cutoff) {
                return {};
            }
            double dist = *target_offset - *source_offset;
            if (dist <= 0) {
                return {};
            }
            paths.emplace_back(this, dist, std::vector<int64_t>{*src_idx},
                               source_offset, target_offset);
        } else {
            double delta = 0.0;
            if (source_offset) {
                delta += src_length->second - *source_offset;
            }
            if (target_offset) {
                delta += *target_offset;
            }
            cutoff -= delta;
            paths = __all_paths(*src_idx, *dst_idx, cutoff, sinks);
            for (auto &p : paths) {
                p.dist += delta;
                p.start_offset = source_offset;
                p.end_offset = target_offset;
            }
        }
        if (round_scale_) {
            for (auto &p : paths) {
                p.round(*round_scale_);
            }
        }
        return paths;
    }

    std::tuple<std::optional<Path>, std::optional<Path>>
    shortest_path_to_bindings(
        const std::string &source,         //
        double cutoff,                     //
        const Bindings &bindings,          //
        std::optional<double> offset = {}, //
        int direction = 0, // 0 -> forwards/backwards, 1->forwards, -1:backwards
        const Sinks *sinks = nullptr) const
    {
        if (bindings.graph != this) {
            return {};
        }
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        auto length = lengths_.find(*src_idx);
        if (length == lengths_.end()) {
            return {};
        }
        std::optional<Path> forward_path;
        if (direction >= 0) {
            forward_path = __shortest_path_to_bindings(
                *src_idx, offset, length->second, cutoff, bindings, sinks);
        }
        std::optional<Path> backward_path;
        if (direction <= 0) {
            backward_path =
                __shortest_path_to_bindings(*src_idx, offset, length->second,
                                            cutoff, bindings, sinks, true);
        }
        if (round_scale_) {
            if (forward_path) {
                forward_path->round(*round_scale_);
            }
            if (backward_path) {
                backward_path->round(*round_scale_);
            }
        }
        return std::make_tuple(backward_path, forward_path);
    }
    std::tuple<std::optional<double>, std::optional<double>>
    distance_to_bindings(const std::string &source,         //
                         double cutoff,                     //
                         const Bindings &bindings,          //
                         std::optional<double> offset = {}, //
                         int direction = 0, const Sinks *sinks = nullptr) const
    {
        auto [backwards, forwards] = shortest_path_to_bindings(
            source, cutoff, bindings, offset, direction, sinks);
        std::optional<double> backward_dist;
        std::optional<double> forward_dist;
        if (backwards) {
            backward_dist = backwards->dist;
        }
        if (forwards) {
            forward_dist = forwards->dist;
        }
        return std::make_tuple(backward_dist, forward_dist);
    }

    std::tuple<std::vector<Path>, std::vector<Path>>
    all_paths_to_bindings(const std::string &source,         //
                          double cutoff,                     //
                          const Bindings &bindings,          //
                          std::optional<double> offset = {}, //
                          int direction = 0,                 //
                          const Sinks *sinks = nullptr) const
    {
        if (bindings.graph != this) {
            return {};
        }
        if (cutoff < 0) {
            return {};
        }
        if (sinks && sinks->graph != this) {
            sinks = nullptr;
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        auto length = lengths_.find(*src_idx);
        if (length == lengths_.end()) {
            return {};
        }
        std::vector<Path> forwards;
        if (direction >= 0) {
            forwards = __all_path_to_bindings(*src_idx, offset, length->second,
                                              cutoff, bindings, sinks);
        }
        std::vector<Path> backwards;
        if (direction <= 0) {
            backwards = __all_path_to_bindings(*src_idx, offset, length->second,
                                               cutoff, bindings, sinks, true);
        }
        if (round_scale_) {
            for (auto &r : forwards) {
                r.round(*round_scale_);
            }
            for (auto &r : backwards) {
                r.round(*round_scale_);
            }
        }
        return std::make_tuple(backwards, forwards);
    }

    // TODO, batching

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
    const Indexer &indexer() const { return indexer_; }

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

    void round(Path &r) const
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
        int64_t start, double cutoff,                                //
        const unordered_map<int64_t, unordered_set<int64_t>> &jumps, //
        unordered_map<int64_t, int64_t> &pmap,                       //
        unordered_map<int64_t, double> &dmap,                        //
        const Sinks *sinks = nullptr,                                //
        double init_offset = 0.0) const
    {
        // https://github.com/cyang-kth/fmm/blob/5cccc608903877b62969e41a58b60197a37a5c01/src/network/network_graph.cpp#L234-L274
        // https://github.com/cubao/nano-fmm/blob/37d2979503f03d0a2759fc5f110e2b812d963014/src/nano_fmm/network.cpp#L449C67-L449C72
        if (cutoff < init_offset) {
            return;
        }
        auto itr = jumps.find(start);
        if (itr == jumps.end()) {
            return;
        }
        Heap Q;
        Q.push(start, 0.0);
        if (!sinks || !sinks->nodes.count(start)) {
            for (auto next : itr->second) {
                Q.push(next, init_offset);
                pmap.insert({next, start});
                dmap.insert({next, init_offset});
            }
        }
        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            if (node.value > cutoff) {
                break;
            }
            auto u = node.index;
            if (sinks && sinks->nodes.count(u)) {
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

    std::optional<Path> __dijkstra(int64_t source, int64_t target,
                                   double cutoff,
                                   const Sinks *sinks = nullptr) const
    {
        // https://github.com/cyang-kth/fmm/blob/5cccc608903877b62969e41a58b60197a37a5c01/src/network/network_graph.cpp#L54-L97
        if (sinks && sinks->nodes.count(source)) {
            return {};
        }
        auto itr = nexts_.find(source);
        if (itr == nexts_.end()) {
            return {};
        }
        unordered_map<int64_t, int64_t> pmap;
        unordered_map<int64_t, double> dmap;
        Heap Q;
        Q.push(source, 0.0);
        for (auto next : itr->second) {
            Q.push(next, 0.0);
            pmap.insert({next, source});
            dmap.insert({next, 0.0});
        }
        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            if (node.value > cutoff) {
                break;
            }
            auto u = node.index;
            if (u == target) {
                break;
            }
            if (sinks && sinks->nodes.count(u)) {
                continue;
            }
            auto itr = nexts_.find(u);
            if (itr == nexts_.end()) {
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
        if (!pmap.count(target)) {
            return {};
        }
        auto path = Path(this);
        path.dist = dmap.at(target);
        while (target != source) {
            path.nodes.push_back(target);
            target = pmap.at(target);
        }
        path.nodes.push_back(target);
        std::reverse(path.nodes.begin(), path.nodes.end());
        return path;
    }

    std::optional<Path>
    __shortest_path_to_bindings(int64_t source,
                                std::optional<double> source_offset,
                                double source_length,
                                double cutoff,                //
                                const Bindings &bindings,     //
                                const Sinks *sinks = nullptr, //
                                bool reverse = false) const
    {
        auto &node2bindings = bindings.node2bindings;
        if (source_offset) {
            // may stop at source node
            auto itr = node2bindings.find(source);
            if (itr != node2bindings.end()) {
                std::optional<Path> path;
                if (!reverse) {
                    for (auto &t : itr->second) {
                        if (std::get<0>(t) >= *source_offset) {
                            path = Path(this);
                            path->dist = std::get<0>(t) - *source_offset;
                            path->nodes = {source};
                            path->start_offset = source_offset;
                            path->end_offset = std::get<0>(t);
                            path->binding = std::make_tuple(source, t);
                            break;
                        }
                    }
                } else {
                    for (auto it = itr->second.rbegin();
                         it != itr->second.rend(); ++it) {
                        auto &t = *it;
                        if (std::get<1>(t) <= *source_offset) {
                            path = Path(this);
                            path->dist = *source_offset - std::get<1>(t);
                            path->nodes = {source};
                            path->start_offset = std::get<1>(t);
                            path->end_offset = source_offset;
                            path->binding = std::make_tuple(source, t);
                            break;
                        }
                    }
                }
                if (path) {
                    return path->dist <= cutoff ? path : std::nullopt;
                }
            }
        }
        if (sinks && sinks->nodes.count(source)) {
            return {};
        }
        auto &jumps = reverse ? prevs_ : nexts_;
        auto itr = jumps.find(source);
        if (itr == jumps.end()) {
            return {};
        }
        unordered_map<int64_t, int64_t> pmap;
        unordered_map<int64_t, double> dmap;
        Heap Q;
        Q.push(source, 0.0);
        double init_offset = 0.0;
        if (source_offset) {
            init_offset =
                reverse ? *source_offset : source_length - *source_offset;
        }
        for (auto next : itr->second) {
            Q.push(next, init_offset);
            pmap.insert({next, source});
            dmap.insert({next, init_offset});
        }
        std::optional<Path> path;
        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            if (node.value > cutoff) {
                break;
            }
            auto u = node.index;
            auto hits = node2bindings.find(u);
            if (u != source && hits != node2bindings.end() &&
                !hits->second.empty()) {
                // check bindings
                auto &t = reverse ? hits->second.back() : hits->second.front();
                double length = lengths_.at(u);
                if (!reverse) {
                    auto &t = hits->second.front();
                    auto c = CLIP(0.0, std::get<0>(t), length);
                    if (node.value + c <= cutoff) {
                        path = Path(this);
                        path->dist = node.value + c;
                        path->nodes = {u};
                        path->start_offset = source_offset;
                        path->end_offset = c;
                        path->binding = std::make_tuple(u, t);
                    }
                } else {
                    auto &t = hits->second.back();
                    auto c = CLIP(0.0, std::get<1>(t), length);
                    if (node.value + (length - c) <= cutoff) {
                        path = Path(this);
                        path->dist = node.value + (length - c);
                        path->nodes = {u};
                        path->start_offset = source_offset;
                        path->end_offset = c;
                        path->binding = std::make_tuple(u, t);
                    }
                }
                break;
            }
            if (sinks && sinks->nodes.count(u)) {
                continue;
            }
            auto itr = nexts_.find(u);
            if (itr == nexts_.end()) {
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
        if (!path) {
            return {};
        }
        auto &nodes = path->nodes;
        int64_t target = path->nodes.back();
        nodes.clear();
        while (target != source) {
            nodes.push_back(target);
            target = pmap.at(target);
        }
        nodes.push_back(target);
        if (!reverse) {
            std::reverse(nodes.begin(), nodes.end());
        } else {
            std::swap(path->start_offset, path->end_offset);
        }
        return path;
    }

    std::vector<Path>
    __all_paths(int64_t source, double cutoff, std::optional<double> offset,
                const unordered_map<int64_t, double> &lengths,
                const unordered_map<int64_t, unordered_set<int64_t>> &jumps,
                const Sinks *sinks = nullptr) const
    {
        auto length = lengths.find(source);
        if (length == lengths.end()) {
            return {};
        }

        if (offset) {
            offset = CLIP(0.0, *offset, length->second);
            double delta = length->second - *offset;
            if (cutoff <= delta) {
                return {
                    Path(this, cutoff, {source}, *offset, *offset + cutoff)};
            }
            cutoff -= delta;
        }
        std::vector<Path> paths;
        std::function<void(std::vector<int64_t> &, double)> backtrace;
        backtrace = [&paths, cutoff, &lengths, &jumps, sinks, this,
                     &backtrace](std::vector<int64_t> &nodes, double length) {
            if (length > cutoff) {
                return;
            }
            auto tail = nodes.back();
            if (nodes.size() > 1) {
                double new_length = length + this->lengths_.at(tail);
                if (new_length > cutoff) {
                    paths.push_back(
                        Path(this, cutoff, nodes, {}, cutoff - length));
                    return;
                }
                length = new_length;
            }
            auto itr = jumps.find(tail);
            if (itr == jumps.end() || itr->second.empty() ||
                (sinks && sinks->nodes.count(tail))) {
                paths.push_back(
                    Path(this, length, nodes, {}, this->lengths_.at(tail)));
                return;
            }
            const auto N = paths.size();
            for (auto next : itr->second) {
                if (std::find(nodes.begin(), nodes.end(), next) !=
                    nodes.end()) {
                    continue;
                }
                nodes.push_back(next);
                backtrace(nodes, length);
                nodes.pop_back();
            }
            if (N == paths.size()) {
                paths.push_back(
                    Path(this, length, nodes, {}, this->lengths_.at(tail)));
            }
        };
        auto nodes = std::vector<int64_t>{source};
        backtrace(nodes, 0.0);

        if (offset) {
            double delta = length->second - *offset;
            for (auto &path : paths) {
                path.dist += delta;
                path.start_offset = *offset;
            }
        }
        std::sort(
            paths.begin(), paths.end(),
            [](const auto &p1, const auto &p2) { return p1.dist < p2.dist; });
        return paths;
    }

    std::vector<Path> __all_paths(int64_t source, int64_t target, double cutoff,
                                  const Sinks *sinks = nullptr) const
    {
        std::vector<Path> paths;
        std::function<void(std::vector<int64_t> &, double)> backtrace;
        backtrace = [&paths, target, cutoff, this, sinks,
                     &backtrace](std::vector<int64_t> &nodes, double length) {
            if (length > cutoff) {
                return;
            }
            auto tail = nodes.back();
            if (nodes.size() > 1) {
                if (tail == target) {
                    paths.push_back(Path(this, length, nodes));
                    return;
                }
                double new_length = length + this->lengths_.at(tail);
                if (new_length > cutoff) {
                    return;
                }
                length = new_length;
            }
            if (tail == target) {
                return;
            }
            if (sinks && sinks->nodes.count(tail)) {
                return;
            }
            auto itr = this->nexts_.find(tail);
            if (itr == this->nexts_.end() || itr->second.empty()) {
                return;
            }
            const auto N = paths.size();
            for (auto next : itr->second) {
                if (std::find(nodes.begin(), nodes.end(), next) !=
                    nodes.end()) {
                    continue;
                }
                nodes.push_back(next);
                backtrace(nodes, length);
                nodes.pop_back();
            }
        };
        auto nodes = std::vector<int64_t>{source};
        backtrace(nodes, 0.0);

        std::sort(
            paths.begin(), paths.end(),
            [](const auto &p1, const auto &p2) { return p1.dist < p2.dist; });
        return paths;
    }

    std::vector<Path>
    __all_path_to_bindings(int64_t source,                      //
                           std::optional<double> source_offset, //
                           double source_length,
                           double cutoff,                //
                           const Bindings &bindings,     //
                           const Sinks *sinks = nullptr, //
                           bool reverse = false) const
    {
        auto &node2bindings = bindings.node2bindings;
        if (source_offset) {
            // may stop at source node
            auto itr = node2bindings.find(source);
            if (itr != node2bindings.end()) {
                std::optional<Path> path;
                if (!reverse) {
                    for (auto &t : itr->second) {
                        if (std::get<0>(t) >= *source_offset) {
                            path = Path(this);
                            path->dist = std::get<0>(t) - *source_offset;
                            path->nodes = {source};
                            path->start_offset = source_offset;
                            path->end_offset = std::get<0>(t);
                            path->binding = std::make_tuple(source, t);
                            break;
                        }
                    }
                } else {
                    for (auto it = itr->second.rbegin();
                         it != itr->second.rend(); ++it) {
                        auto &t = *it;
                        if (std::get<1>(t) <= *source_offset) {
                            path = Path(this);
                            path->dist = *source_offset - std::get<1>(t);
                            path->nodes = {source};
                            path->start_offset = std::get<1>(t);
                            path->end_offset = source_offset;
                            path->binding = std::make_tuple(source, t);
                            break;
                        }
                    }
                }
                if (path) {
                    if (path->dist <= cutoff) {
                        return {std::move(*path)};
                    } else {
                        return {};
                    }
                }
            }
        }
        if (sinks && sinks->nodes.count(source)) {
            return {};
        }
        double init_offset = 0.0;
        if (source_offset) {
            source_offset = CLIP(0.0, *source_offset, source_length);
            if (reverse) {
                if (*source_offset > cutoff) {
                    return {};
                }
                init_offset = *source_offset;
            } else {
                if (source_length - *source_offset > cutoff) {
                    return {};
                }
                init_offset = source_length - *source_offset;
            }
        }
        auto &jumps = reverse ? prevs_ : nexts_;
        auto itr = jumps.find(source);
        if (itr == jumps.end()) {
            return {};
        }
        std::vector<Path> paths;
        std::function<void(std::vector<int64_t> &, double)> backtrace;
        backtrace = [&paths, source, source_offset, cutoff, reverse, &jumps,
                     &node2bindings, sinks, this,
                     &backtrace](std::vector<int64_t> &nodes, double length) {
            if (length > cutoff) {
                return;
            }
            auto tail = nodes.back();
            double this_length = this->lengths_.at(tail);

            auto hits = node2bindings.find(tail);
            if (tail != source && hits != node2bindings.end() &&
                !hits->second.empty()) {
                auto &t = reverse ? hits->second.back() : hits->second.front();
                if (!reverse) {
                    auto &t = hits->second.front();
                    auto c = CLIP(0.0, std::get<0>(t), this_length);
                    if (length + c <= cutoff) {
                        auto path = Path(this);
                        path.dist = length + c;
                        path.nodes = nodes;
                        path.start_offset = source_offset;
                        path.end_offset = c;
                        path.binding = std::make_tuple(tail, t);
                        paths.push_back(std::move(path));
                    }
                } else {
                    auto &t = hits->second.back();
                    auto c = CLIP(0.0, std::get<1>(t), this_length);
                    if (length + (this_length - c) <= cutoff) {
                        auto path = Path(this);
                        path.dist = length + (this_length - c);
                        path.nodes = nodes;
                        path.start_offset = source_offset;
                        path.end_offset = c;
                        path.binding = std::make_tuple(tail, t);
                        paths.push_back(std::move(path));
                    }
                }
                return;
            }
            if (sinks && sinks->nodes.count(tail)) {
                return;
            }
            auto itr = jumps.find(tail);
            if (itr == jumps.end() || itr->second.empty()) {
                return;
            }
            if (nodes.size() > 1) {
                length += this_length;
            }
            for (auto next : itr->second) {
                if (std::find(nodes.begin(), nodes.end(), next) !=
                    nodes.end()) {
                    continue;
                }
                nodes.push_back(next);
                backtrace(nodes, length);
                nodes.pop_back();
            }
        };
        auto nodes = std::vector<int64_t>{source};
        backtrace(nodes, init_offset);
        if (reverse) {
            for (auto &p : paths) {
                std::reverse(p.nodes.begin(), p.nodes.end());
                std::swap(p.start_offset, p.end_offset);
            }
        }
        std::sort(
            paths.begin(), paths.end(),
            [](const auto &p1, const auto &p2) { return p1.dist < p2.dist; });
        return paths;
    }
};
} // namespace nano_fmm

using namespace nano_fmm;

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
        ;

    py::class_<Node>(m, "Node", py::module_local(), py::dynamic_attr()) //
        .def(py::init<>())
        .def_property_readonly("length",
                               [](const Node &self) { return self.length; })
        .def(
            "__getitem__",
            [](const Node &self, const std::string &attr_name) -> py::object {
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
             [](const Node &self) {
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
            [](const Edge &self, const std::string &attr_name) -> py::object {
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
             [](const Edge &self) {
                 py::dict ret;
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<Path>(m, "Path", py::module_local(), py::dynamic_attr()) //
        .def_property_readonly(
            "graph", [](const Path &self) { return self.graph; },
            rvp::reference_internal)
        .def_property_readonly("dist",
                               [](const Path &self) { return self.dist; })
        .def_property_readonly(
            "nodes",
            [](const Path &self) { return self.graph->__node_ids(self.nodes); })
        .def_property_readonly(
            "start",
            [](const Path &self)
                -> std::tuple<std::string, std::optional<double>> {
                return std::make_tuple(
                    self.graph->__node_id(self.nodes.front()),
                    self.start_offset);
            })
        .def_property_readonly(
            "end",
            [](const Path &self)
                -> std::tuple<std::string, std::optional<double>> {
                return std::make_tuple(self.graph->__node_id(self.nodes.back()),
                                       self.end_offset);
            })
        .def_property_readonly("binding",
                               [](const Path &self) { return self.binding; })
        .def("through_sinks", &Path::through_sinks, "sinks"_a)
        .def("through_bindings", &Path::through_bindings, "bindings"_a)
        .def("through_jumps",
             [](const Path &p,
                const std::unordered_map<std::string, std::vector<std::string>>
                    &jumps) -> bool {
                 // TODO, implement
                 // maybe integrate into dijkstra(source,target)?
                 return true;
             })
        .def(
            "__getitem__",
            [](const Path &self, const std::string &attr_name) -> py::object {
                if (attr_name == "dist") {
                    return py::float_(self.dist);
                } else if (attr_name == "nodes") {
                    auto path = self.graph->__node_ids(self.nodes);
                    py::list ret;
                    for (auto &node : path) {
                        ret.append(node);
                    }
                    return ret;
                } else if (attr_name == "start") {
                    auto start = self.graph->__node_id(self.nodes.front());
                    return py::make_tuple(start, self.start_offset);
                } else if (attr_name == "end") {
                    auto end = self.graph->__node_id(self.nodes.back());
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
             [](Path &self, const std::string &attr_name,
                py::object obj) -> py::object {
                 if (attr_name == "graph" || attr_name == "dist" ||
                     attr_name == "nodes" || attr_name == "start" ||
                     attr_name == "end") {
                     throw py::key_error(
                         fmt::format("{} is readonly", attr_name));
                 }
                 py::cast(self).attr(attr_name.c_str()) = obj;
                 return obj;
             })
        .def("to_dict",
             [](const Path &self) {
                 py::dict ret;
                 ret["dist"] = self.dist;
                 py::list nodes;
                 for (auto &node : self.graph->__node_ids(self.nodes)) {
                     nodes.append(node);
                 }
                 ret["nodes"] = nodes;
                 auto start = self.graph->__node_id(self.nodes.front());
                 ret["start"] = py::make_tuple(start, self.start_offset);
                 auto end = self.graph->__node_id(self.nodes.back());
                 ret["end"] = py::make_tuple(end, self.end_offset);
                 if (self.binding) {
                     ret["binding"] = std::make_tuple( //
                         self.graph->__node_id(std::get<0>(*self.binding)),
                         std::get<1>(*self.binding));
                 }
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<Sinks>(m, "Sinks", py::module_local(), py::dynamic_attr()) //
        .def_property_readonly(
            "graph", [](const Sinks &self) { return self.graph; },
            rvp::reference_internal)
        //
        .def("__call__",
             [](const Sinks &self) {
                 std::set<std::string> ret;
                 for (auto &n : self.nodes) {
                     ret.emplace(self.graph->__node_id(n));
                 }
                 return ret;
             })
        //
        ;

    py::class_<Bindings>(m, "Bindings", py::module_local(),
                         py::dynamic_attr()) //
        .def_property_readonly(
            "graph", [](const Bindings &self) { return self.graph; },
            rvp::reference_internal)
        .def("__call__",
             [](const Bindings &self) {
                 std::map<std::string, std::vector<Binding>> ret;
                 for (auto &pair : self.node2bindings) {
                     ret.emplace(self.graph->__node_id(pair.first),
                                 pair.second);
                 }
                 return ret;
             })
        //
        ;

    py::class_<ShortestPathGenerator>(m, "ShortestPathGenerator",
                                      py::module_local(),
                                      py::dynamic_attr()) //
                                                          //
        .def(py::init<>())
        //
        .def("prevs",
             [](const ShortestPathGenerator &self) {
                 std::unordered_map<std::string, std::string> ret;
                 if (!self.ready()) {
                     return ret;
                 }
                 auto &indexer = self.graph->indexer();
                 for (auto &pair : self.prevs) {
                     auto k = indexer.get_id(pair.first);
                     auto v = indexer.get_id(pair.second);
                     if (k && v) {
                         ret.emplace(std::move(*k), std::move(*v));
                     }
                 }
                 return ret;
             })
        .def("dists",
             [](const ShortestPathGenerator &self) {
                 std::unordered_map<std::string, double> ret;
                 if (!self.ready()) {
                     return ret;
                 }
                 auto &indexer = self.graph->indexer();
                 for (auto &pair : self.dists) {
                     auto k = indexer.get_id(pair.first);
                     if (k) {
                         ret.emplace(std::move(*k), pair.second);
                     }
                 }
                 return ret;
             })
        .def("cutoff",
             [](const ShortestPathGenerator &self) { return self.cutoff; })
        .def("source",
             [](const ShortestPathGenerator &self) {
                 auto ret = std::optional<
                     std::tuple<std::string, std::optional<double>>>();
                 if (self.ready() && self.source) {
                     auto k = self.graph->indexer().get_id(
                         std::get<0>(*self.source));
                     if (k) {
                         ret = std::make_tuple(*k, std::get<1>(*self.source));
                     }
                 }
                 return ret;
             })
        .def("target",
             [](const ShortestPathGenerator &self) {
                 auto ret = std::optional<
                     std::tuple<std::string, std::optional<double>>>();
                 if (self.ready() && self.target) {
                     auto k = self.graph->indexer().get_id(
                         std::get<0>(*self.target));
                     if (k) {
                         ret = std::make_tuple(*k, std::get<1>(*self.target));
                     }
                 }
                 return ret;
             })
        .def("destinations",
             [](const ShortestPathGenerator &self)
                 -> std::vector<std::tuple<double, std::string>> {
                 if (!self.ready()) {
                     return {};
                 }
                 auto ret = std::vector<std::tuple<double, std::string>>{};
                 ret.reserve(self.dists.size());
                 auto &indexer = self.graph->indexer();
                 for (auto &pair : self.dists) {
                     ret.emplace_back(
                         std::make_tuple(pair.second, indexer.id(pair.first)));
                 }
                 std::sort(ret.begin(), ret.end());
                 return ret;
             })
        .def("paths",
             [](const ShortestPathGenerator &self) -> std::vector<Path> {
                 if (!self.ready()) {
                     return {};
                 }
                 auto scale = self.graph->round_scale();
                 auto paths = std::vector<Path>();
                 if (self.prevs.empty()) {
                     if (self.source && std::get<1>(*self.source)) {
                         auto node = std::get<0>(*self.source);
                         double length = self.graph->length(node);
                         auto start_offset =
                             CLIP(0.0, *std::get<1>(*self.source), length);
                         auto end_offset =
                             CLIP(0.0, start_offset + self.cutoff, length);
                         if (start_offset < end_offset) {
                             auto path = Path(self.graph);
                             path.dist = end_offset - start_offset;
                             path.nodes.push_back(node);
                             path.start_offset = start_offset;
                             path.end_offset = end_offset;
                             if (scale) {
                                 path.round(*scale);
                             }
                             paths.push_back(std::move(path));
                         }
                     } else if (self.target && std::get<1>(*self.target)) {
                         auto node = std::get<0>(*self.target);
                         double length = self.graph->length(node);
                         auto end_offset =
                             CLIP(0.0, *std::get<1>(*self.target), length);
                         auto start_offset =
                             CLIP(0.0, end_offset - self.cutoff, length);
                         if (start_offset < end_offset) {
                             auto path = Path(self.graph);
                             path.dist = end_offset - start_offset;
                             path.nodes.push_back(node);
                             path.start_offset = start_offset;
                             path.end_offset = end_offset;
                             if (scale) {
                                 path.round(*scale);
                             }
                             paths.push_back(std::move(path));
                         }
                     }
                     return paths;
                 }
                 unordered_set<int64_t> ends;
                 for (auto &pair : self.prevs) {
                     ends.insert(pair.first);
                 }
                 for (auto &pair : self.prevs) {
                     ends.erase(pair.second);
                 }
                 paths.reserve(ends.size());

                 const int64_t source = self.source ? std::get<0>(*self.source)
                                                    : std::get<0>(*self.target);
                 for (auto end : ends) {
                     double length = self.graph->length(end);
                     auto path = Path(self.graph);
                     double dist = self.dists.at(end);
                     path.dist = std::min(self.cutoff, dist + length);
                     while (end != source) {
                         path.nodes.push_back(end);
                         end = self.prevs.at(end);
                     }
                     path.nodes.push_back(end);
                     if (self.source) {
                         path.start_offset = std::get<1>(*self.source);
                         std::reverse(path.nodes.begin(), path.nodes.end());
                         double offset = self.cutoff - dist;
                         path.end_offset = CLIP(0.0, offset, length);
                     } else {
                         double offset = length - (self.cutoff - dist);
                         path.start_offset = CLIP(0.0, offset, length);
                         path.end_offset = std::get<1>(*self.target);
                     }
                     if (scale) {
                         path.round(*scale);
                     }
                     paths.push_back(std::move(path));
                 }
                 std::sort(paths.begin(), paths.end(),
                           [](const auto &p1, const auto &p2) {
                               return p1.dist > p2.dist;
                           });
                 return paths;
             })
        .def("path",
             [](const ShortestPathGenerator &self,
                const std::string &node) -> std::optional<Path> {
                 if (!self.ready()) {
                     return {};
                 }
                 auto node_idx = self.graph->__node_id(node);
                 if (!node_idx || !self.prevs.count(*node_idx)) {
                     return {};
                 }
                 const int64_t source = self.source ? std::get<0>(*self.source)
                                                    : std::get<0>(*self.target);
                 auto end = *node_idx;
                 double length = self.graph->length(end);
                 auto path = Path(self.graph);
                 double dist = self.dists.at(end);
                 path.dist = std::min(self.cutoff, dist + length);
                 while (end != source) {
                     path.nodes.push_back(end);
                     end = self.prevs.at(end);
                 }
                 path.nodes.push_back(end);
                 if (self.source) {
                     path.start_offset = std::get<1>(*self.source);
                     std::reverse(path.nodes.begin(), path.nodes.end());
                     double offset = self.cutoff - dist;
                     path.end_offset = CLIP(0.0, offset, length);
                 } else {
                     double offset = length - (self.cutoff - dist);
                     path.start_offset = CLIP(0.0, offset, length);
                     path.end_offset = std::get<1>(*self.target);
                 }
                 auto scale = self.graph->round_scale();
                 if (scale) {
                     path.round(*scale);
                 }
                 return path;
             })
        .def("to_dict",
             [](const ShortestPathGenerator &self) {
                 py::dict ret;
                 if (self.ready()) {
                     ret["cutoff"] = self.cutoff;
                     if (self.source) {
                         auto k = self.graph->indexer().get_id(
                             std::get<0>(*self.source));
                         if (k) {
                             ret["source"] =
                                 std::make_tuple(*k, std::get<1>(*self.source));
                         }
                     } else {
                         auto k = self.graph->indexer().get_id(
                             std::get<0>(*self.target));
                         if (k) {
                             ret["target"] =
                                 std::make_tuple(*k, std::get<1>(*self.target));
                         }
                     }
                 }
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<DiGraph>(m, "DiGraph", py::module_local(),
                        py::dynamic_attr()) //
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
        .def("encode_sinks", &DiGraph::encode_sinks, "sinks"_a)
        .def("encode_bindings", &DiGraph::encode_bindings, "bindings"_a)
        // shortest paths
        .def(
            "shortest_path",
            [](const DiGraph &self,
               const std::string &source,           //
               const std::string &target,           //
               double cutoff,                       //
               std::optional<double> source_offset, //
               std::optional<double> target_offset, //
               const Sinks *sinks) {
                return self.shortest_path(source, target, cutoff, //
                                          source_offset, target_offset, sinks);
            },
            "source"_a,                       //
            "target"_a,                       //
            py::kw_only(),                    //
            "cutoff"_a,                       //
            "source_offset"_a = std::nullopt, //
            "target_offset"_a = std::nullopt, //
            "sinks"_a = nullptr,              //
            py::call_guard<py::gil_scoped_release>())
        .def(
            "shortest_paths_from",
            [](const DiGraph &self, const std::string &source, //
               double cutoff, std::optional<double> offset,
               const Sinks *sinks) {
                return self.shortest_paths(source, cutoff, offset, false,
                                           sinks);
            },
            "source"_a,                //
            py::kw_only(),             //
            "cutoff"_a,                //
            "offset"_a = std::nullopt, //
            "sinks"_a = nullptr,       //
            py::call_guard<py::gil_scoped_release>())
        .def(
            "shortest_paths_to",
            [](const DiGraph &self, const std::string &target, //
               double cutoff, std::optional<double> offset,
               const Sinks *sinks) {
                return self.shortest_paths(target, cutoff, offset, true, sinks);
            },
            "target"_a,                //
            py::kw_only(),             //
            "cutoff"_a,                //
            "offset"_a = std::nullopt, //
            "sinks"_a = nullptr, py::call_guard<py::gil_scoped_release>())
        // all paths
        .def("all_paths_from", &DiGraph::all_paths_from, //
             "source"_a,                                 //
             py::kw_only(),                              //
             "cutoff"_a,                                 //
             "offset"_a = std::nullopt,                  //
             "sinks"_a = nullptr,                        //
             py::call_guard<py::gil_scoped_release>())
        .def("all_paths_to", &DiGraph::all_paths_to, //
             "target"_a,                             //
             py::kw_only(),                          //
             "cutoff"_a,                             //
             "offset"_a = std::nullopt,
             "sinks"_a = nullptr, //
             py::call_guard<py::gil_scoped_release>())
        .def("all_paths", &DiGraph::all_paths,
             "source"_a,                       //
             "target"_a,                       //
             py::kw_only(),                    //
             "cutoff"_a,                       //
             "source_offset"_a = std::nullopt, //
             "target_offset"_a = std::nullopt, //
             "sinks"_a = nullptr,              //
             py::call_guard<py::gil_scoped_release>())
        // shortest path to bindings
        .def("shortest_path_to_bindings", &DiGraph::shortest_path_to_bindings,
             "source"_a,                //
             py::kw_only(),             //
             "cutoff"_a,                //
             "bindings"_a,              //
             "offset"_a = std::nullopt, //
             "direction"_a = 0,
             "sinks"_a = nullptr, //
             py::call_guard<py::gil_scoped_release>())
        .def("distance_to_bindings", &DiGraph::distance_to_bindings,
             "source"_a,                //
             py::kw_only(),             //
             "cutoff"_a,                //
             "bindings"_a,              //
             "offset"_a = std::nullopt, //
             "direction"_a = 0,
             "sinks"_a = nullptr, //
             py::call_guard<py::gil_scoped_release>())
        // all paths to bindings
        .def("all_paths_to_bindings", &DiGraph::all_paths_to_bindings,
             "source"_a,                //
             py::kw_only(),             //
             "cutoff"_a,                //
             "bindings"_a,              //
             "offset"_a = std::nullopt, //
             "direction"_a = 0,
             "sinks"_a = nullptr, //
             py::call_guard<py::gil_scoped_release>())
        //
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
