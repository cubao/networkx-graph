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

inline double ROUND(double v, std::optional<double> s)
{
    return s ? ROUND(v, *s) : v;
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

struct Sequences
{
    const DiGraph *graph{nullptr};
    unordered_map<int64_t, std::vector<std::vector<int64_t>>> head2seqs;
    std::map<int, std::vector<std::vector<int64_t>>>
    search_in(const std::vector<int64_t> &nodes, bool quick_return = true) const
    {
        std::map<int, std::vector<std::vector<int64_t>>> ret;
        for (int i = 0, N = nodes.size(); i < N; ++i) {
            auto itr = head2seqs.find(nodes[i]);
            if (itr == head2seqs.end()) {
                continue;
            }
            for (auto &c : itr->second) {
                if (c.size() > N - i) {
                    continue;
                }
                if (std::equal(c.begin(), c.end(), &nodes[i])) {
                    ret[i].push_back(c);
                    if (quick_return) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }
};

inline std::array<double, 2> cheap_ruler_k(double latitude)
{
    // https://github.com/cubao/headers/blob/8ed287a7a1e2a5cd221271b19611ba4a3f33d15c/include/cubao/crs_transform.hpp#L212
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double RE = 6378.137;
    static constexpr double FE = 1.0 / 298.257223563;
    static constexpr double E2 = FE * (2 - FE);
    static constexpr double RAD = PI / 180.0;
    static constexpr double MUL = RAD * RE * 1000.;
    double coslat = std::cos(latitude * RAD);
    double w2 = 1.0 / (1.0 - E2 * (1.0 - coslat * coslat));
    double w = std::sqrt(w2);
    return {MUL * w * coslat, MUL * w * w2 * (1.0 - E2)};
}

using Point = std::array<double, 3>;
struct Endpoints
{
    const DiGraph *graph{nullptr};
    bool is_wgs84 = true;
    unordered_map<int64_t, std::tuple<Point, Point>> endpoints; // (head, tail)
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
};

struct ZigzagPath : Path
{
    // two-way routing on a DiGraph
    ZigzagPath() = default;
    ZigzagPath(const DiGraph *graph, double dist,
               const std::vector<int64_t> &nodes,
               const std::vector<int> &directions)
        : Path(graph, dist, nodes), directions(directions)
    {
    }
    std::vector<int> directions;
    void round(double scale) { dist = ROUND(dist, scale); }
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

struct ZigzagPathGenerator
{
    using State = std::tuple<int64_t, int>;
    ZigzagPathGenerator() = default;
    ZigzagPathGenerator(const DiGraph *graph, double cutoff)
        : graph(graph), cutoff(cutoff)
    {
    }

    const DiGraph *graph{nullptr};
    double cutoff{0.0};
    std::optional<int64_t> source;
    unordered_map<State, State> prevs;
    unordered_map<State, double> dists;

    bool ready() const { return graph && cutoff > 0 && source; }

    static std::optional<ZigzagPath>
    Path(const State &state, const int64_t source, //
         const DiGraph *self,                      //
         const unordered_map<State, State> &pmap,
         const unordered_map<State, double> &dmap)
    {
        std::vector<State> states;
        int64_t target = std::get<0>(state);
        int dir = -std::get<1>(state);
        double dist = dmap.at(state);
        auto cursor = state;
        while (true) {
            auto prev = pmap.find(cursor);
            if (prev == pmap.end()) {
                // assert cursor at source
                if (std::get<0>(cursor) != source) {
                    return {};
                }
                states.push_back({source, -std::get<1>(cursor)});
                break;
            }
            cursor = prev->second;
            states.push_back(cursor);
        }
        std::reverse(states.begin(), states.end());
        size_t N = states.size();
        if (N % 2 != 0) {
            return {};
        }
        auto nodes = std::vector<int64_t>{};
        auto dirs = std::vector<int>{};
        for (size_t i = 0; i < N; i += 2) {
            if (std::get<0>(states[i]) != std::get<0>(states[i + 1])) {
                return {};
            }
            nodes.push_back(std::get<0>(states[i]));
            dirs.push_back(
                std::get<1>(states[i]) < std::get<1>(states[i + 1]) ? 1 : -1);
        }
        nodes.push_back(target);
        dirs.push_back(dir);
        return ZigzagPath(self, dist, nodes, dirs);
    }
};

// https://github.com/cubao/nano-fmm/blob/master/src/nano_fmm/network/ubodt.hpp
struct UbodtRecord
{
    UbodtRecord() {}
    UbodtRecord(int64_t source_road, int64_t target_road, //
                int64_t source_next, int64_t target_prev, //
                double cost)
        : source_road(source_road), target_road(target_road), //
          source_next(source_next), target_prev(target_prev), //
          cost(cost)
    {
    }

    bool operator<(const UbodtRecord &rhs) const
    {
        if (source_road != rhs.source_road) {
            return source_road < rhs.source_road;
        }
        if (cost != rhs.cost) {
            return cost < rhs.cost;
        }
        return std::make_tuple(source_next, target_prev, target_road) <
               std::make_tuple(rhs.source_next, rhs.target_prev,
                               rhs.target_road);
    }
    bool operator==(const UbodtRecord &rhs) const
    {
        return source_road == rhs.source_road &&
               target_road == rhs.target_road &&
               source_next == rhs.source_next &&
               target_prev == rhs.target_prev && cost == rhs.cost;
    }

    int64_t source_road{0};
    int64_t target_road{0};
    int64_t source_next{0};
    int64_t target_prev{0};
    double cost{0.0};
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

    const std::unordered_map<std::string, std::unordered_set<std::string>>
    sibs_under_next() const
    {
        auto ret =
            std::unordered_map<std::string, std::unordered_set<std::string>>{};
        for (auto &kv : cache().sibs_under_next) {
            auto &sibs = ret[indexer_.id(kv.first)];
            for (auto s : kv.second) {
                sibs.insert(indexer_.id(s));
            }
        }
        return ret;
    }
    const std::unordered_map<std::string, std::unordered_set<std::string>>
    sibs_under_prev() const
    {
        auto ret =
            std::unordered_map<std::string, std::unordered_set<std::string>>{};
        for (auto &kv : cache().sibs_under_prev) {
            auto &sibs = ret[indexer_.id(kv.first)];
            for (auto s : kv.second) {
                sibs.insert(indexer_.id(s));
            }
        }
        return ret;
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
            std::sort(itr->second.begin(), itr->second.end(),
                      [](const auto &a, const auto &b) {
                          return std::tie(std::get<0>(a), std::get<1>(a)) <
                                 std::tie(std::get<0>(b), std::get<1>(b));
                      });
        }
        return ret;
    }
    Sequences
    encode_sequences(const std::vector<std::vector<std::string>> &sequences)
    {
        Sequences ret;
        ret.graph = this;
        for (auto &seq : sequences) {
            if (seq.empty()) {
                continue;
            }
            std::vector<int64_t> nodes;
            nodes.reserve(seq.size());
            for (auto s : seq) {
                nodes.push_back(indexer_.id(s));
            }
            ret.head2seqs[nodes[0]].push_back(nodes);
        }
        return ret;
    }
    Endpoints encode_endpoints(
        const std::unordered_map<std::string, std::tuple<Point, Point>>
            &endpoints,
        bool is_wgs84 = true)
    {
        Endpoints ret;
        ret.graph = this;
        ret.is_wgs84 = is_wgs84;
        for (auto &pair : endpoints) {
            ret.endpoints.emplace(indexer_.id(pair.first), pair.second);
        }
        return ret;
    }
    std::optional<UbodtRecord> encode_ubodt(const std::string &source_road,
                                            const std::string &target_road,
                                            const std::string &source_next,
                                            const std::string &target_prev,
                                            double cost) const
    {
        auto sroad = indexer_.get_id(source_road);
        if (!sroad) {
            return {};
        }
        auto troad = indexer_.get_id(target_road);
        if (!troad) {
            return {};
        }
        auto snext = indexer_.get_id(source_next);
        if (!snext) {
            return {};
        }
        auto tprev = indexer_.get_id(target_prev);
        if (!tprev) {
            return {};
        }
        return UbodtRecord(*sroad, *troad, *snext, *tprev, cost);
    }

    std::optional<int64_t> __node_id(const std::string &node) const
    {
        return indexer_.get_id(node);
    }
    std::optional<std::tuple<int64_t, double>>
    __node_length(const std::string &node) const
    {
        auto nid = __node_id(node);
        if (!nid) {
            return {};
        }
        auto len = lengths_.find(*nid);
        if (len == lengths_.end()) {
            return {};
        }
        return std::make_tuple(*nid, len->second);
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

    std::optional<Path>
    shortest_path(const std::string &source,           //
                  const std::string &target,           //
                  double cutoff,                       //
                  std::optional<double> source_offset, //
                  std::optional<double> target_offset,
                  const Sinks *sinks = nullptr,
                  const Endpoints *endpoints = nullptr) const
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
            if (!source_offset && !target_offset) {
                path = Path(this, 0.0, std::vector<int64_t>{*src_idx});
            } else if (source_offset && target_offset) {
                double dist = *target_offset - *source_offset;
                if (dist < 0 || dist > cutoff) {
                    return {};
                }
                path = Path(this, dist, std::vector<int64_t>{*src_idx},
                            source_offset, target_offset);
            } else {
                return {};
            }
        } else {
            double delta = 0.0;
            if (source_offset) {
                delta += src_length->second - *source_offset;
            }
            if (target_offset) {
                delta += *target_offset;
            }
            path = endpoints
                       ? __astar(*src_idx, *dst_idx, cutoff - delta, *endpoints,
                                 sinks)
                       : __dijkstra(*src_idx, *dst_idx, cutoff - delta, sinks);
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

    std::optional<ZigzagPath>
    shortest_zigzag_path(const std::string &source,                //
                         const std::optional<std::string> &target, //
                         double cutoff,                            //
                         int direction = 0,
                         ZigzagPathGenerator *generator = nullptr) const
    {
        if (cutoff < 0) {
            return {};
        }
        bool one_and_only = bool(target) ^ bool(generator);
        if (!one_and_only) {
            return {};
        }
        auto src_idx = indexer_.get_id(source);
        if (!src_idx) {
            return {};
        }
        std::optional<int64_t> dst_idx;
        if (target) {
            dst_idx = indexer_.get_id(*target);
            if (!dst_idx) {
                return {};
            }
        }
        auto path = __shortest_zigzag_path(*src_idx, dst_idx, cutoff, direction,
                                           generator);
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
        double init_offset = 0.0;
        if (offset) {
            offset = CLIP(0.0, *offset, length->second);
            init_offset = reverse ? *offset : length->second - *offset;
        }
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

    std::vector<UbodtRecord> build_ubodt(double thresh, int pool_size = 1,
                                         int nodes_thresh = 100) const
    {
        if (pool_size > 1 && nodes_.size() > nodes_thresh) {
            return build_ubodt_parallel(thresh, pool_size);
        }
        auto records = std::vector<UbodtRecord>();
        for (auto &kv : nodes_) {
            auto rows = build_ubodt(kv.first, thresh);
            records.insert(records.end(), rows.begin(), rows.end());
        }
        return records;
    }

    std::vector<UbodtRecord> build_ubodt(int64_t source, double thresh) const
    {
        unordered_map<int64_t, int64_t> pmap;
        unordered_map<int64_t, double> dmap;
        single_source_dijkstra(source, thresh, nexts_, pmap, dmap);
        std::vector<UbodtRecord> records;
        for (const auto &iter : pmap) {
            auto curr = iter.first;
            if (curr == source) {
                continue;
            }
            const auto prev = iter.second;
            auto succ = curr;
            int64_t u;
            while ((u = pmap[succ]) != source) {
                succ = u;
            }
            double dist = dmap[curr];
            if (round_scale_) {
                dist = ROUND(dist, *round_scale_);
            }
            records.push_back({source, curr, succ, prev, dist});
        }
        return records;
    }

    std::vector<UbodtRecord> build_ubodt_parallel(double thresh,
                                                  int poolsize) const
    {
        return {};
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
        unordered_map<int64_t, unordered_set<int64_t>> sibs_under_prev,
            sibs_under_next;
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
        {
            auto &sibs = cache_->sibs_under_next;
            for (auto &kv : nexts_) {
                if (kv.second.size() > 1) {
                    for (auto pid : kv.second) {
                        sibs[pid].insert(kv.second.begin(), kv.second.end());
                    }
                }
            }
            for (auto &kv : sibs) {
                kv.second.erase(kv.first);
            }
        }
        {
            auto &sibs = cache_->sibs_under_prev;
            for (auto &kv : prevs_) {
                if (kv.second.size() > 1) {
                    for (auto nid : kv.second) {
                        sibs[nid].insert(kv.second.begin(), kv.second.end());
                    }
                }
            }
            for (auto &kv : sibs) {
                kv.second.erase(kv.first);
            }
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
                        if (Q.contain_node(v)) {
                            Q.decrease_key(v, c);
                        } else {
                            Q.push(v, c);
                        }
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
        if (source == target) {
            return Path(this, 0.0, {source});
        }
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
            auto c = node.value + u_cost;
            if (c > cutoff) {
                continue;
            }
            for (auto v : itr->second) {
                auto iter = dmap.find(v);
                if (iter != dmap.end()) {
                    if (c < iter->second) {
                        pmap[v] = u;
                        dmap[v] = c;
                        if (Q.contain_node(v)) {
                            Q.decrease_key(v, c);
                        } else {
                            Q.push(v, c);
                        }
                    }
                } else {
                    pmap.insert({v, u});
                    dmap.insert({v, c});
                    Q.push(v, c);
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

    std::optional<Path> __astar(int64_t source, int64_t target, double cutoff,
                                const Endpoints &endpoints,
                                const Sinks *sinks = nullptr) const
    {
        // https://github.com/cyang-kth/fmm/blob/5cccc608903877b62969e41a58b60197a37a5c01/src/network/network_graph.cpp#L105-L158
        if (source == target) {
            return Path(this, 0.0, {source});
        }
        if (sinks && sinks->nodes.count(source)) {
            return {};
        }
        auto itr = nexts_.find(source);
        if (itr == nexts_.end()) {
            return {};
        }
        auto &KV = endpoints.endpoints;
        auto END = std::get<0>(KV.at(target));

        std::optional<std::array<double, 2>> k;
        if (endpoints.is_wgs84) {
            k = cheap_ruler_k(END[1]);
        }

        auto calc_heuristic_dist = [&KV, k, END](int64_t src) {
            auto &CUR = std::get<1>(KV.at(src));
            double dx = END[0] - CUR[0];
            double dy = END[1] - CUR[1];
            double dz = END[2] - CUR[2];
            if (k) {
                dx *= (*k)[0];
                dy *= (*k)[1];
            }
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };

        unordered_map<int64_t, int64_t> pmap;
        unordered_map<int64_t, double> dmap;
        Heap Q;
        Q.push(source, calc_heuristic_dist(source));
        pmap.insert({source, source});
        dmap.insert({source, 0.0});
        for (auto next : itr->second) {
            auto h = calc_heuristic_dist(next);
            Q.push(next, h + lengths_.at(next));
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
            auto c = dmap.at(u) + u_cost;
            if (c > cutoff) {
                continue;
            }
            for (auto v : itr->second) {
                auto h = calc_heuristic_dist(v) + lengths_.at(v);
                auto iter = dmap.find(v);
                if (iter != dmap.end()) {
                    if (c < iter->second) {
                        pmap[v] = u;
                        dmap[v] = c;
                        if (Q.contain_node(v)) {
                            Q.decrease_key(v, c + h);
                        } else {
                            Q.push(v, c + h);
                        }
                    }
                } else {
                    pmap.insert({v, u});
                    dmap.insert({v, c});
                    Q.push(v, c + h);
                }
            }
        }
        if (!pmap.count(target)) {
            return {};
        }
        double dist = dmap.at(target);
        if (dist > cutoff) {
            return {};
        }
        auto path = Path(this);
        path.dist = dist;
        while (target != source) {
            path.nodes.push_back(target);
            target = pmap.at(target);
        }
        path.nodes.push_back(target);
        std::reverse(path.nodes.begin(), path.nodes.end());
        return path;
    }

    std::optional<ZigzagPath>
    __shortest_zigzag_path(int64_t source, std::optional<int64_t> target,
                           double cutoff, int direction = 0,
                           ZigzagPathGenerator *generator = nullptr) const
    {
        if (!lengths_.count(source)) {
            return {};
        }
        if (target && !lengths_.count(*target)) {
            return {};
        }

        if (target && source == target) {
            return ZigzagPath(this, 0.0, {source}, {1});
        }

        const auto &sibs_under_next = cache().sibs_under_next;
        const auto &sibs_under_prev = cache().sibs_under_prev;

        using State = std::tuple<int64_t, int>;
        unordered_map<State, State> pmap;
        unordered_map<State, double> dmap;
        Heap<State> Q;
        if (direction >= 0) {
            dmap.insert({{source, 1}, 0.0});
            Q.push({source, 1}, 0.0);
        }
        if (direction <= 0) {
            dmap.insert({{source, -1}, 0.0});
            Q.push({source, -1}, 0.0);
        }

        auto update_state = [&](const State &state, double dist,
                                const State &prev) -> bool {
            if (dist > cutoff) {
                return false;
            }
            auto dist_itr = dmap.find(state);
            if (dist_itr == dmap.end()) {
                Q.push(state, dist);
                dmap[state] = dist;
                pmap[state] = prev;
                return true;
            } else if (dist < dist_itr->second) {
                if (Q.contain_node(state)) {
                    Q.decrease_key(state, dist);
                } else {
                    Q.push(state, dist);
                }
                dmap[state] = dist;
                pmap[state] = prev;
                return true;
            }
            return false;
        };

        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            double dist = node.value;
            if (dist > cutoff) {
                break;
            }
            const auto &state = node.index;
            auto idx = std::get<0>(state);
            auto dir = std::get<1>(state);
            if (target && idx == *target) {
                // backtrace from current state to source
                return ZigzagPathGenerator::Path(state, source, this, //
                                                 pmap, dmap);
            }

            if (dir == 1) {
                // forwards to nexts, or reverse to sibs
                auto next_itr = nexts_.find(idx);
                if (next_itr != nexts_.end()) {
                    for (auto n : next_itr->second) {
                        if (update_state({n, -1}, dist, state)) {
                            update_state({n, 1}, dist + lengths_.at(n),
                                         {n, -1});
                        }
                    }
                }
                auto sib_itr = sibs_under_prev.find(idx);
                if (sib_itr != sibs_under_prev.end()) {
                    for (auto s : sib_itr->second) {
                        if (update_state({s, 1}, dist, state)) {
                            update_state({s, -1}, dist + lengths_.at(s),
                                         {s, 1});
                        }
                    }
                }
            } else if (dir == -1) {
                // backwards to prevs, or reverse to sibs
                auto prev_itr = prevs_.find(idx);
                if (prev_itr != prevs_.end()) {
                    for (auto p : prev_itr->second) {
                        if (update_state({p, 1}, dist, state)) {
                            update_state({p, -1}, dist + lengths_.at(p),
                                         {p, 1});
                        }
                    }
                }
                auto sib_itr = sibs_under_next.find(idx);
                if (sib_itr != sibs_under_next.end()) {
                    for (auto s : sib_itr->second) {
                        if (update_state({s, -1}, dist, state)) {
                            update_state({s, 1}, dist + lengths_.at(s),
                                         {s, -1});
                        }
                    }
                }
            }
        }

        if (generator) {
            generator->source = source;
            generator->prevs = std::move(pmap);
            generator->dists = std::move(dmap);
        }
        return {};
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
                        if (Q.contain_node(v)) {
                            Q.decrease_key(v, c);
                        } else {
                            Q.push(v, c);
                        }
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

struct ShortestPathWithUbodt
{
    const DiGraph *graph{nullptr};
    unordered_map<std::pair<int64_t, int64_t>, UbodtRecord> ubodt;
    ShortestPathWithUbodt(const DiGraph *graph,
                          const std::vector<UbodtRecord> &ubodt)
        : graph(graph)
    {
        load_ubodt(ubodt);
    }
    void load_ubodt(const std::vector<UbodtRecord> &ubodt)
    {
        for (auto &r : ubodt) {
            this->ubodt.emplace(std::make_pair(r.source_road, r.target_road),
                                r);
            by_source_[r.source_road].push_back(
                std::make_tuple(r.cost, r.target_road));
            by_target_[r.target_road].push_back(
                std::make_tuple(r.cost, r.source_road));
        }
        for (auto &pair : by_source_) {
            auto &items = pair.second;
            std::sort(items.begin(), items.end());
        }
        for (auto &pair : by_target_) {
            auto &items = pair.second;
            std::sort(items.begin(), items.end());
        }
    }
    ShortestPathWithUbodt(const DiGraph *graph, double thresh,
                          int pool_size = 1, int nodes_thresh = 100)
        : ShortestPathWithUbodt(
              graph, graph->build_ubodt(thresh, pool_size, nodes_thresh))
    {
    }
    ShortestPathWithUbodt(const DiGraph *graph, const std::string &path)
    {
        load_ubodt(path);
    }
    void load_ubodt(const std::string &path)
    {
        return load_ubodt(Load_Ubodt(path));
    }
    std::vector<UbodtRecord> dump_ubodt() const
    {
        std::vector<UbodtRecord> rows;
        rows.reserve(ubodt.size());
        for (auto &pair : ubodt) {
            rows.push_back(pair.second);
        }
        std::sort(rows.begin(), rows.end());
        return rows;
    }
    bool dump_ubodt(const std::string &path) const
    {
        return Dump_Ubodt(dump_ubodt(), path);
    }
    size_t size() const { return ubodt.size(); }

    std::vector<std::tuple<double, std::string>>
    by_source(const std::string &source, std::optional<double> cutoff) const
    {
        return __by(source, cutoff, true);
    }
    std::vector<std::tuple<double, std::string>>
    by_target(const std::string &target, std::optional<double> cutoff) const
    {
        return __by(target, cutoff, false);
    }
    std::optional<Path> path(const std::string &source,
                             const std::string &target) const
    {
        auto src_idx = graph->indexer().get_id(source);
        if (!src_idx) {
            return {};
        }
        auto dst_idx = graph->indexer().get_id(target);
        if (!dst_idx) {
            return {};
        }
        return path(*src_idx, *dst_idx);
    }
    std::optional<double> dist(const std::string &source,
                               const std::string &target) const
    {
        auto src_idx = graph->indexer().get_id(source);
        if (!src_idx) {
            return {};
        }
        auto dst_idx = graph->indexer().get_id(target);
        if (!dst_idx) {
            return {};
        }
        auto itr = ubodt.find({*src_idx, *dst_idx});
        if (itr == ubodt.end()) {
            return {};
        }
        return itr->second.cost;
    }

    static std::vector<UbodtRecord> Load_Ubodt(const std::string &path)
    {
        auto f = std::ifstream(path.c_str(), std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            return {};
        }
        const size_t N = static_cast<size_t>(f.tellg()) / sizeof(UbodtRecord);
        std::vector<UbodtRecord> rows;
        rows.reserve(N);
        f.seekg(0);
        UbodtRecord row;
        while (f.read(reinterpret_cast<char *>(&row.source_road),
                      sizeof(UbodtRecord))) {
            rows.push_back(row);
        }
        return rows;
    }
    static bool Dump_Ubodt(const std::vector<UbodtRecord> &ubodt,
                           const std::string &path)
    {
        auto f = std::ofstream(path.c_str(), std::ios::binary);
        if (!f.is_open()) {
            return false;
        }
        for (auto &row : ubodt) {
            f.write(reinterpret_cast<const char *>(&row.source_road),
                    sizeof(row));
        }
        return true;
    }

  private:
    std::optional<Path> path(int64_t source, int64_t target) const
    {
        auto itr = ubodt.find({source, target});
        if (itr == ubodt.end()) {
            return {};
        }
        double dist = itr->second.cost;
        std::vector<int64_t> nodes;
        nodes.push_back(source);
        source = itr->second.source_next;
        while (source != target) {
            auto itr = ubodt.find({source, target});
            if (itr == ubodt.end()) {
                return {};
            }
            nodes.push_back(source);
            source = itr->second.source_next;
        }
        nodes.push_back(target);
        return Path(graph, dist, nodes);
    }

    // by source (origin), by target (destination)
    unordered_map<int64_t, std::vector<std::tuple<double, int64_t>>> by_source_,
        by_target_;
    std::vector<std::tuple<double, std::string>>
    __by(const std::string &source, std::optional<double> cutoff,
         bool by_from) const
    {
        auto &indexer = graph->indexer();
        auto idx = indexer.get_id(source);
        if (!idx) {
            return {};
        }
        auto &kv = by_from ? by_source_ : by_target_;
        auto itr = kv.find(*idx);
        if (itr == kv.end()) {
            return {};
        }
        auto ret = std::vector<std::tuple<double, std::string>>{};
        if (!cutoff) {
            ret.reserve(itr->second.size());
        }
        for (auto &p : itr->second) {
            if (cutoff && std::get<0>(p) > *cutoff) {
                break;
            }
            ret.push_back(
                std::make_tuple(std::get<0>(p), indexer.id(std::get<1>(p))));
        }
        return ret;
    }
};

} // namespace nano_fmm

using namespace nano_fmm;

inline std::tuple<int, double> __path_along(const Path &self, double offset)
{
    if (offset <= 0) {
        int idx = 0;
        auto nid = self.nodes.at(idx);
        return std::make_tuple(
            idx, self.start_offset.value_or(self.graph->length(nid)));
    } else if (offset >= self.dist) {
        int idx = self.nodes.size() - 1;
        return std::make_tuple(idx, self.end_offset.value_or(0.0));
    }
    if (self.start_offset) {
        double remain = std::max(0.0, self.graph->length(self.nodes.front()) -
                                          *self.start_offset);
        if (offset <= remain) {
            return std::make_tuple(0, *self.start_offset + offset);
        }
        offset -= remain;
    }
    for (int i = 1; i < self.nodes.size(); ++i) {
        auto nid = self.nodes.at(i);
        double length = self.graph->length(nid);
        if (offset <= length) {
            return std::make_tuple(i, offset);
        }
        offset -= length;
    }
    int idx = self.nodes.size() - 1;
    return std::make_tuple(idx, self.end_offset.value_or(0.0));
}

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
        .def(py::init<const std::map<std::string, int64_t>>(), "index"_a)
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
        .def("index",
             py::overload_cast<const std::map<std::string, int64_t> &>(
                 &Indexer::index),
             "index"_a)
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
        .def_static(
            "Build",
            [](const DiGraph &graph, const std::vector<std::string> &nodes,
               std::optional<double> start_offset = {},
               std::optional<double> end_offset = {},
               std::optional<std::tuple<std::string, Binding>> binding = {})
                -> Path {
                if (nodes.empty()) {
                    throw std::invalid_argument("not any nodes");
                }
                std::vector<int64_t> nids;
                std::vector<double> lengths;
                const auto N = nodes.size();
                nids.reserve(N);
                lengths.reserve(N);
                for (auto &node : nodes) {
                    auto nid_len = graph.__node_length(node);
                    if (!nid_len) {
                        throw std::invalid_argument(
                            fmt::format("missing node {}", node));
                    }
                    nids.push_back(std::get<0>(*nid_len));
                    lengths.push_back(std::get<1>(*nid_len));
                }
                double dist = 0.0;
                for (size_t i = 1; i < N - 1; ++i) {
                    dist += lengths[i];
                }
                if (N == 1 && start_offset && end_offset) {
                    start_offset = CLIP(0.0, *start_offset, lengths.front());
                    end_offset = CLIP(0.0, *end_offset, lengths.back());
                    dist = *end_offset - *start_offset;
                } else {
                    if (start_offset) {
                        start_offset =
                            CLIP(0.0, *start_offset, lengths.front());
                        dist += lengths.front() - *start_offset;
                    }
                    if (end_offset) {
                        end_offset = CLIP(0.0, *end_offset, lengths.back());
                        dist += *end_offset;
                    }
                }
                auto p = Path(&graph, dist, nids, start_offset, end_offset);
                auto round_scale = graph.round_scale();
                if (round_scale) {
                    p.round(*round_scale);
                }
                if (binding) {
                    auto node = std::get<0>(*binding);
                    auto nid = graph.__node_id(node);
                    if (!nid) {
                        throw std::invalid_argument(
                            fmt::format("invalid binding node {}", node));
                    }
                    p.binding = std::make_tuple(*nid, std::get<1>(*binding));
                }
                return p;
            },
            "graph"_a, "nodes"_a,            //
            py::kw_only(),                   //
            "start_offset"_a = std::nullopt, //
            "end_offset"_a = std::nullopt,   //
            "binding"_a = std::nullopt)

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
        .def_property_readonly(
            "binding",
            [](const Path &self)
                -> std::optional<std::tuple<std::string, Binding>> {
                if (!self.binding) {
                    return {};
                }
                return std::make_tuple( //
                    self.graph->__node_id(std::get<0>(*self.binding)),
                    std::get<1>(*self.binding));
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
                } else if (attr_name == "binding") {
                    if (self.binding) {
                        return py::make_tuple( //
                            self.graph->__node_id(std::get<0>(*self.binding)),
                            std::get<1>(*self.binding));
                    } else {
                        return py::none();
                    }
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
                     attr_name == "end" || attr_name == "binding") {
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
        .def(
            "search_for_seqs",
            [](const Path &self, const Sequences &seqs,
               bool quick_return = true) {
                std::map<int, std::vector<Path>> idx2paths;
                for (auto &kv : seqs.search_in(self.nodes, quick_return)) {
                    std::vector<Path> paths;
                    paths.reserve(kv.second.size());
                    for (auto &seq : kv.second) {
                        paths.push_back(Path(self.graph, 0.0, seq));
                    }
                    idx2paths.emplace(kv.first, std::move(paths));
                }
                return idx2paths;
            },
            "sequences"_a, "quick_return"_a = true)
        .def(
            "along",
            [](const Path &self,
               double offset) -> std::tuple<std::string, double> {
                auto [idx, off] = __path_along(self, offset);
                auto nid = self.graph->__node_id(self.nodes.at(idx));
                auto scale = self.graph->round_scale();
                if (scale) {
                    off = ROUND(off, *scale);
                }
                return std::make_tuple(nid, off);
            },
            "offset"_a)
        .def(
            "slice",
            [](const Path &self, double start, double end) -> Path {
                auto [idx0, off0] = __path_along(self, start);
                std::vector<int64_t> nids;
                double dist = 0.0;
                double off1 = 0.0;
                if (end <= start) {
                    nids.push_back(self.nodes.at(idx0));
                    dist = 0.0;
                    off1 = off0;
                } else {
                    auto idx_off = __path_along(self, end);
                    auto idx1 = std::get<0>(idx_off);
                    off1 = std::get<1>(idx_off);
                    if (idx0 > idx1) {
                        nids.push_back(self.nodes.at(idx0));
                        dist = 0.0;
                        off1 = off0;
                    } else if (idx0 == idx1) {
                        nids.push_back(self.nodes.at(idx0));
                        dist = off1 - off0;
                    } else {
                        int nid = self.nodes.at(idx0);
                        nids.push_back(nid);
                        dist += self.graph->length(nid) - off0;
                        for (int idx = idx0 + 1; idx < idx1; ++idx) {
                            nid = self.nodes.at(idx);
                            nids.push_back(nid);
                            dist += self.graph->length(nid);
                        }
                        nid = self.nodes.at(idx1);
                        nids.push_back(nid);
                        dist += off1;
                    }
                }
                auto p = Path(self.graph, dist, nids, off0, off1);
                auto round_scale = self.graph->round_scale();
                if (round_scale) {
                    p.round(*round_scale);
                }
                return p;
            },
            "start"_a, "end"_a)
        //
        ;

    // ZigzagPath
    py::class_<ZigzagPath, Path>(m, "ZigzagPath", py::module_local(),
                                 py::dynamic_attr()) //
        //
        .def_property_readonly(
            "graph", [](const ZigzagPath &self) { return self.graph; })
        .def_property_readonly("dist",
                               [](const ZigzagPath &self) { return self.dist; })
        .def_property_readonly("nodes",
                               [](const ZigzagPath &self) {
                                   return self.graph->__node_ids(self.nodes);
                               })
        .def_property_readonly(
            "directions",
            [](const ZigzagPath &self) { return self.directions; })
        .def("to_dict",
             [](const ZigzagPath &self) {
                 py::dict ret;
                 ret["dist"] = self.dist;
                 py::list nodes;
                 for (auto &node : self.graph->__node_ids(self.nodes)) {
                     nodes.append(node);
                 }
                 ret["nodes"] = nodes;
                 py::list dirs;
                 for (auto d : self.directions) {
                     dirs.append(d);
                 }
                 ret["directions"] = dirs;
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

    py::class_<Sequences>(m, "Sequences", py::module_local(),
                          py::dynamic_attr()) //
        .def_property_readonly(
            "graph", [](const Sequences &self) { return self.graph; },
            rvp::reference_internal)
        //
        ;

    py::class_<Endpoints>(m, "Endpoints", py::module_local(),
                          py::dynamic_attr()) //
        .def_property_readonly(
            "graph", [](const Endpoints &self) { return self.graph; },
            rvp::reference_internal)
        .def_property_readonly(
            "is_wgs84", [](const Endpoints &self) { return self.is_wgs84; },
            rvp::reference_internal)
        //
        ;

    py::class_<UbodtRecord>(m, "UbodtRecord", py::module_local(),
                            py::dynamic_attr()) //
        .def(py::init<int64_t, int64_t, int64_t, int64_t, double>(),
             "source_road"_a, "target_road"_a, //
             "source_next"_a, "target_prev"_a, //
             "cost"_a)
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def_property_readonly(
            "source_road",
            [](const UbodtRecord &self) { return self.source_road; })
        .def_property_readonly(
            "target_road",
            [](const UbodtRecord &self) { return self.target_road; })
        .def_property_readonly(
            "source_next",
            [](const UbodtRecord &self) { return self.source_next; })
        .def_property_readonly(
            "target_prev",
            [](const UbodtRecord &self) { return self.target_prev; })
        .def_property_readonly(
            "cost", [](const UbodtRecord &self) { return self.cost; })
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
                     // TODO, prevs, dists
                 }
                 auto kv = py::cast(self).attr("__dict__");
                 for (const py::handle &k : kv) {
                     ret[k] = kv[k];
                 }
                 return ret;
             })
        //
        ;

    py::class_<ZigzagPathGenerator>(m, "ZigzagPathGenerator",
                                    py::module_local(),
                                    py::dynamic_attr()) //
                                                        //
        .def(py::init<>())
        //
        .def("cutoff",
             [](const ZigzagPathGenerator &self) { return self.cutoff; })
        .def("source",
             [](const ZigzagPathGenerator &self) {
                 std::optional<std::string> source;
                 if (self.ready()) {
                     source = self.graph->__node_id(*self.source);
                 }
                 return source;
             })
        .def("prevs",
             [](const ZigzagPathGenerator &self) {
                 using State = std::tuple<std::string, int>;
                 std::unordered_map<State, State> ret;
                 if (!self.ready()) {
                     return ret;
                 }
                 for (auto &kv : self.prevs) {
                     ret.emplace(std::make_tuple(self.graph->__node_id(
                                                     std::get<0>(kv.first)),
                                                 std::get<1>(kv.first)),
                                 std::make_tuple(self.graph->__node_id(
                                                     std::get<0>(kv.second)),
                                                 std::get<1>(kv.second)));
                 }
                 return ret;
             })
        .def("dists",
             [](const ZigzagPathGenerator &self) {
                 using State = std::tuple<std::string, int>;
                 std::unordered_map<State, double> ret;
                 if (!self.ready()) {
                     return ret;
                 }
                 for (auto &kv : self.dists) {
                     ret.emplace(std::make_tuple(self.graph->__node_id(
                                                     std::get<0>(kv.first)),
                                                 std::get<1>(kv.first)),
                                 ROUND(kv.second, self.graph->round_scale()));
                 }
                 return ret;
             })
        .def("destinations",
             [](const ZigzagPathGenerator &self)
                 -> std::vector<std::tuple<double, std::string>> {
                 if (!self.ready()) {
                     return {};
                 }
                 auto node2dist = std::unordered_map<std::string, double>{};
                 for (auto &kv : self.dists) {
                     auto node = self.graph->__node_id(std::get<0>(kv.first));
                     double dist = kv.second;
                     auto itr = node2dist.find(node);
                     if (itr == node2dist.end() || dist < itr->second) {
                         node2dist[node] = dist;
                     }
                 }
                 auto ret = std::vector<std::tuple<double, std::string>>{};
                 ret.reserve(node2dist.size());
                 for (auto &kv : node2dist) {
                     ret.push_back(
                         std::make_tuple(std::get<1>(kv), std::get<0>(kv)));
                 }
                 std::sort(ret.begin(), ret.end(),
                           [](const auto &n1, const auto &n2) {
                               return std::get<0>(n1) < std::get<0>(n2);
                           });
                 return ret;
             })
        .def("paths",
             [](const ZigzagPathGenerator &self) -> std::vector<ZigzagPath> {
                 if (!self.ready()) {
                     return {};
                 }
                 std::unordered_map<int64_t, ZigzagPath> node2path;
                 for (auto &kv : self.prevs) {
                     auto path = ZigzagPathGenerator::Path(
                         kv.first, *self.source, self.graph, self.prevs,
                         self.dists);
                     if (!path) {
                         continue;
                     }
                     auto dst = path->nodes.back();
                     auto itr = node2path.find(dst);
                     if (itr == node2path.end() ||
                         path->dist < itr->second.dist) {
                         node2path[dst] = std::move(*path);
                     }
                 }
                 auto paths = std::vector<ZigzagPath>{};
                 paths.reserve(node2path.size());
                 for (auto &kv : node2path) {
                     paths.push_back(kv.second);
                 }
                 std::sort(paths.begin(), paths.end(),
                           [](const auto &p1, const auto &p2) {
                               return p1.dist > p2.dist;
                           });
                 return paths;
             })
        .def("path",
             [](const ZigzagPathGenerator &self,
                const std::string &node) -> std::optional<ZigzagPath> {
                 if (!self.ready()) {
                     return {};
                 }
                 auto node_idx = self.graph->__node_id(node);
                 if (!node_idx) {
                     return {};
                 }
                 std::optional<ZigzagPath> path1;
                 auto state1 = std::make_tuple(*node_idx, 1);
                 if (self.prevs.count(state1)) {
                     path1 = ZigzagPathGenerator::Path(state1, *self.source,
                                                       self.graph, //
                                                       self.prevs, self.dists);
                 }
                 auto state2 = std::make_tuple(*node_idx, -1);
                 std::optional<ZigzagPath> path2;
                 if (self.prevs.count(state2)) {
                     path2 = ZigzagPathGenerator::Path(state2, *self.source,
                                                       self.graph, //
                                                       self.prevs, self.dists);
                 }
                 if (path1 && path2) {
                     return path1->dist < path2->dist ? path1 : path2;
                 }
                 return path1 ? path1 : path2;
             })
        .def("to_dict",
             [](const ZigzagPathGenerator &self) {
                 py::dict ret;
                 if (self.ready()) {
                     ret["cutoff"] = self.cutoff;
                     ret["source"] = self.graph->__node_id(*self.source);
                     // TODO, prevs, dists
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
        .def_property_readonly("sibs_under_next", &DiGraph::sibs_under_next)
        .def_property_readonly("sibs_under_prev", &DiGraph::sibs_under_prev)
        .def_property_readonly("nodes", &DiGraph::nodes,
                               rvp::reference_internal)
        .def_property_readonly("edges", &DiGraph::edges,
                               rvp::reference_internal)
        //
        .def_property_readonly("indexer",
                               py::overload_cast<>(&DiGraph::indexer),
                               rvp::reference_internal)
        //
        .def("predecessors", &DiGraph::predecessors, "id"_a)
        .def("successors", &DiGraph::successors, "id"_a)
        //
        .def("encode_sinks", &DiGraph::encode_sinks, "sinks"_a)
        .def("encode_bindings", &DiGraph::encode_bindings, "bindings"_a)
        .def("encode_sequences", &DiGraph::encode_sequences, "sequences"_a)
        .def("encode_endpoints", &DiGraph::encode_endpoints, "endpoints"_a,
             py::kw_only(), "is_wgs84"_a = true)
        .def("encode_ubodt", &DiGraph::encode_ubodt, //
             "source_road"_a,                        //
             "target_road"_a,                        //
             "source_next"_a,                        //
             "target_prev"_a,                        //
             "cost"_a)
        // shortest paths
        .def(
            "shortest_path",
            [](const DiGraph &self,
               const std::string &source,           //
               const std::string &target,           //
               double cutoff,                       //
               std::optional<double> source_offset, //
               std::optional<double> target_offset, //
               const Sinks *sinks,                  //
               const Endpoints *endpoints) {
                return self.shortest_path(source, target, cutoff,       //
                                          source_offset, target_offset, //
                                          sinks,                        //
                                          endpoints);
            },
            "source"_a,                       //
            "target"_a,                       //
            py::kw_only(),                    //
            "cutoff"_a,                       //
            "source_offset"_a = std::nullopt, //
            "target_offset"_a = std::nullopt, //
            "sinks"_a = nullptr,              //
            "endpoints"_a = nullptr,          //
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
        // zigzag path
        .def(
            "shortest_zigzag_path",
            [](const DiGraph &self,       //
               const std::string &source, //
               const std::string &target, //
               double cutoff,             //
               int direction) {
                return self.shortest_zigzag_path(source, target, //
                                                 cutoff, direction);
            },             //
            "source"_a,    //
            "target"_a,    //
            py::kw_only(), //
            "cutoff"_a,    //
            "direction"_a = 0)
        .def(
            "shortest_zigzag_path",
            [](const DiGraph &self,       //
               const std::string &source, //
               double cutoff,             //
               int direction) {
                ZigzagPathGenerator generator(&self, cutoff);
                self.shortest_zigzag_path(source, {}, cutoff, //
                                          direction, &generator);
                return generator;
            },             //
            "source"_a,    //
            py::kw_only(), //
            "cutoff"_a,    //
            "direction"_a = 0)
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
        .def("build_ubodt",
             py::overload_cast<double, int, int>(&DiGraph::build_ubodt,
                                                 py::const_),
             "thresh"_a, py::kw_only(), //
             "pool_size"_a = 1,         //
             "nodes_thresh"_a = 100)
        .def("build_ubodt",
             py::overload_cast<int64_t, double>(&DiGraph::build_ubodt,
                                                py::const_),
             "source"_a, "thresh"_a)
        //
        ;

    py::class_<ShortestPathWithUbodt>(m, "ShortestPathWithUbodt",
                                      py::module_local(),
                                      py::dynamic_attr()) //
        .def(py::init<const DiGraph *, const std::vector<UbodtRecord> &>(),
             "graph"_a, "ubodt"_a)
        .def(py::init<const DiGraph *, double, int, int>(), //
             "graph"_a, "thresh"_a, py::kw_only(),          //
             "pool_size"_a = 1,                             //
             "nodes_thresh"_a = 100)
        .def(py::init<const DiGraph *, const std::string &>(), //
             "graph"_a, "path"_a)
        //
        .def(
            "load_ubodt",
            [](ShortestPathWithUbodt &self, const std::string &path) {
                return self.load_ubodt(path);
            },
            "path"_a)
        .def(
            "load_ubodt",
            [](ShortestPathWithUbodt &self,
               const std::vector<UbodtRecord> &rows) {
                return self.load_ubodt(rows);
            },
            "rows"_a)
        .def(
            "dump_ubodt",
            [](const ShortestPathWithUbodt &self) { return self.dump_ubodt(); })
        .def("dump_ubodt",
             [](const ShortestPathWithUbodt &self, const std::string &path) {
                 return self.dump_ubodt(path);
             })
        .def("size", &ShortestPathWithUbodt::size)
        //
        .def_static("Load_Ubodt", &ShortestPathWithUbodt::Load_Ubodt, //
                    "path"_a)
        .def_static("Dump_Ubodt", &ShortestPathWithUbodt::Dump_Ubodt, //
                    "ubodt"_a, "path"_a)
        //
        .def("by_source", &ShortestPathWithUbodt::by_source, //
             "source"_a, "cutoff"_a = std::nullopt)
        .def("by_target", &ShortestPathWithUbodt::by_target, //
             "target"_a, "cutoff"_a = std::nullopt)
        .def("path",
             py::overload_cast<const std::string &, const std::string &>(
                 &ShortestPathWithUbodt::path, py::const_),
             "source"_a, "target"_a)
        .def("dist",
             py::overload_cast<const std::string &, const std::string &>(
                 &ShortestPathWithUbodt::dist, py::const_),
             "source"_a, "target"_a)
        //
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
