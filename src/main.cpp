#include <pybind11/pybind11.h>

#include "heap.hpp"
#include "indexer.hpp"
#include "types.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;

namespace nano_fmm
{

struct Node
{
    double length{0.0};
};

struct Edge
{
};

struct Route
{
    double dist;
    std::vector<std::string> path;
};

struct DiGraph
{

    std::vector<std::pair<double, std::string>> single_source_dijkstra(
        const std::string &start, double cutoff,
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
                               pmap, dmap, sinks_ptr.get());
        if (prevs) {
            for (auto pair : pmap) {
                (*prevs)[indexer_.id(pair.first)] = indexer_.id(pair.second);
            }
        }
        auto ret = std::vector<std::pair<double, std::string>>{};
        ret.reserve(dmap.size());
        for (auto &pair : dmap) {
            ret.emplace_back(pair.second, indexer_.id(pair.first));
        }
        std::sort(ret.begin(), ret.end(),
                  [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
        return ret;
    }

  private:
    unordered_map<int64_t, Node> nodes_;
    unordered_map<int64_t, unordered_set<int64_t>> nexts_, prevs_;
    unordered_map<std::pair<int64_t, int64_t>, Edge> edges_;
    mutable Indexer indexer_;

    void single_source_dijkstra(
        int64_t start, double cutoff, //
        const unordered_map<int64_t, unordered_set<int64_t>> &jumps,
        unordered_map<int64_t, int64_t> &pmap,
        unordered_map<int64_t, double> &dmap,
        const unordered_set<int64_t> *sinks = nullptr) const
    {
        // https://github.com/cubao/nano-fmm/blob/37d2979503f03d0a2759fc5f110e2b812d963014/src/nano_fmm/network.cpp#L449C67-L449C72
        auto itr = jumps.find(start);
        if (itr == jumps.end()) {
            return;
        }
        Heap Q;
        Q.push(start, 0.0);
        dmap.emplace(start, 0.0);
        for (auto next : itr->second) {
            Q.push(next, 0.0);
            pmap.emplace(next, start);
            dmap.emplace(next, 0.0);
        }
        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            if (node.value > cutoff)
                break;
            auto u = node.index;
            if (sinks && sinks->find(u) != sinks->end()) {
                continue;
            }
            auto itr = jumps.find(u);
            if (itr == jumps.end()) {
                continue;
            }
            double u_cost = nodes_.at(u).length;
            for (auto v : itr->second) {
                auto c = node.value + u_cost;
                auto iter = dmap.find(v);
                if (iter != dmap.end()) {
                    if (c < iter->second) {
                        pmap[v] = u;
                        dmap[v] = c;
                        Q.decrease_key(v, c);
                    };
                } else {
                    if (c <= cutoff) {
                        Q.push(v, c);
                        pmap.insert({v, u});
                        dmap.insert({v, c});
                    }
                }
            }
        }
    }
};
} // namespace nano_fmm

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

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
