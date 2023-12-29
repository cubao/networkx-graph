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

struct DiGraph
{

  private:
    unordered_map<int64_t, Node> nodes_;
    unordered_map<int64_t, unordered_set<int64_t>> nexts_, prevs_;
    unordered_map<std::pair<int64_t, int64_t>, Edge> edges_;
    mutable Indexer indexer;

    using IndexMap = unordered_map<int64_t, int64_t>;
    using DistanceMap = unordered_map<int64_t, double>;

    void single_source_dijkstra(
        const std::string &start, double cutoff,
        const std::unordered_set<std::string> *sinks = nullptr,
        std::unordered_map<std::string, std::string> *prevs = nullptr) const
    {
    }

    void single_source_dijkstra(
        int64_t start, double cutoff,
        const unordered_set<int64_t> *sinks = nullptr,
        unordered_map<int64_t, int64_t> *prevs = nullptr) const
    {
    }

    void single_source_upperbound_dijkstra(int64_t s, double delta, //
                                           IndexMap &pmap,
                                           DistanceMap &dmap) const
    {
        Heap Q;
        Q.push(s, -nodes_.at(s).length);
        pmap.insert({s, s});
        dmap.insert({s, 0});
        while (!Q.empty()) {
            HeapNode node = Q.top();
            Q.pop();
            if (node.value > delta)
                break;
            auto u = node.index;
            auto itr = nexts_.find(u);
            if (itr == nexts_.end()) {
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
                    if (c <= delta) {
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
