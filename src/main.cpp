#include "indexer.hpp"
#include "types.hpp"
#include <pybind11/pybind11.h>

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

    void single_source_upperbound_dijkstra(int64_t source, double distance, //
                                           IndexMap &predecessor_map,
                                           DistanceMap &distance_map) const
    {
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
