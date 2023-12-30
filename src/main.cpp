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

namespace nano_fmm
{

struct Node
{
    double length{1.0};
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
    DiGraph() = default;
    Node &add_node(const std::string &id, double length = 1.0)
    {
        reset();
        auto &node = nodes_[indexer_.id(id)];
        node.length = length;
        return node;
    }
    Edge &add_edge(const std::string &node0, const std::string &node1)
    {
        reset();
        auto idx0 = indexer_.id(node0);
        auto idx1 = indexer_.id(node1);
        nexts_[idx0].insert(idx1);
        prevs_[idx1].insert(idx0);
        nodes_[idx0];
        nodes_[idx1];
        auto &edge = edges_[std::make_tuple(idx0, idx1)];
        return edge;
    }

    const std::vector<std::string> &nodes() const { return cache().nodes; }
    const std::vector<std::tuple<std::string, std::string>> &edges() const
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

    std::vector<std::tuple<double, std::string>> single_source_dijkstra(
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
        auto ret = std::vector<std::tuple<double, std::string>>{};
        ret.reserve(dmap.size());
        for (auto &pair : dmap) {
            ret.emplace_back(
                std::make_tuple(pair.second, indexer_.id(pair.first)));
        }
        std::sort(ret.begin(), ret.end());
        return ret;
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
    unordered_map<int64_t, Node> nodes_;
    unordered_map<int64_t, unordered_set<int64_t>> nexts_, prevs_;
    unordered_map<std::tuple<int64_t, int64_t>, Edge> edges_;
    mutable Indexer indexer_;
    struct Cache
    {
        std::vector<std::string> nodes;
        std::vector<std::tuple<std::string, std::string>> edges;
    };
    mutable std::optional<Cache> cache_;
    Cache &cache() const
    {
        if (cache_) {
            return *cache_;
        }
        // build nodes, edges
        std::vector<std::string> nodes;
        nodes.reserve(nodes_.size());
        for (auto &pair : nodes_) {
            nodes.push_back(indexer_.id(pair.first));
        }
        std::sort(nodes.begin(), nodes.end());
        std::vector<std::tuple<std::string, std::string>> edges;
        edges.reserve(edges_.size());
        for (auto &pair : edges_) {
            edges.push_back(
                std::make_tuple(indexer_.id(std::get<0>(pair.first)),
                                indexer_.id(std::get<1>(pair.first))));
        }
        cache_ = Cache();
        cache_->nodes = std::move(nodes);
        cache_->edges = std::move(edges);
        return *cache_;
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
        //
        ;

    py::class_<Edge>(m, "Edge", py::module_local(), py::dynamic_attr()) //
        .def(py::init<>())
        .def(
            "__getitem__",
            [](Node &self, const std::string &attr_name) -> py::object {
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
                 py::cast(self).attr(attr_name.c_str()) = obj;
                 return obj;
             })
        //
        ;

    py::class_<DiGraph>(m, "DiGraph", py::module_local(), py::dynamic_attr()) //
        .def(py::init<>())
        //
        .def("add_node", &DiGraph::add_node, "id"_a, py::kw_only(), "length"_a,
             rvp::reference_internal)
        .def("add_edge", &DiGraph::add_edge, "node0"_a, "node1"_a,
             rvp::reference_internal)
        //
        .def("nodes", &DiGraph::nodes)
        .def("edges", &DiGraph::edges)
        //
        .def("predecessors", &DiGraph::predecessors, "id"_a)
        .def("successors", &DiGraph::successors, "id"_a)
        //
        .def("single_source_dijkstra",
             py::overload_cast<const std::string &, double,
                               const std::unordered_set<std::string> *,
                               std::unordered_map<std::string, std::string> *,
                               bool>(&DiGraph::single_source_dijkstra,
                                     py::const_),
             "id"_a, py::kw_only(), //
             "cutoff"_a,            //
             "sinks"_a = nullptr,   //
             "prevs"_a = nullptr,   //
             "reverse"_a = false)
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
