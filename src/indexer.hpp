#pragma once

#include "types.hpp"
#include <optional>
#include <string>

#include "spdlog/spdlog.h"
// fix exposed macro 'GetObject' from wingdi.h (included by spdlog.h) under
// windows, see https://github.com/Tencent/rapidjson/issues/1448
#ifdef GetObject
#undef GetObject
#endif

namespace nano_fmm
{
struct Indexer
{
    Indexer() = default;
    Indexer(const std::map<std::string, int64_t> &index) { this->index(index); }
    bool contains(const std::string &id) const
    {
        return str2int_.find(id) != str2int_.end();
    }
    bool contains(int64_t id) const
    {
        return int2str_.find(id) != int2str_.end();
    }
    std::optional<std::string> get_id(int64_t id) const
    {
        auto itr = int2str_.find(id);
        if (itr == int2str_.end()) {
            return {};
        }
        return itr->second;
    }
    std::optional<int64_t> get_id(const std::string &id) const
    {
        auto itr = str2int_.find(id);
        if (itr == str2int_.end()) {
            return {};
        }
        return itr->second;
    }

    // get str id
    std::string id(int64_t id) const { return int2str_.at(id); }
    // get str id (with auto setup)
    std::string id(int64_t id)
    {
        auto itr = int2str_.find(id);
        if (itr != int2str_.end()) {
            return itr->second;
        }
        int round = 0;
        auto id_str = fmt::format("{}", id);
        auto str_id = id_str;
        while (str2int_.count(str_id)) {
            ++round;
            str_id = fmt::format("{}/{}", id_str, round);
        }
        index(str_id, id);
        return str_id;
    }
    // get int id
    int64_t id(const std::string &id) const { return str2int_.at(id); }
    // get int id (with auto setup)
    int64_t id(const std::string &id)
    {
        auto itr = str2int_.find(id);
        if (itr != str2int_.end()) {
            return itr->second;
        }
        try {
            // '44324' -> 44324
            // 'w44324' -> 44324
            int64_t ii =
                id[0] == 'w' ? std::stoll(id.substr(1)) : std::stoll(id);
            if (index(id, ii)) {
                return ii;
            }
        } catch (...) {
        }
        while (!index(id, id_cursor_)) {
            ++id_cursor_;
        }
        return id_cursor_++;
    }
    // setup str/int id, returns true (setup) or false (skip)
    bool index(const std::string &str_id, int64_t int_id)
    {
        if (str2int_.count(str_id) || int2str_.count(int_id)) {
            return false;
        }
        str2int_.emplace(str_id, int_id);
        int2str_.emplace(int_id, str_id);
        return true;
    }
    bool index(const std::map<std::string, int64_t> &index)
    {
        bool succ = true;
        for (auto &kv : index) {
            succ &= this->index(kv.first, kv.second);
        }
        return succ;
    }
    std::map<std::string, int64_t> index() const
    {
        return {str2int_.begin(), str2int_.end()};
    }

  private:
    unordered_map<std::string, int64_t> str2int_;
    unordered_map<int64_t, std::string> int2str_;
    int64_t id_cursor_{1000000};
};
} // namespace nano_fmm
