#pragma once

// https://github.com/microsoft/vscode-cpptools/issues/9692
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "rapidjson/document.h"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef NANO_FMM_DISABLE_UNORDERED_DENSE
#define NANO_FMM_DISABLE_UNORDERED_DENSE 0
#endif

#if !NANO_FMM_DISABLE_UNORDERED_DENSE
#include "ankerl/unordered_dense.h"
#endif

namespace nano_fmm
{
#if NANO_FMM_DISABLE_UNORDERED_DENSE
template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Equal = std::equal_to<Key>>
using unordered_map = std::unordered_map<Key, Value, Hash, Equal>;
template <typename Value, typename Hash = std::hash<Value>,
          typename Equal = std::equal_to<Value>>
using unordered_set = std::unordered_set<Value, Hash>;
#else
template <typename Key, typename Value,
          typename Hash = ankerl::unordered_dense::hash<Key>,
          typename Equal = std::equal_to<Key>>
using unordered_map = ankerl::unordered_dense::map<Key, Value, Hash, Equal>;
template <typename Value, typename Hash = ankerl::unordered_dense::hash<Value>,
          typename Equal = std::equal_to<Value>>
using unordered_set = ankerl::unordered_dense::set<Value, Hash, Equal>;
#endif

// Use the CrtAllocator, because the MemoryPoolAllocator is broken on ARM
// https://github.com/miloyip/rapidjson/issues/200, 301, 388
using RapidjsonAllocator = rapidjson::CrtAllocator;
using RapidjsonDocument =
    rapidjson::GenericDocument<rapidjson::UTF8<>, RapidjsonAllocator>;
using RapidjsonValue =
    rapidjson::GenericValue<rapidjson::UTF8<>, RapidjsonAllocator>;
} // namespace nano_fmm
