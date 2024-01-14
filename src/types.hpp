#pragma once

// https://github.com/microsoft/vscode-cpptools/issues/9692
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ankerl/unordered_dense.h"

namespace nano_fmm
{
template <typename Key, typename Value,
          typename Hash = ankerl::unordered_dense::hash<Key>,
          typename Equal = std::equal_to<Key>>
using unordered_map = ankerl::unordered_dense::map<Key, Value, Hash, Equal>;
template <typename Value, typename Hash = ankerl::unordered_dense::hash<Value>,
          typename Equal = std::equal_to<Value>>
using unordered_set = ankerl::unordered_dense::set<Value, Hash, Equal>;
} // namespace nano_fmm
