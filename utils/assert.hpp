// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "fmt/core.h"
#include "utils/env.hpp"

namespace tt
{
template <typename A, typename B>
struct OStreamJoin
{
    OStreamJoin(A const& a, B const& b, char const* delim = " ") : a(a), b(b), delim(delim) {}
    A const& a;
    B const& b;
    char const* delim;
};

template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, tt::OStreamJoin<A, B> const& join)
{
    os << join.a << join.delim << join.b;
    return os;
}
}  // namespace tt

namespace tt::assert
{
inline std::string demangle(const char* str)
{
    size_t size = 0;
    int status = 0;
    std::string rt(1025, '\0');
    if (1 == sscanf(str, "%*[^(]%*[^_]%1024[^)+]", &rt[0]))
    {
        char* v = abi::__cxa_demangle(&rt[0], nullptr, &size, &status);
        if (v)
        {
            std::string result(v);
            free(v);
            return result;
        }
    }
    return str;
}

/**
 * @brief Get the current call stack
 * @param[out] bt Save Call Stack
 * @param[in] size Maximum number of return layers
 * @param[in] skip Skip the number of layers at the top of the stack
 */
inline std::vector<std::string> backtrace(int size, int skip)
{
    std::vector<std::string> bt;
    void** array = (void**)malloc((sizeof(void*) * size));
    size_t s = ::backtrace(array, size);
    char** strings = backtrace_symbols(array, s);
    if (strings == NULL)
    {
        std::cout << "backtrace_symbols error." << std::endl;
        return bt;
    }

    for (size_t i = skip; i < s; ++i)
    {
        bt.push_back(demangle(strings[i]));
    }
    free(strings);
    free(array);

    return bt;
}

/**
 * @brief String to get current stack information
 * @param[in] size Maximum number of stacks
 * @param[in] skip Skip the number of layers at the top of the stack
 * @param[in] prefix Output before stack information
 */
inline std::string backtrace_to_string(int size, int skip, const std::string& prefix)
{
    std::vector<std::string> bt = backtrace(size, skip);
    std::stringstream ss;
    for (size_t i = 0; i < bt.size(); ++i)
    {
        ss << prefix << bt[i] << std::endl;
    }
    return ss.str();
}

inline void tt_assert_message(std::ostream&) {}

template <typename T, typename... Ts>
void tt_assert_message(std::ostream& os, T const& t, Ts const&... ts)
{
    os << t << std::endl;
    tt_assert_message(os, ts...);
}

template <typename... Args>
void format_as_fallback(std::stringstream& ss, Args const&... args)
{
    if constexpr (sizeof...(Args) > 0)
    {
        auto tuple = std::tuple<const Args&...>(args...);
        using First = std::decay_t<decltype(std::get<0>(tuple))>;

        if constexpr (std::is_convertible_v<First, std::string_view> && sizeof...(Args) > 1)
        {
            auto format_args = std::apply(
                [&](auto const& fmt, auto const&... rest)
                {
                    if constexpr ((fmt::is_formattable<std::decay_t<decltype(rest)>>::value && ...))
                    {
                        return fmt::format(fmt::runtime(fmt), rest...);
                    }
                    else
                    {
                        return std::string();
                    }
                },
                tuple);

            if (!format_args.empty())
            {
                ss << format_args;
                return;
            }
        }

        // Fallback
        ss << "info:\n";
        tt_assert_message(ss, args...);
    }
}

template <bool fmt_present, typename... Ts>
void tt_assert(
    char const* file,
    int line,
    char const* assert_type,
    char const* condition_str,
    std::string_view format_str,
    Ts const&... messages)
{
    std::stringstream trace_message_ss = {};
    trace_message_ss << assert_type << " @ " << file << ":" << line << ": " << condition_str << std::endl;
    if constexpr (fmt_present)
    {
        trace_message_ss << fmt::format(fmt::runtime(format_str), messages...);
    }
    else if constexpr (sizeof...(messages) > 0)
    {
        trace_message_ss << "info:" << std::endl;
        format_as_fallback(trace_message_ss, messages...);
    }

    if (env_as<bool>("TT_ASSERT_ABORT"))
    {
        // Just abort, the signal handler will print the stack trace.
        abort();
    }

    trace_message_ss << "\nbacktrace:\n";
    trace_message_ss << tt::assert::backtrace_to_string(100, 3, " --- ");
    trace_message_ss << std::flush;

    throw std::runtime_error(trace_message_ss.str());
}

}  // namespace tt::assert

/**
 * @brief Function to mark a code path as unreachable
 *
 * This function is marked [[noreturn]] and will trigger compiler-specific
 * mechanisms to indicate unreachable code. It can be used to tell the compiler
 * that a certain branch will not be taken, helping with optimization.
 */
[[noreturn]] inline void unreachable()
{
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__)  // MSVC
    __assume(false);
#else  // GCC, Clang
    __builtin_unreachable();
#endif
}

#define TT_ASSERT(condition, ...)                                                             \
    __builtin_expect(not(condition), 0)                                                       \
        ? ::tt::assert::tt_assert<false>(                                                     \
              __FILE__, __LINE__, "TT_ASSERT", #condition, std::string_view{}, ##__VA_ARGS__) \
        : void()
#define TT_LOG_ASSERT(condition, f, ...)                                                               \
    __builtin_expect(not(condition), 0)                                                                \
        ? ::tt::assert::tt_assert<true>(__FILE__, __LINE__, "TT_ASSERT", #condition, f, ##__VA_ARGS__) \
        : void()
#define TT_THROW(...) \
    ::tt::assert::tt_assert<false>(__FILE__, __LINE__, "TT_THROW", "tt::exception", std::string_view{}, ##__VA_ARGS__)

#ifndef DEBUG
// Do nothing in release mode.
#define TT_DBG_ASSERT(condition, ...) ((void)0)
#else
#define TT_DBG_ASSERT(condition, ...)                                                             \
    __builtin_expect(not(condition), 0)                                                           \
        ? ::tt::assert::tt_assert<false>(                                                         \
              __FILE__, __LINE__, "TT_DBG_ASSERT", #condition, std::string_view{}, ##__VA_ARGS__) \
        : void()
#endif
