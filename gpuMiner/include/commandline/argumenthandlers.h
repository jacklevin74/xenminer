/*
 * Copyright (C) 2015, Ondrej Mosnacek <omosnacek@gmail.com>
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation: either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBCOMMANDLINE_ARGUMENTHANDLERS_H
#define LIBCOMMANDLINE_ARGUMENTHANDLERS_H

#include "commandlineoption.h"

namespace libcommandline {

template<class T>
struct NumericParsingTraits {};

template<>
struct NumericParsingTraits<int>
{
    static constexpr const char *typeName = "int";

    static int parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stoi(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<long>
{
    static constexpr const char *typeName = "long";

    static long parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stol(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<unsigned long>
{
    static constexpr const char *typeName = "unsigned long";

    static unsigned long parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stoul(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<long long>
{
    static constexpr const char *typeName = "long long";

    static long long parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stoll(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<unsigned long long>
{
    static constexpr const char *typeName = "unsigned long long";

    static unsigned long long parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stoull(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<float>
{
    static constexpr const char *typeName = "float";

    static float parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stof(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<double>
{
    static constexpr const char *typeName = "double";

    static double parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stod(arg, endpos);
    }
};

template<>
struct NumericParsingTraits<long double>
{
    static constexpr const char *typeName = "long double";

    static long double parse(const std::string &arg, std::size_t *endpos)
    {
        return std::stold(arg, endpos);
    }
};

template<class TState, class TNum>
static std::function<typename ArgumentOption<TState>::Callback>
    makeNumericHandler(std::function<void(TState &, TNum)> callback)
{
    return [=] (TState &state, const std::string &arg) {
        std::size_t end;
        TNum ret;
        try {
            ret = NumericParsingTraits<TNum>::parse(arg, &end);
        } catch (const std::invalid_argument&) {
            throw ArgumentFormatException(
                        std::string("value not valid for ") +
                        NumericParsingTraits<TNum>::typeName);
        } catch (const std::out_of_range &) {
            throw ArgumentFormatException(
                        std::string("value is out of range for ") +
                        NumericParsingTraits<TNum>::typeName);
        }
        if (end != arg.size()) {
            throw ArgumentFormatException(
                        std::string("value must be a valid ") +
                        NumericParsingTraits<TNum>::typeName);
        }
        callback(state, ret);
    };
}

template<class TState, typename T>
static std::function<void(TState &, T)> makeCheckHandler(
        std::function<bool(T)> predicate,
        const std::string &failMessage,
        std::function<void(TState &, T)> callback)
{
    return [=, &failMessage] (TState &state, T arg) {
        if (!predicate(arg)) {
            throw ArgumentFormatException(failMessage);
        }
        callback(state, arg);
    };
}

template<class TState>
static std::function<void(TState &, const std::string &)>
makeArgumentWithOptionsHandler(
        std::function<void(TState &, const std::string &,
            const std::string &)> callback,
        char delim = ':')
{
    return [=] (TState &state, const std::string &arg) {
        std::string name, opts;
        std::size_t delimPos = arg.find(delim);
        if (delimPos == std::string::npos) {
            name.assign(arg);
            opts.clear();
        } else {
            name.assign(arg.begin(), arg.begin() + delimPos);
            opts.assign(arg.begin() + delimPos + 1, arg.end());
        }
        callback(state, name, opts);
    };
}

} // namespace libcommandline

#endif // LIBCOMMANDLINE_ARGUMENTHANDLERS_H
