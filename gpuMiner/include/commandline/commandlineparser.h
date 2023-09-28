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

#ifndef LIBCOMMANDLINE_COMMANDLINEPARSER_H
#define LIBCOMMANDLINE_COMMANDLINEPARSER_H

#include "commandlineoption.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <unordered_map>

namespace libcommandline {

template<class TState>
class CommandLineParser
{
private:
    std::string helpText;
    PositionalArgumentHandler<TState> posArgHandler;
    std::vector<std::unique_ptr<const CommandLineOption<TState>>> options;

    std::unordered_map<char, const CommandLineOption<TState> *> mapShort;
    std::unordered_map<std::string, const CommandLineOption<TState> *> mapLong;

    static int tryProcessOption(
            const std::string &progname, const std::string &optname,
            const CommandLineOption<TState> *opt,
            TState &state, const std::string &argument);

    static int tryProcessOption(
            const std::string &progname, const std::string &optname,
            const CommandLineOption<TState> *opt,
            TState &state);

public:
    CommandLineParser(const CommandLineParser &) = delete;
    CommandLineParser &operator=(const CommandLineParser &) = delete;

    CommandLineParser(CommandLineParser &&) = default;
    CommandLineParser &operator=(CommandLineParser &&) = default;

    /* NOTE: the paser takes ownership of the pointers
     * passed in the 'options' vector */
    CommandLineParser(
            const std::string &helpText,
            const PositionalArgumentHandler<TState> &posArgHandler,
            const std::vector<const CommandLineOption<TState>*> &options)
        : helpText(helpText), posArgHandler(posArgHandler), options(),
          mapShort(), mapLong()
    {
        for (auto &opt : options) {
            mapLong.insert(std::make_pair(opt->getLongName(), opt));

            char shortName = opt->getShortName();
            if (shortName != '\0') {
                mapShort.insert(std::make_pair(shortName, opt));
            }
            this->options.emplace_back(opt);
        }
    }

    void printHelp(const char * const * argv) const;

    int parseArguments(TState &state, const char * const * argv) const;
};

template<class TState>
void CommandLineParser<TState>::printHelp(const char * const * argv) const
{
    const std::string progname(argv[0]);

    std::cout << "usage: " << progname << " [options]";
    if (!posArgHandler.getName().empty()) {
        std::cout << " " << posArgHandler.getName();
    }
    std::cout << std::endl;

    if (!helpText.empty()) {
        std::cout << "  " << helpText << std::endl;
    }
    std::cout << std::endl;

    std::size_t longest = 0;
    std::vector<std::string> names {};
    for (auto &opt : options) {
        auto ptr = opt.get();

        std::string name = ptr->formatOptionName();
        if (name.size() > longest) {
            longest = name.size();
        }
        names.push_back(std::move(name));
    }
    std::cout << "Options:" << std::endl;

    for (std::size_t i = 0; i < options.size(); i++) {
        auto ptr = options[i].get();
        auto &name = names[i];

        std::size_t padding = longest - name.size();
        std::cout << "  " << name
                  << std::string(padding + 2, ' ')
                  << ptr->getHelpText() << std::endl;
    }
    if (!posArgHandler.getHelpText().empty()) {
        std::cout << std::endl;
        std::cout << "Positional arguments:" << std::endl;
        std::cout << "  " << posArgHandler.getHelpText() << std::endl;
    }
}

template<class TState>
int CommandLineParser<TState>::tryProcessOption(
        const std::string &progname, const std::string &optname,
        const CommandLineOption<TState> *opt,
        TState &state, const std::string &argument)
{
    try {
        opt->processOption(state, argument);
    } catch (const ArgumentFormatException& e) {
        std::cerr << progname << ": "
                  << "'" << optname << "': "
                  << "'" << argument << "': "
                  << "invalid argument format: "
                  << e.getMessage() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << progname << ": "
                  << "'-" << optname << "': "
                  << "error while processing option: "
                  << e.what() << std::endl;
        return 1;
    }
    return 0;
}

template<class TState>
int CommandLineParser<TState>::tryProcessOption(
        const std::string &progname, const std::string &optname,
        const CommandLineOption<TState> *opt,
        TState &state)
{
    try {
        opt->processOption(state, std::string());
    } catch (const std::exception &e) {
        std::cerr << progname << ": "
                  << "'-" << optname << "': "
                  << "error while processing option: "
                  << e.what() << std::endl;
        return 1;
    }
    return 0;
}

template<class TState>
int CommandLineParser<TState>::parseArguments(
        TState &state, const char * const * argv) const
{
    const std::string progname(argv[0]);

    bool endOfOptions = false;
    for (auto argp = argv + 1; *argp != nullptr; argp++) {
        const std::string arg(*argp);
        if (endOfOptions || arg.size() < 2 || arg[0] != '-') {
            /* positional argument */
            try {
                posArgHandler.processArgument(state, arg);
            } catch (const ArgumentFormatException& e) {
                std::cerr << progname << ": "
                          << "'" << arg << "': "
                          << "invalid argument format: "
                          << e.getMessage() << std::endl;
                return 1;
            } catch (const std::exception &e) {
                std::cerr << progname << ": "
                          << "'" << arg << "': "
                          << "error while parsing argument: "
                          << e.what() << std::endl;
                return 1;
            }
        } else if (arg[1] != '-') {
            /* short option */
            std::size_t optIndex = 1;
            do {
                char name = arg[optIndex++];
                auto entry = mapShort.find(name);
                if (entry == mapShort.end()) {
                    std::cerr << progname << ": "
                              << "unrecognized option: "
                              "'-" << name << "'" << std::endl;
                    return 1;
                }
                auto opt = entry->second;
                if (opt->doesTakeArgument()) {
                    /* if the option takes an argument, then
                     * the rest of this cmdline arg is passed
                     * to the handler as that argument */
                    std::string argument;
                    if (optIndex == arg.size()) {
                        /* no characters left in this arg,
                         * eat the next cmdline argument */
                        ++argp;
                        if (*argp == nullptr) {
                            std::cerr << progname << ": "
                                      << "'-" << name << "': "
                                      << "option expects an argument"
                                      << std::endl;
                            return 1;
                        }
                        argument = std::string(*argp);
                    } else {
                        argument = arg.substr(optIndex);
                    }
                    int ret = tryProcessOption(progname, { '-', name }, opt,
                                               state, argument);
                    if (ret != 0) return ret;
                    break;
                }
                /* if the option doesn't take an argument,
                 * then process the rest of this arg as
                 * another short option(s) */
                int ret = tryProcessOption(progname, { '-', name }, opt, state);
                if (ret != 0) return ret;
            } while (optIndex < arg.size());
        } else if (arg.size() == 2) {
            /* -- (end of options) */
            endOfOptions = true;
        } else {
            /* long option */
            std::string name;

            /* split arg by equal sign: */
            auto eqSignPos = arg.find('=', 2);
            if (eqSignPos != std::string::npos) {
                name = std::string(arg.begin() + 2, arg.begin() + eqSignPos);
            } else {
                name = std::string(arg.begin() + 2, arg.end());
            }

            auto entry = mapLong.find(name);
            if (entry == mapLong.end()) {
                std::cerr << progname << ": "
                          << "unrecognized option: "
                          "'--" << name << "'" << std::endl;
                return 1;
            }
            auto opt = entry->second;
            if (opt->doesTakeArgument()) {
                std::string argument;
                if (eqSignPos != std::string::npos) {
                    /* take whatever is after the equal
                     * sign as the argument: */
                    argument = { arg.begin() + eqSignPos + 1, arg.end() };
                } else {
                    /* no equal sign was found --
                     * eat the next cmdline argument: */
                    ++argp;
                    if (*argp == nullptr) {
                        std::cerr << progname << ": "
                                  << "'--" << name << "': "
                                  << "option expects an argument"
                                  << std::endl;
                        return 1;
                    }
                    argument = std::string(*argp);
                }
                int ret = tryProcessOption(progname, "--" + name, opt,
                                           state, argument);
                if (ret != 0) return ret;
            } else if (eqSignPos != std::string::npos) {
                std::cerr << progname << ": "
                          << "option "
                          "'--" << name << "'"
                          << " does not take an argument"
                          << std::endl;
                return 1;
            } else {
                int ret = tryProcessOption(progname, "--" + name, opt, state);
                if (ret != 0) return ret;
            }
        }
    } /* for */
    return 0;
}

} // namespace libcommandline

#endif // LIBCOMMANDLINE_COMMANDLINEPARSER_H
