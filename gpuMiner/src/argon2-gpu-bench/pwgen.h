#ifndef PWGEN_H
#define PWGEN_H

#include <random>
#include <string>
#include <chrono>

class PasswordGenerator
{
public:
    virtual void nextPassword(const void *&pw, std::size_t &pwSize) = 0;
};
namespace aaa{
    constexpr char hex_chars[] = "0123456789abcdef";
}
class DummyPasswordGenerator : public PasswordGenerator
{
private:
    std::mt19937 gen;
    std::string currentPw;

    static constexpr std::size_t PASSWORD_LENGTH = 64;

public:
    DummyPasswordGenerator()
        : gen(std::chrono::system_clock::now().time_since_epoch().count())
    {
        currentPw.resize(PASSWORD_LENGTH);
        // Generate a random hex string
        for (std::size_t i = 0; i < PASSWORD_LENGTH; i++) {
            currentPw[i] = aaa::hex_chars[gen()&15];
        }
    }

    void nextPassword(const void *&pw, std::size_t &pwSize) override
    {
        // Modify one character randomly
        for(char& c : currentPw)
            c = aaa::hex_chars[gen() & 15];
        pw = currentPw.data();
        pwSize = currentPw.size();
    }
};

#endif // PWGEN_H
