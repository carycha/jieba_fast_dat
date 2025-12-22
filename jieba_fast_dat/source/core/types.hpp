#ifndef JIEBA_FAST_DAT_TYPES_HPP
#define JIEBA_FAST_DAT_TYPES_HPP

#include <string>
#include <vector>

namespace jieba_fast_dat {

// Pair class to replace Python-side pair for performance
class Pair {
public:
    std::string word;
    std::string flag;

    Pair(std::string w, std::string f) : word(std::move(w)), flag(std::move(f)) {}

    std::string toString() const {
        return word + "/" + flag;
    }

    std::string repr() const {
        return "pair('" + word + "', '" + flag + "')";
    }

    bool operator<(const Pair& other) const {
        if (word != other.word) return word < other.word;
        return flag < other.flag;
    }

    bool operator==(const Pair& other) const {
        return word == other.word && flag == other.flag;
    }
};

struct ViterbiResult {
    double prob;
    std::vector<Pair> word_tags;
};

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_TYPES_HPP
