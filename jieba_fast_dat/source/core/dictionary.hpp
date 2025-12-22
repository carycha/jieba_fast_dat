#ifndef JIEBA_FAST_DAT_DICTIONARY_HPP
#define JIEBA_FAST_DAT_DICTIONARY_HPP

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include "trie.hpp"

namespace py = pybind11;

namespace jieba_fast_dat {

struct WordRecord {
    std::string word;
    int val;
    bool operator<(const WordRecord& other) const {
        return word < other.word;
    }
};

inline void _load_word_tag_pybind(const std::string& filename, py::dict word_tag_tab_py) {
    word_tag_tab_py.clear();
    std::ifstream file(filename);
    if (!file.is_open()) throw py::value_error("Could not open dictionary file: " + filename);
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }
        size_t first = line.find_first_not_of(" \t\r\n");
        size_t last = line.find_last_not_of(" \t\r\n");
        if (std::string::npos == first || std::string::npos == last) continue;
        line = line.substr(first, (last - first + 1));
        if (line.empty() || line[0] == '#') continue;
        std::string word_str, tag_str = "x";
        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) parts.push_back(part);
        if (parts.empty()) continue;
        word_str = parts[0];
        if (parts.size() > 1) {
            bool is_digit = !parts[1].empty() && std::all_of(parts[1].begin(), parts[1].end(), ::isdigit);
            if (!is_digit) tag_str = parts[1];
        }
        if (parts.size() > 2) tag_str = parts[2];
        word_tag_tab_py[py::str(word_str)] = py::str(tag_str);
    }
}

inline double load_main_dict_from_path_pybind(DatTrie& trie, const std::string& filename, py::dict& main_word_tag_tab_py) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open dictionary file: " + filename);
    std::vector<WordRecord> records;
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }
        const char* p = line.c_str();
        const char* end = p + line.size();
        while (p < end && std::isspace(*p)) p++;
        if (p == end || *p == '#') continue;
        const char* word_start = p;
        while (p < end && !std::isspace(*p)) p++;
        std::string word_str(word_start, p - word_start);
        int freq = 3; std::string tag = "x";
        while (p < end && std::isspace(*p)) p++;
        if (p < end) {
            const char* next_start = p;
            while (p < end && !std::isspace(*p)) p++;
            std::string part(next_start, p - next_start);
            bool is_digit = !part.empty() && std::all_of(part.begin(), part.end(), ::isdigit);
            if (is_digit) {
                freq = std::stoi(part);
                while (p < end && std::isspace(*p)) p++;
                if (p < end) {
                    const char* tag_start = p;
                    while (p < end && !std::isspace(*p)) p++;
                    tag.assign(tag_start, p - tag_start);
                }
            } else tag = part;
        }
        int tag_id = trie.get_or_create_tag_id(tag);
        records.push_back({word_str, trie.encode(freq, tag_id)});
        main_word_tag_tab_py[py::str(word_str)] = py::str(tag);
        size_t len_word = word_str.length();
        const char* str_ptr = word_str.c_str();
        for (size_t i = 0; i < len_word; ) {
            size_t char_len = 1;
            unsigned char c = static_cast<unsigned char>(str_ptr[i]);
            if (c < 0x80) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            else char_len = 1;
            i += char_len;
            if (i < len_word) records.push_back({word_str.substr(0, i), trie.encode(0, 0)});
        }
    }
    std::sort(records.begin(), records.end());
    std::vector<const char*> keys; std::vector<size_t> lengths; std::vector<int> freqs;
    keys.reserve(records.size()); lengths.reserve(records.size()); freqs.reserve(records.size());
    double new_total_freq = 0.0;
    for (size_t i = 0; i < records.size(); ++i) {
        if (i > 0 && records[i].word == records[i-1].word) {
            if (records[i].val > records[i-1].val) {
                keys.back() = records[i].word.c_str();
                lengths.back() = records[i].word.length();
                new_total_freq -= trie.decode_freq(freqs.back());
                freqs.back() = records[i].val;
                new_total_freq += trie.decode_freq(records[i].val);
            }
            continue;
        }
        keys.push_back(records[i].word.c_str());
        lengths.push_back(records[i].word.length());
        freqs.push_back(records[i].val);
        new_total_freq += trie.decode_freq(records[i].val);
    }
    trie.build(keys.size(), keys.data(), lengths.data(), freqs.data());
    trie.total_freq = new_total_freq;
    return new_total_freq;
}

inline double load_userdict_from_path_pybind(DatTrie& trie, const std::string& filename, py::dict& user_word_tag_tab_py, py::object batch_add_force_split_func) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open dictionary file: " + filename);
    std::vector<WordRecord> records;
    std::vector<std::string> force_split_words_to_add;
    std::vector<std::pair<std::string, int>> existing_trie_words;
    trie.extract_words(existing_trie_words);
    for (const auto& pair : existing_trie_words) records.push_back({pair.first, pair.second});
    std::string line; bool first_line = true;
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) line = line.substr(3);
        }
        const char* p = line.c_str(); const char* end = p + line.size();
        while (p < end && std::isspace(*p)) p++;
        if (p == end || *p == '#') continue;
        const char* word_start = p;
        while (p < end && !std::isspace(*p)) p++;
        std::string word_str(word_start, p - word_start);
        int freq = 3; std::string tag = "x";
        while (p < end && std::isspace(*p)) p++;
        if (p < end) {
            const char* next_start = p;
            while (p < end && !std::isspace(*p)) p++;
            std::string part(next_start, p - next_start);
            bool is_digit = !part.empty() && std::all_of(part.begin(), part.end(), ::isdigit);
            if (is_digit) {
                freq = std::stoi(part);
                while (p < end && std::isspace(*p)) p++;
                if (p < end) {
                    const char* tag_start = p;
                    while (p < end && !std::isspace(*p)) p++;
                    tag.assign(tag_start, p - tag_start);
                }
            } else tag = part;
        }
        int tag_id = trie.get_or_create_tag_id(tag);
        records.push_back({word_str, trie.encode(freq, tag_id)});
        user_word_tag_tab_py[py::str(word_str)] = py::str(tag);
        size_t len_word = word_str.length(); const char* str_ptr = word_str.c_str();
        for (size_t i = 0; i < len_word; ) {
            size_t char_len = 1; unsigned char c = static_cast<unsigned char>(str_ptr[i]);
            if (c < 0x80) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            else char_len = 1;
            i += char_len;
            if (i < len_word) records.push_back({word_str.substr(0, i), trie.encode(0, 0)});
        }
        if (freq == 0) force_split_words_to_add.push_back(word_str);
    }
    std::sort(records.begin(), records.end());
    std::vector<const char*> keys; std::vector<size_t> lengths; std::vector<int> freqs;
    keys.reserve(records.size()); lengths.reserve(records.size()); freqs.reserve(records.size());
    double new_total_freq = 0.0;
    for (size_t i = 0; i < records.size(); ++i) {
        if (i > 0 && records[i].word == records[i-1].word) {
            if (records[i].val > records[i-1].val) {} else continue;
            keys.back() = records[i].word.c_str(); lengths.back() = records[i].word.length();
            new_total_freq -= trie.decode_freq(freqs.back()); freqs.back() = records[i].val;
            new_total_freq += trie.decode_freq(records[i].val); continue;
        }
        keys.push_back(records[i].word.c_str()); lengths.push_back(records[i].word.length());
        freqs.push_back(records[i].val); new_total_freq += trie.decode_freq(records[i].val);
    }
    trie.build(keys.size(), keys.data(), lengths.data(), freqs.data());
    trie.total_freq = new_total_freq;
    if (!force_split_words_to_add.empty() && batch_add_force_split_func.ptr() != nullptr && PyCallable_Check(batch_add_force_split_func.ptr())) {
        batch_add_force_split_func(py::cast(force_split_words_to_add));
    }
    return new_total_freq;
}

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_DICTIONARY_HPP
