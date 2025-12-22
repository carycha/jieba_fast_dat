#ifndef JIEBA_FAST_DAT_TRIE_HPP
#define JIEBA_FAST_DAT_TRIE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <memory>
#include <algorithm>
#include <pybind11/pybind11.h>
#include "cedarpp.h"

namespace py = pybind11;

namespace jieba_fast_dat {

class DatTrie {
public:
    DatTrie() : total_freq(0.0) {
        // ID 0 is reserved for empty/default tag
        tag_list.push_back("x");
        tag_to_id["x"] = 0;
    }

    // Encoding: 21 bits for freq (max 2,097,151), 10 bits for tag_id (max 1023)
    static const int TAG_BITS = 10;
    static const int TAG_MASK = (1 << TAG_BITS) - 1;
    static const int MAX_FREQ = (1 << (31 - TAG_BITS)) - 1;

    inline int encode(int freq, int tag_id) const {
        if (freq > MAX_FREQ) freq = MAX_FREQ;
        return (freq << TAG_BITS) | (tag_id & TAG_MASK);
    }

    inline int decode_freq(int val) const {
        if (val <= 0) return 0;
        return val >> TAG_BITS;
    }

    inline int decode_tag_id(int val) const {
        if (val <= 0) return 0; // Return 0 (default tag 'x')
        return val & TAG_MASK;
    }

    int get_or_create_tag_id(const std::string& tag) {
        if (tag.empty()) return 0;
        auto it = tag_to_id.find(tag);
        if (it != tag_to_id.end()) return it->second;
        int new_id = static_cast<int>(tag_list.size());
        if (new_id > TAG_MASK) return 0; // Fallback to default
        tag_list.push_back(tag);
        tag_to_id[tag] = new_id;
        return new_id;
    }

    double build(size_t num_keys, const char** keys, const size_t* lengths, const int* freqs) {
        trie_.clear();
        total_freq = 0.0;
        trie_.build(num_keys, keys, lengths, freqs);
        for(size_t i = 0; i < num_keys; ++i) {
            total_freq += decode_freq(freqs[i]);
        }
        return total_freq;
    }

    double build(py::iterable word_freqs_iterable) {
        trie_.clear();
        total_freq = 0.0;
        for (py::handle item : word_freqs_iterable) {
            py::tuple pair = item.cast<py::tuple>();
            std::string word = pair[0].cast<std::string>();
            int freq = pair[1].cast<int>();
            trie_.update(word.c_str(), word.length()) = encode(freq, 0);
            total_freq += static_cast<double>(freq);
        }
        return total_freq;
    }

    void clear() {
        trie_.clear();
        total_freq = 0.0;
        tag_list.clear();
        tag_list.push_back("x");
        tag_to_id.clear();
        tag_to_id["x"] = 0;
    }

    void add_word(const std::string& word, int freq, const std::string& tag = "x") {
        int tag_id = get_or_create_tag_id(tag);
        int old_val = trie_.exactMatchSearch<int>(word.c_str(), word.length());
        if (old_val != cedar::da<int>::CEDAR_NO_VALUE && old_val != cedar::da<int>::CEDAR_NO_PATH) {
            total_freq -= decode_freq(old_val);
        }
        trie_.update(word.c_str(), word.length()) = encode(freq, tag_id);
        total_freq += freq;
    }

    void del_word(const std::string& word) {
        int old_val = trie_.exactMatchSearch<int>(word.c_str(), word.length());
        if (old_val != cedar::da<int>::CEDAR_NO_VALUE && old_val != cedar::da<int>::CEDAR_NO_PATH) {
            total_freq -= decode_freq(old_val);
        }
        trie_.erase(word.c_str(), word.length());
    }

    int search(const std::string& word) const {
        int val = trie_.exactMatchSearch<int>(word.c_str(), word.length());
        return decode_freq(val);
    }

    const std::string& get_tag(const std::string& word) const {
        int val = trie_.exactMatchSearch<int>(word.c_str(), word.length());
        int tag_id = decode_tag_id(val);
        if (tag_id >= 0 && static_cast<size_t>(tag_id) < tag_list.size()) {
            return tag_list[tag_id];
        }
        static const std::string default_tag = "x";
        return default_tag;
    }

    int search(const char* s, size_t len) const {
        int val = trie_.exactMatchSearch<int>(s, len);
        return decode_freq(val);
    }

    int open(const std::string& filename, size_t offset = 0) {
        return trie_.open(filename.c_str(), "rb", offset);
    }

    int save(const std::string& filename) {
        return trie_.save(filename.c_str());
    }

    int save_all(const std::string& filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) return -1;
        const uint32_t magic = 0x4A424441; // "JBDA"
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&total_freq), sizeof(total_freq));
        uint32_t list_size = static_cast<uint32_t>(tag_list.size());
        ofs.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
        for (const auto& tag : tag_list) {
            uint32_t s_size = static_cast<uint32_t>(tag.size());
            ofs.write(reinterpret_cast<const char*>(&s_size), sizeof(s_size));
            ofs.write(tag.data(), s_size);
        }
        char* trie_data = nullptr;
        size_t trie_data_size = 0;
        if (trie_.save_to_memory(&trie_data, &trie_data_size) != 0) return -1;
        uint64_t trie_size_u64 = static_cast<uint64_t>(trie_data_size);
        ofs.write(reinterpret_cast<const char*>(&trie_size_u64), sizeof(trie_size_u64));
        ofs.write(trie_data, trie_data_size);
        std::free(trie_data);
        return 0;
    }

    int load_all(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return -1;
        uint32_t magic = 0;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x4A424441) return -1;
        ifs.read(reinterpret_cast<char*>(&total_freq), sizeof(total_freq));
        if (ifs.gcount() != sizeof(total_freq)) return -1;
        uint32_t list_size = 0;
        ifs.read(reinterpret_cast<char*>(&list_size), sizeof(list_size));
        if (ifs.gcount() != sizeof(list_size)) return -1;
        tag_list.clear();
        tag_to_id.clear();
        for (uint32_t i = 0; i < list_size; ++i) {
            uint32_t s_size = 0;
            ifs.read(reinterpret_cast<char*>(&s_size), sizeof(s_size));
            std::string s(s_size, '\0');
            ifs.read(&s[0], s_size);
            tag_list.push_back(s);
            tag_to_id[s] = static_cast<int>(i);
        }
        uint64_t trie_data_size = 0;
        ifs.read(reinterpret_cast<char*>(&trie_data_size), sizeof(trie_data_size));
        if (ifs.gcount() != sizeof(trie_data_size)) return -1;
        std::vector<char> trie_data(trie_data_size);
        ifs.read(trie_data.data(), trie_data_size);
        if (static_cast<uint64_t>(ifs.gcount()) != trie_data_size) return -1;
        if (trie_.open_from_memory(trie_data.data(), static_cast<size_t>(trie_data_size)) != 0) return -1;
        return 0;
    }

    py::bytes save_to_bytes() const {
        char* data = nullptr;
        size_t data_size = 0;
        if (trie_.save_to_memory(&data, &data_size) != 0) {
            throw std::runtime_error("Failed to save trie to memory");
        }
        std::string result;
        const uint32_t magic = 0x4A424441;
        result.append(reinterpret_cast<const char*>(&magic), sizeof(magic));
        result.append(reinterpret_cast<const char*>(&total_freq), sizeof(total_freq));
        uint32_t list_size = static_cast<uint32_t>(tag_list.size());
        result.append(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
        for (const auto& tag : tag_list) {
            uint32_t s_size = static_cast<uint32_t>(tag.size());
            result.append(reinterpret_cast<const char*>(&s_size), sizeof(s_size));
            result.append(tag.data(), s_size);
        }
        uint64_t trie_size_u64 = static_cast<uint64_t>(data_size);
        result.append(reinterpret_cast<const char*>(&trie_size_u64), sizeof(trie_size_u64));
        result.append(data, data_size);
        std::free(data);
        return py::bytes(result);
    }

    void load_from_bytes(py::bytes data) {
        std::string_view sv = data;
        const char* p = sv.data();
        const char* end = p + sv.size();
        if (sv.size() < sizeof(uint32_t)) throw std::runtime_error("Invalid cache data: too short");
        uint32_t magic = *reinterpret_cast<const uint32_t*>(p); p += sizeof(uint32_t);
        if (magic != 0x4A424441) throw std::runtime_error("Invalid cache data: magic mismatch");
        if (p + sizeof(total_freq) > end) throw std::runtime_error("Invalid cache data: no total_freq");
        total_freq = *reinterpret_cast<const double*>(p); p += sizeof(total_freq);
        if (p + sizeof(uint32_t) > end) throw std::runtime_error("Invalid cache data: no list_size");
        uint32_t list_size = *reinterpret_cast<const uint32_t*>(p); p += sizeof(uint32_t);
        tag_list.clear();
        tag_to_id.clear();
        for (uint32_t i = 0; i < list_size; ++i) {
            if (p + sizeof(uint32_t) > end) throw std::runtime_error("Invalid cache data: tag size truncated");
            uint32_t s_size = *reinterpret_cast<const uint32_t*>(p); p += sizeof(uint32_t);
            if (p + s_size > end) throw std::runtime_error("Invalid cache data: tag data truncated");
            std::string s(p, s_size);
            tag_list.push_back(s);
            tag_to_id[s] = static_cast<int>(i);
            p += s_size;
        }
        if (p + sizeof(uint64_t) > end) throw std::runtime_error("Invalid cache data: no trie size");
        uint64_t trie_size = *reinterpret_cast<const uint64_t*>(p); p += sizeof(uint64_t);
        if (p + trie_size > end) throw std::runtime_error("Invalid cache data: trie data truncated");
        if (trie_.open_from_memory(p, static_cast<size_t>(trie_size)) != 0) {
            throw std::runtime_error("Cedar open_from_memory failed");
        }
    }

    size_t num_keys() const { return trie_.num_keys(); }

    void extract_words(std::vector<std::pair<std::string, int>>& words_with_freqs) {
        size_t count = trie_.num_keys();
        if (count == 0) return;
        words_with_freqs.reserve(count);
        char key_buf[1024];
        cedar::npos_t from = 0;
        size_t len_p = 0;
        for (int val = trie_.begin(from, len_p);
             val != cedar::da<int>::CEDAR_NO_PATH;
             val = trie_.next(from, len_p)) {
            trie_.suffix(key_buf, len_p, from);
            words_with_freqs.emplace_back(std::string(key_buf, len_p), val);
        }
    }

    void update_word_tag_tab(py::dict new_tab) {
        for (auto item : new_tab) {
            std::string word = item.first.cast<std::string>();
            std::string tag = item.second.cast<std::string>();
            int tag_id = get_or_create_tag_id(tag);
            int old_val = trie_.exactMatchSearch<int>(word.c_str(), word.length());
            int freq = 3;
            if (old_val != cedar::da<int>::CEDAR_NO_VALUE && old_val != cedar::da<int>::CEDAR_NO_PATH) {
                freq = decode_freq(old_val);
            }
            trie_.update(word.c_str(), word.length()) = encode(freq, tag_id);
        }
    }

    const cedar::da<int>& trie_ref() const { return trie_; }
    cedar::da<int>& trie_ref() { return trie_; }

    std::vector<std::string> tag_list;
    std::unordered_map<std::string, int> tag_to_id;
    double total_freq;

private:
    cedar::da<int> trie_;
};

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_TRIE_HPP
