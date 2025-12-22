#include "segmenter.hpp"
#include "utils.hpp"
#include "viterbi_engine.hpp"
#include "hmm_model.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <atomic>

namespace jieba_fast_dat {

int _calc_pybind(DatTrie& trie, const std::string& sentence, py::dict DAG, py::dict& route, double total) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    const double logtotal = log(total);
    route[py::cast(N)] = py::make_tuple(0.0, 0);

    for(int idx_signed = (int)N - 1; idx_signed >= 0 ; idx_signed--) {
        size_t idx = (size_t)idx_signed;
        double max_freq_val = std::numeric_limits<double>::lowest();
        size_t max_x_val = 0;
        py::object idx_key = py::cast(idx);
        py::list t_list = DAG[idx_key].cast<py::list>();
        size_t t_list_len = py::len(t_list);

        for(size_t i = 0; i < t_list_len; i++) {
            size_t x_val = t_list[i].cast<size_t>();
            size_t start = offsets[idx];
            size_t len = offsets[x_val + 1] - start;
            std::string word(sentence.data() + start, len);
            int fq_val = trie.search(word);
            if (fq_val <= 0) fq_val = 1;
            py::object route_key = py::cast(x_val + 1);
            py::tuple t_tuple = route[route_key].cast<py::tuple>();
            double fq_2_val = t_tuple[0].cast<double>();
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;
            if(fq_last_val > max_freq_val) {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        route[py::cast(idx)] = py::make_tuple(max_freq_val, max_x_val);
    }
    return 1;
}

int _get_DAG_pybind(py::dict DAG, py::dict FREQ, const std::string& sentence) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    for(size_t k = 0; k < N; k++) {
        py::list tmplist;
        size_t start = offsets[k];
        const char* ptr = sentence.data() + start;
        for(size_t i = k; i < N; i++) {
            size_t len = offsets[i + 1] - start;
            std::string word(ptr, len);
            if (FREQ.contains(word)) {
                py::object freq_item = FREQ[py::cast(word)];
                if (!freq_item.is_none() && freq_item.cast<long>()) tmplist.append(i);
            } else break;
        }
        if (py::len(tmplist) == 0) tmplist.append(k);
        DAG[py::cast(k)] = tmplist;
    }
    return 1;
}

int _get_DAG_and_calc_pybind(DatTrie& trie, const std::string& sentence, py::list route, double total) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    std::vector<std::vector<size_t>> DAG(N);
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);

    for(size_t k = 0; k < N; k++) {
        size_t start = offsets[k];
        const char* ptr = sentence.data() + start;
        size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        if (num > 0) {
            size_t current_res_idx = 0;
            for (size_t i = k; i < N && current_res_idx < num; ++i) {
                size_t prefix_len = offsets[i + 1] - start;
                if (prefix_len == results[current_res_idx].length) {
                    if (results[current_res_idx].value > 0) DAG[k].push_back(i);
                    current_res_idx++;
                }
            }
        }
        if(DAG[k].empty()) DAG[k].push_back(k);
    }
    _route[N][0] = 0.0; _route[N][1] = 0.0;
    for(int idx_signed = (int)N - 1; idx_signed >= 0 ; idx_signed--) {
        size_t idx = (size_t)idx_signed;
        double max_freq_val = std::numeric_limits<double>::lowest();
        size_t max_x_val = 0;
        for(size_t x_val : DAG[idx]) {
            size_t start = offsets[idx];
            size_t len = offsets[x_val + 1] - start;
            std::string word(sentence.data() + start, len);
            int fq_val = trie.search(word);
            if (fq_val <= 0) fq_val = 1;
            double fq_2_val = _route[x_val + 1][0];
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;
            if(fq_last_val >= max_freq_val) {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = (double)max_x_val;
    }
    for(size_t i = 0; i <= N; i++) route.append((long)_route[i][1]);
    return 1;
}

py::dict _get_DAG(DatTrie& trie, const std::string& sentence) {
    py::dict DAG;
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    for (size_t k = 0; k < N; k++) {
        py::list tmplist;
        size_t start = offsets[k];
        const char* ptr = sentence.data() + start;
        size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        if (num > 0) {
            size_t current_res_idx = 0;
            for (size_t i = k; i < N && current_res_idx < num; ++i) {
                size_t prefix_len = offsets[i + 1] - start;
                if (prefix_len == results[current_res_idx].length) {
                    if (results[current_res_idx].value > 0) tmplist.append(i);
                    current_res_idx++;
                }
            }
        }
        if (py::len(tmplist) == 0) tmplist.append(k);
        DAG[py::cast(k)] = tmplist;
    }
    return DAG;
}

int _get_freq(DatTrie& trie, py::object word) {
    std::string word_str = word.cast<std::string>();
    int freq = trie.search(word_str);
    return (freq != -1) ? freq : 0;
}

py::list _posseg_cut_DAG_cpp(DatTrie& trie, const std::string& sentence, double total) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return py::list();

    auto model = std::atomic_load(&global_hmm_model);

    struct DAGNode { size_t end_idx; int freq; int tag_id; };
    std::vector<std::vector<DAGNode>> DAG(N);
    std::vector<std::pair<double, size_t>> route(N + 1);
    double logtotal = log(total);
    for (size_t k = 0; k < N; ++k) {
        size_t start = offsets[k];
        const char* ptr = sentence.data() + start;
        size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        size_t current_res_idx = 0;
        for (size_t i = k; i < N && current_res_idx < num; ++i) {
            size_t prefix_len = offsets[i + 1] - start;
            if (prefix_len == results[current_res_idx].length) {
                int val = results[current_res_idx].value;
                int fq = trie.decode_freq(val);
                if (fq > 0) DAG[k].push_back({i, fq, trie.decode_tag_id(val)});
                current_res_idx++;
            }
        }
        if (DAG[k].empty()) DAG[k].push_back({k, 1, 0});
    }
    route[N] = {0.0, 0};
    for (int idx_signed = (int)N - 1; idx_signed >= 0 ; idx_signed--) {
        size_t idx = (size_t)idx_signed;
        double max_freq_val = std::numeric_limits<double>::lowest();
        size_t max_x_val = 0;
        for(const auto& node : DAG[idx]) {
            double fq_last_val = log(static_cast<double>(node.freq)) - logtotal + route[node.end_idx + 1].first;
            if(fq_last_val >= max_freq_val) { max_freq_val = fq_last_val; max_x_val = node.end_idx; }
        }
        route[idx] = {max_freq_val, max_x_val};
    }
    py::list result; size_t x = 0; std::string buf;
    auto process_buffer = [&](const std::string& buffer) {
        if (buffer.empty()) return;
        std::u32string buf_u32 = utf8_to_u32(buffer);
        if (buf_u32.length() == 1) { result.append(Pair(buffer, trie.get_tag(buffer))); return; }
        std::u32string current_block; enum CharType { HAN, ALPHANUM, OTHER }; CharType last_type = OTHER;
        auto get_type = [](char32_t ch) {
            if (ch >= 0x4E00 && ch <= 0x9FD5) return HAN;
            if ((ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9')) return ALPHANUM;
            return OTHER;
        };
        auto flush_block = [&]() {
            if (current_block.empty()) return;
            std::string block_utf8 = u32_to_utf8(current_block);
            if (last_type == HAN) {
                ViterbiResult viterbi_result = posseg_viterbi_impl(current_block);
                for (auto& word_tag : viterbi_result.word_tags) result.append(std::move(word_tag));
            } else if (last_type == ALPHANUM) {
                if (is_english(current_block)) result.append(Pair(block_utf8, "eng"));
                else result.append(Pair(block_utf8, "m"));
            } else result.append(Pair(block_utf8, "x"));
            current_block.clear();
        };
        for (char32_t ch : buf_u32) {
            CharType current_type = get_type(ch);
            if (current_block.empty()) last_type = current_type;
            else if (current_type != last_type) { flush_block(); last_type = current_type; }
            current_block += ch;
        }
        flush_block();
    };
    while (x < N) {
        size_t best_end = route[x].second; size_t y = best_end + 1;
        size_t start = offsets[x]; size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);
        if (y - x == 1) buf += word;
        else {
            if (!buf.empty()) { process_buffer(buf); buf.clear(); }
            int tag_id = 0;
            for (const auto& node : DAG[x]) { if (node.end_idx == best_end) { tag_id = node.tag_id; break; } }
            result.append(Pair(word, trie.tag_list[tag_id]));
        }
        x = y;
    }
    if (!buf.empty()) process_buffer(buf);
    return result;
}

py::list _posseg_cut_DAG_NO_HMM_cpp(DatTrie& trie, const std::string& sentence, double total) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return py::list();
    struct DAGNode { size_t end_idx; int freq; int tag_id; };
    std::vector<std::vector<DAGNode>> DAG(N);
    std::vector<std::pair<double, size_t>> route(N + 1);
    double logtotal = log(total);
    for (size_t k = 0; k < N; ++k) {
        size_t start = offsets[k]; const char* ptr = sentence.data() + start; size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        size_t current_res_idx = 0;
        for (size_t i = k; i < N && current_res_idx < num; ++i) {
            size_t prefix_len = offsets[i + 1] - start;
            if (prefix_len == results[current_res_idx].length) {
                int val = results[current_res_idx].value; int fq = trie.decode_freq(val);
                if (fq > 0) DAG[k].push_back({i, fq, trie.decode_tag_id(val)});
                current_res_idx++;
            }
        }
        if (DAG[k].empty()) DAG[k].push_back({k, 1, 0});
    }
    route[N] = {0.0, 0};
    for (int idx_signed = (int)N - 1; idx_signed >= 0 ; idx_signed--) {
        size_t idx = (size_t)idx_signed; double max_freq_val = std::numeric_limits<double>::lowest(); size_t max_x_val = 0;
        for(const auto& node : DAG[idx]) {
            double fq_last_val = log(static_cast<double>(node.freq)) - logtotal + route[node.end_idx + 1].first;
            if(fq_last_val >= max_freq_val) { max_freq_val = fq_last_val; max_x_val = node.end_idx; }
        }
        route[idx] = {max_freq_val, max_x_val};
    }
    py::list result; size_t x = 0; std::string buf;
    while (x < N) {
        size_t best_end = route[x].second; size_t y = best_end + 1;
        size_t start = offsets[x]; size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);
        if (y - x == 1) {
            unsigned char c = static_cast<unsigned char>(word[0]);
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) buf += word;
            else {
                if (!buf.empty()) { result.append(Pair(buf, "eng")); buf.clear(); }
                int tag_id = 0;
                for (const auto& node : DAG[x]) { if (node.end_idx == best_end) { tag_id = node.tag_id; break; } }
                result.append(Pair(word, trie.tag_list[tag_id]));
            }
        } else {
            if (!buf.empty()) { result.append(Pair(buf, "eng")); buf.clear(); }
            int tag_id = 0;
            for (const auto& node : DAG[x]) { if (node.end_idx == best_end) { tag_id = node.tag_id; break; } }
            result.append(Pair(word, trie.tag_list[tag_id]));
        }
        x = y;
    }
    if (!buf.empty()) result.append(Pair(buf, "eng"));
    return result;
}

py::list _cut_DAG_cpp(DatTrie& trie, const std::string& sentence, double total) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1; if (N == 0) return py::list();
    struct DAGNode { size_t end_idx; int freq; };
    std::vector<std::vector<DAGNode>> DAG(N);
    std::vector<std::pair<double, size_t>> route(N + 1);
    double logtotal = log(total);
    for (size_t k = 0; k < N; ++k) {
        size_t start = offsets[k]; const char* ptr = sentence.data() + start; size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        size_t current_res_idx = 0;
        for (size_t i = k; i < N && current_res_idx < num; ++i) {
            size_t prefix_len = offsets[i + 1] - start;
            if (prefix_len == results[current_res_idx].length) {
                int val = results[current_res_idx].value; int fq = trie.decode_freq(val);
                if (fq > 0) DAG[k].push_back({i, fq});
                current_res_idx++;
            }
        }
        if (DAG[k].empty()) DAG[k].push_back({k, 1});
    }
    route[N] = {0.0, 0};
    for (int i = (int)N - 1; i >= 0; --i) {
        double max_prob = -1e100; size_t best_x = 0;
        for (const auto& node : DAG[i]) {
            double prob = log((double)node.freq) - logtotal + route[node.end_idx + 1].first;
            if (prob > max_prob) { max_prob = prob; best_x = node.end_idx; }
        }
        route[i] = {max_prob, best_x};
    }
    py::list result; size_t x = 0; std::string buf; size_t buf_char_count = 0;
    auto process_buffer = [&]() {
        if (buf.empty()) return;
        if (buf_char_count == 1) result.append(buf);
        else {
            if (trie.search(buf) <= 0) {
                std::u32string buf_u32 = utf8_to_u32(buf);
                std::u32string current_block; bool is_han = false;
                auto flush_sub_block = [&]() {
                    if (current_block.empty()) return;
                    if (is_han) {
                        std::vector<std::string> words = finalseg_viterbi_internal(current_block);
                        for (const auto& w : words) result.append(w);
                    } else {
                        std::u32string sub; bool is_alnum = false;
                        auto flush_alnum = [&]() { if (sub.empty()) return; result.append(u32_to_utf8(sub)); sub.clear(); };
                        for (char32_t ch : current_block) {
                            bool ch_alnum = (ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9');
                            if (sub.empty()) is_alnum = ch_alnum;
                            else if (ch_alnum != is_alnum) { flush_alnum(); is_alnum = ch_alnum; }
                            sub += ch;
                        }
                        flush_alnum();
                    }
                    current_block.clear();
                };
                for (char32_t ch : buf_u32) {
                    bool ch_is_han = (ch >= 0x4E00 && ch <= 0x9FD5);
                    if (current_block.empty()) is_han = ch_is_han;
                    else if (ch_is_han != is_han) { flush_sub_block(); is_han = ch_is_han; }
                    current_block += ch;
                }
                flush_sub_block();
            } else {
                std::u32string buf_u32 = utf8_to_u32(buf);
                for (char32_t ch : buf_u32) result.append(u32_to_utf8(std::u32string(1, ch)));
            }
        }
        buf.clear(); buf_char_count = 0;
    };
    while (x < N) {
        size_t y = route[x].second + 1; size_t start = offsets[x]; size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);
        if (y - x == 1) { buf += word; buf_char_count++; }
        else { process_buffer(); result.append(word); }
        x = y;
    }
    process_buffer();
    return result;
}

py::list _cut_DAG_NO_HMM_cpp(DatTrie& trie, const std::string& sentence, double total) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1; if (N == 0) return py::list();
    struct DAGNode { size_t end_idx; int freq; };
    std::vector<std::vector<DAGNode>> DAG(N);
    std::vector<std::pair<double, size_t>> route(N + 1);
    double logtotal = log(total);
    for (size_t i = 0; i < N; ++i) {
        size_t start = offsets[i]; const char* ptr = sentence.data() + start; size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        if (num > 0) {
            size_t current_res_idx = 0;
            for (size_t j = i; j < N && current_res_idx < num; ++j) {
                size_t prefix_len = offsets[j + 1] - start;
                if (prefix_len == results[current_res_idx].length) {
                    int val = results[current_res_idx].value; int fq = trie.decode_freq(val);
                    if (fq > 0) DAG[i].push_back({j, fq});
                    current_res_idx++;
                }
            }
        }
        if (DAG[i].empty()) DAG[i].push_back({i, 1});
    }
    route[N] = {0.0, 0};
    for (int i = (int)N - 1; i >= 0; i--) {
        double max_prob = -1e100; size_t best_x = i;
        for (const auto& node : DAG[i]) {
            double prob = log((double)node.freq) - logtotal + route[node.end_idx + 1].first;
            if (prob > max_prob) { max_prob = prob; best_x = node.end_idx; }
        }
        route[i] = {max_prob, best_x};
    }
    py::list result; size_t x = 0; std::string buf;
    while (x < N) {
        size_t y = route[x].second + 1; size_t start = offsets[x]; size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);
        if (y - x == 1 && ((word[0] >= 'a' && word[0] <= 'z') || (word[0] >= 'A' && word[0] <= 'Z') || (word[0] >= '0' && word[0] <= '9'))) buf += word;
        else { if (!buf.empty()) { result.append(buf); buf.clear(); } result.append(word); }
        x = y;
    }
    if (!buf.empty()) result.append(buf);
    return result;
}

void _cut_all_block(DatTrie& trie, const std::string& sentence, py::list& result) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1; if (N == 0) return;
    std::vector<std::vector<size_t>> DAG(N);
    for (size_t k = 0; k < N; k++) {
        size_t start = offsets[k]; const char* ptr = sentence.data() + start; size_t remain_len = sentence.size() - start;
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(ptr, results, 64, remain_len);
        if (num > 0) {
            for (size_t i = 0; i < num; ++i) {
                if (results[i].value > 0) {
                    size_t match_len = results[i].length;
                    for (size_t j = k; j < N; ++j) { if (offsets[j+1] - start == match_len) { DAG[k].push_back(j); break; } }
                }
            }
        }
        if (DAG[k].empty()) DAG[k].push_back(k);
    }
    int old_j = -1; int eng_scan = 0; std::string eng_buf = "";
    for (size_t k = 0; k < N; k++) {
        const auto& L = DAG[k]; size_t start_k = offsets[k]; size_t len_k = offsets[k+1] - start_k; std::string char_k = sentence.substr(start_k, len_k);
        bool is_eng = (char_k.size() == 1 && ((char_k[0] >= 'a' && char_k[0] <= 'z') || (char_k[0] >= 'A' && char_k[0] <= 'Z') || (char_k[0] >= '0' && char_k[0] <= '9')));
        if (eng_scan == 1 && !is_eng) { eng_scan = 0; result.append(eng_buf); eng_buf = ""; }
        if (L.size() == 1 && (int)k > old_j) {
            size_t start = offsets[k]; size_t len = offsets[L[0] + 1] - start; std::string word = sentence.substr(start, len);
            if (is_eng && word.size() == len_k) { if (eng_scan == 0) { eng_scan = 1; eng_buf = word; } else eng_buf += word; }
            else { if (eng_scan == 0) result.append(word); }
            old_j = (int)L[0];
        } else {
            for (size_t j : L) { if (j > k) { size_t start = offsets[k]; size_t len = offsets[j + 1] - start; result.append(sentence.substr(start, len)); old_j = (int)j; } }
        }
    }
    if (eng_scan == 1) result.append(eng_buf);
}

py::list _cut_all_internal_cpp(DatTrie& trie, const std::string& sentence) {
    py::list result; const char* p = sentence.data(); const char* end = p + sentence.size(); const char* block_start = nullptr; bool last_is_han_alnum = false;
    while (p < end) {
        const char* current_p = p; char32_t ch = utf8_next_char(p, end); bool is_han_alnum = is_han_alnum_fast(ch);
        if (block_start == nullptr) { block_start = current_p; last_is_han_alnum = is_han_alnum; }
        else if (is_han_alnum != last_is_han_alnum) {
            std::string block(block_start, current_p - block_start);
            if (last_is_han_alnum) _cut_all_block(trie, block, result);
            else {
                const char* sp = block.data(); const char* send = sp + block.size(); const char* s_start = nullptr;
                while (sp < send) {
                    const char* cur_sp = sp; char32_t sch = utf8_next_char(sp, send);
                    bool is_space = (sch == U' ' || sch == U'\t' || sch == U'\r' || sch == U'\n' || sch == 0x3000);
                    if (is_space) { if (s_start) { result.append(std::string(s_start, cur_sp - s_start)); s_start = nullptr; } }
                    else { if (!s_start) s_start = cur_sp; }
                }
                if (s_start) result.append(std::string(s_start, send - s_start));
            }
            block_start = current_p; last_is_han_alnum = is_han_alnum;
        }
    }
    if (block_start) {
        std::string block(block_start, end - block_start);
        if (last_is_han_alnum) _cut_all_block(trie, block, result);
        else {
            const char* sp = block.data(); const char* send = sp + block.size(); const char* s_start = nullptr;
            while (sp < send) {
                const char* cur_sp = sp; char32_t sch = utf8_next_char(sp, send);
                bool is_space = (sch == U' ' || sch == U'\t' || sch == U'\r' || sch == U'\n' || sch == 0x3000);
                if (is_space) { if (s_start) { result.append(std::string(s_start, cur_sp - s_start)); s_start = nullptr; } }
                else { if (!s_start) s_start = cur_sp; }
            }
            if (s_start) result.append(std::string(s_start, send - s_start));
        }
    }
    return result;
}

py::list _cut_internal_cpp(DatTrie& trie, const std::string& sentence, double total, bool HMM) {
    py::list result; const char* p = sentence.data(); const char* end = p + sentence.size(); const char* block_start = nullptr;
    while (p < end) {
        const char* current_p = p; char32_t ch = utf8_next_char(p, end);
        if (is_han_alnum_fast(ch)) { if (!block_start) block_start = current_p; }
        else {
            if (block_start) {
                std::string block(block_start, current_p - block_start);
                py::list words = HMM ? _cut_DAG_cpp(trie, block, total) : _cut_DAG_NO_HMM_cpp(trie, block, total);
                for (auto w : words) result.append(w);
                block_start = nullptr;
            }
            result.append(std::string(current_p, p - current_p));
        }
    }
    if (block_start) {
        std::string block(block_start, end - block_start);
        py::list words = HMM ? _cut_DAG_cpp(trie, block, total) : _cut_DAG_NO_HMM_cpp(trie, block, total);
        for (auto w : words) result.append(w);
    }
    return result;
}

py::list _cut_for_search_internal_cpp(DatTrie& trie, const std::string& sentence, double total, bool HMM) {
    py::list words = _cut_internal_cpp(trie, sentence, total, HMM); py::list result;
    for (auto w_handle : words) {
        std::string w = w_handle.cast<std::string>();
        std::u32string w_u32 = utf8_to_u32(w); size_t w_len = w_u32.length();
        if (w_len > 2) {
            for (size_t i = 0; i < w_len - 1; i++) {
                std::u32string gram2_u32 = w_u32.substr(i, 2); std::string gram2 = u32_to_utf8(gram2_u32);
                if (trie.search(gram2) > 0) result.append(gram2);
            }
        }
        if (w_len > 3) {
            for (size_t i = 0; i < w_len - 2; i++) {
                std::u32string gram3_u32 = w_u32.substr(i, 3); std::string gram3 = u32_to_utf8(gram3_u32);
                if (trie.search(gram3) > 0) result.append(gram3);
            }
        }
        result.append(w);
    }
    return result;
}

py::list _posseg_cut_internal_cpp(DatTrie& trie, const std::string& sentence, double total, bool HMM) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1; py::list result; if (N == 0) return result;
    const char* start_ptr = sentence.data(); std::u32string block_u32; std::string block_utf8;
    auto flush_block = [&]() {
        if (block_utf8.empty()) return;
        py::list words = HMM ? _posseg_cut_DAG_cpp(trie, block_utf8, total) : _posseg_cut_DAG_NO_HMM_cpp(trie, block_utf8, total);
        for (auto w : words) result.append(w);
        block_u32.clear(); block_utf8.clear();
    };
    for (size_t i = 0; i < N; ++i) {
        size_t pos = offsets[i]; size_t next_pos = offsets[i + 1]; const char* p = start_ptr + pos; const char* end = start_ptr + next_pos;
        char32_t ch = utf8_next_char(p, end);
        if (is_han_alnum_fast(ch)) { block_u32 += ch; block_utf8.append(start_ptr + pos, next_pos - pos); }
        else { flush_block(); std::string ch_utf8(start_ptr + pos, next_pos - pos); result.append(Pair(ch_utf8, "x")); }
    }
    flush_block(); return result;
}

} // namespace jieba_fast_dat
