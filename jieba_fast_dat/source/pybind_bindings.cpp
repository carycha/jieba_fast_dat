#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <pybind11/detail/common.h> // For PyCallable_Check

#include <math.h>
#include <stdlib.h>
#include <limits> // For std::numeric_limits
#include <array> // For std::array
#include <string> // For std::string
#include "cedarpp.h"
#include <unordered_map>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <codecvt>
#include <locale>

// HMM model data structures
namespace HMM {
    const double MIN_FLOAT = -3.14e100;
    const double MIN_INF = -std::numeric_limits<double>::infinity();
    // Map from state char ('B', 'M', 'E', 'S') to int (0-3)
    const std::unordered_map<char, int> state_map = {
        {'B', 0}, {'M', 1}, {'E', 2}, {'S', 3}
    };
    // Reverse map from int to state char
    const std::vector<char> reverse_state_map = {'B', 'M', 'E', 'S'};

    // Map from POS tag string to int
    std::unordered_map<std::string, int> pos_tag_map;
    // Reverse map from int to POS tag string
    std::vector<std::string> reverse_pos_tag_map;

    size_t NUM_STATES = 0;

    // Combined ID for (state, pos_tag)
    // id = pos_tag_id * 4 + state_id
    int get_state_tag_id(const std::string& pos_tag, char state) {
        auto it = pos_tag_map.find(pos_tag);
        if (it == pos_tag_map.end()) {
            return -1;
        }
        return it->second * 4 + state_map.at(state);
    }

    // HMM parameters (optimized with flat vectors)
    std::vector<double> start_P;
    std::vector<double> trans_P_flat; // Flattened trans_P
    std::vector<std::unordered_map<char32_t, double>> emit_P;
    std::unordered_map<char32_t, std::vector<int>> char_state_tab_P;
    // For pruning, to replicate original logic
    std::vector<std::vector<int>> trans_P_keys;

    inline double get_trans_P(int from, int to) {
        return trans_P_flat[from * NUM_STATES + to];
    }
}

// Finalseg HMM model data structures
namespace FinalHMM {
    std::vector<double> start_P; // 4 states: B, M, E, S
    std::vector<std::vector<double>> trans_P; // 4x4 matrix
    std::vector<std::unordered_map<char32_t, double>> emit_P; // 4 maps
    bool initialized = false;

    const std::unordered_map<char, int> state_map = {
        {'B', 0}, {'M', 1}, {'E', 2}, {'S', 3}
    };
    const std::vector<char> reverse_state_map = {'B', 'M', 'E', 'S'};
    // PrevStatus: B:ES, M:MB, S:SE, E:BM
    const std::vector<std::vector<int>> prev_states = {
        {2, 3}, // B <- E, S
        {1, 0}, // M <- M, B
        {0, 1}, // E <- B, M
        {3, 2}  // S <- S, E
    };
}


namespace py = pybind11;

class DatTrie {
public:
    DatTrie() {}

    double build(size_t num_keys, const char** keys, const size_t* lengths, const int* freqs) {
        trie_.clear(); // Clear existing trie before building
        double total_freq = 0.0;
        // cedar::da::build expects keys to be sorted. My `all_words` map ensures this.
        trie_.build(num_keys, keys, lengths, freqs);
        for(size_t i = 0; i < num_keys; ++i) {
            total_freq += freqs[i];
        }
        return total_freq;
    }

    double build(py::iterable word_freqs_iterable) {
        trie_.clear(); // Clear existing trie on build.
        double total_freq = 0.0;
        for (py::handle item : word_freqs_iterable) {
            py::tuple pair = item.cast<py::tuple>();
            std::string word = pair[0].cast<std::string>();
            int freq = pair[1].cast<int>();
            trie_.update(word.c_str(), word.length()) = freq;
            total_freq += static_cast<double>(freq);
        }
        return total_freq;
    }

    void clear() {
        trie_.clear();
        word_tag_tab.clear();
    }

    void add_word(const std::string& word, int freq, const std::string& tag = "x") {
        trie_.update(word.c_str(), word.length()) = freq;
        if (!tag.empty()) {
            word_tag_tab[word] = tag;
        }
    }

    void del_word(const std::string& word) {
        trie_.erase(word.c_str(), word.length());
        word_tag_tab.erase(word);
    }

    int search(const std::string& word) const {
        return trie_.exactMatchSearch<int>(word.c_str(), word.length());
    }

    const std::string& get_tag(const std::string& word) const {
        auto it = word_tag_tab.find(word);
        if (it != word_tag_tab.end()) return it->second;
        static const std::string default_tag = "x";
        return default_tag;
    }

    int search(const char* s, size_t len) const {
        return trie_.exactMatchSearch<int>(s, len);
    }

    std::unordered_map<std::string, std::string> word_tag_tab;

    int open(const std::string& filename, size_t offset = 0) {
        return trie_.open(filename.c_str(), "rb", offset);
    }

    int save(const std::string& filename) {
        return trie_.save(filename.c_str());
    }

    size_t num_keys() const {
        return trie_.num_keys();
    }

    py::bytes save_to_bytes() const {
        char* data = nullptr;
        size_t data_size = 0;
        if (trie_.save_to_memory(&data, &data_size) != 0) {
            throw std::runtime_error("Failed to save trie to memory");
        }
        py::bytes result(data, data_size);
        std::free(data);
        return result;
    }

    void load_from_bytes(py::bytes data) {
        std::string_view sv = data;
        if (trie_.open_from_memory(sv.data(), sv.size()) != 0) {
            throw std::runtime_error("Failed to load trie from memory");
        }
    }

    void extract_words(std::vector<std::pair<std::string, int>>& words_with_freqs) {
        size_t count = trie_.num_keys();
        if (count == 0) {
            return;
        }
        words_with_freqs.reserve(count);

        char key_buf[1024]; // Assuming max key length 1023
        cedar::npos_t from = 0; // Represents the node ID
        size_t len_p = 0;       // Represents the length of the current key

        // Iterate through all keys in the trie
        for (int val = trie_.begin(from, len_p);
             val != cedar::da<int>::CEDAR_NO_PATH;
             val = trie_.next(from, len_p)) {

            // Reconstruct the key string using suffix method
            // The 'len_p' argument to suffix should be the length of the key
            trie_.suffix(key_buf, len_p, from);
            words_with_freqs.emplace_back(std::string(key_buf, len_p), val);
        }
    }

    void update_word_tag_tab(py::dict new_tab) {
        for (auto item : new_tab) {
            word_tag_tab[item.first.cast<std::string>()] = item.second.cast<std::string>();
        }
    }

    const cedar::da<int>& trie_ref() const { return trie_; }
    cedar::da<int>& trie_ref() { return trie_; }

private:
    cedar::da<int> trie_;
};

// Helper to get long from py::object
long get_long_from_py_object(py::object obj) {
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<long>();
    }
    throw py::type_error("Expected an integer object.");
}

// Helper to get double from py::object
double get_double_from_py_object(py::object obj) {
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
        return obj.cast<double>();
    }
    throw py::type_error("Expected a float or integer object.");
}

// Helper to safely get an item from a dict, returning a default if not found
py::object get_dict_item_safe(py::dict d, py::object key, py::object default_val = py::none()) {
    if (d.contains(key)) {
        return d[key];
    }
    return default_val;
}


// Helper to get byte offsets for each character in a UTF-8 string
std::vector<size_t> get_utf8_offsets(const std::string& s) {
    std::vector<size_t> offsets;
    offsets.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size(); ) {
        offsets.push_back(i);
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) i += 1;
        else if ((c & 0xE0) == 0xC0) i += 2;
        else if ((c & 0xF0) == 0xE0) i += 3;
        else if ((c & 0xF8) == 0xF0) i += 4;
        else i += 1; // Should not happen in valid UTF-8
    }
    offsets.push_back(s.size());
    return offsets;
}

int _calc_pybind(DatTrie& trie, const std::string& sentence, py::dict DAG, py::dict& route, double total)
{
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    const double logtotal = log(total);
    double max_freq_val, fq_last_val;
    size_t max_x_val, idx, i, t_list_len, x_val;

    route[py::cast(N)] = py::make_tuple(0.0, 0);

    for(int idx_signed = (int)N - 1; idx_signed >= 0 ; idx_signed--)
    {
        idx = (size_t)idx_signed;
        max_freq_val = std::numeric_limits<double>::lowest();
        max_x_val = 0;

        py::object idx_key = py::cast(idx);
        py::list t_list = DAG[idx_key].cast<py::list>();
        t_list_len = py::len(t_list);

        for(i = 0; i < t_list_len; i++)
        {
            x_val = t_list[i].cast<size_t>();

            size_t start = offsets[idx];
            size_t len = offsets[x_val + 1] - start;
            std::string_view word_view(sentence.data() + start, len);

            int fq_val = trie.search(std::string(word_view));
            if (fq_val <= 0) fq_val = 1;

            py::object route_key = py::cast(x_val + 1);
            py::tuple t_tuple = route[route_key].cast<py::tuple>();

            double fq_2_val = t_tuple[0].cast<double>();
            fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if(fq_last_val > max_freq_val)
            {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        route[py::cast(idx)] = py::make_tuple(max_freq_val, max_x_val);
    }
    return 1;
}

int _get_DAG_pybind(py::dict DAG, py::dict FREQ, const std::string& sentence)
{
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;

    for(size_t k = 0; k < N; k++)
    {
        py::list tmplist;
        for(size_t i = k; i < N; i++)
        {
            size_t start = offsets[k];
            size_t len = offsets[i + 1] - start;
            std::string word(sentence.data() + start, len);

            if (FREQ.contains(word))
            {
                py::object freq_item = FREQ[py::cast(word)];
                if (!freq_item.is_none() && freq_item.cast<long>())
                {
                    tmplist.append(i);
                }
            } else {
                break;
            }
        }

        if (py::len(tmplist) == 0) {
            tmplist.append(k);
        }
        DAG[py::cast(k)] = tmplist;
    }
    return 1;
}

int _get_DAG_and_calc_pybind(DatTrie& trie, const std::string& sentence, py::list route, double total)
{
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;

    std::vector<std::vector<size_t>> DAG(N);
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);

    for(size_t k = 0; k < N; k++)
    {
        for(size_t i = k; i < N; i++)
        {
            size_t start = offsets[k];
            size_t len = offsets[i + 1] - start;
            std::string word(sentence.data() + start, len);

            int freq = trie.search(word);
            if (freq == -1) break;
            if (freq > 0) {
                DAG[k].push_back(i);
            }
        }
        if(DAG[k].empty()) {
            DAG[k].push_back(k);
        }
    }

    _route[N][0] = 0.0;
    _route[N][1] = 0.0;

    for(int idx_signed = (int)N - 1; idx_signed >= 0 ; idx_signed--)
    {
        size_t idx = (size_t)idx_signed;
        double max_freq_val = std::numeric_limits<double>::lowest();
        size_t max_x_val = 0;

        for(size_t x_val : DAG[idx])
        {
            size_t start = offsets[idx];
            size_t len = offsets[x_val + 1] - start;
            std::string word(sentence.data() + start, len);

            int fq_val = trie.search(word);
            if (fq_val <= 0) fq_val = 1;

            double fq_2_val = _route[x_val + 1][0];
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if(fq_last_val >= max_freq_val)
            {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = (double)max_x_val;
    }

    for(size_t i = 0; i <= N; i++)
    {
        route.append((long)_route[i][1]);
    }
    return 1;
}

// Define MIN_FLOAT_VAL
const double MIN_FLOAT_VAL = std::numeric_limits<double>::lowest(); // Or a sufficiently small number like -3.14e100

py::tuple _viterbi_pybind(py::sequence obs, py::str _states_py, py::dict start_p, py::dict trans_p, py::dict emip_p)
{
    const py::ssize_t obs_len = py::len(obs);
    const int states_num = 4; // Assuming 'B', 'M', 'S', 'E'

    // Convert Python string to C++ string for easier char access
    std::string states_str = _states_py.cast<std::string>();
    const char* states = states_str.c_str();

    // PrevStatus_str lookup table
    std::array<std::string, 22> PrevStatus_str_cpp;
    PrevStatus_str_cpp['B'-'B'] = "ES";
    PrevStatus_str_cpp['M'-'B'] = "MB";
    PrevStatus_str_cpp['S'-'B'] = "SE";
    PrevStatus_str_cpp['E'-'B'] = "BM";


    // Dynamic 2D arrays V and path
    std::vector<std::array<double, 22>> V(obs_len);
    std::vector<std::array<char, 22>> path(obs_len);

    // py_states: array of py::str objects for state characters
    std::array<py::str, 4> py_states_cpp;
    for(int i=0; i<states_num; ++i) {
        py_states_cpp[i] = py::str(std::string(1, states[i])); // Corrected
    }

    // emip_p_dict: array of py::dict objects
    std::array<py::dict, 4> emip_p_dict_cpp;
    for(int i=0; i<states_num; ++i) {
        emip_p_dict_cpp[i] = emip_p[py_states_cpp[i]].cast<py::dict>();
    }

    // trans_p_dict: 2D array of py::object (can be dict or None)
    // The original C code uses PyDict_GetItem which can return NULL.
    // We'll use dict_get_item and check for None.
    std::array<std::array<py::object, 2>, 22> trans_p_dict_cpp_obj; // Store py::object

    trans_p_dict_cpp_obj['B'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[2]); // 'S'
    trans_p_dict_cpp_obj['B'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[3]); // 'E'
    trans_p_dict_cpp_obj['M'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[1]); // 'M'
    trans_p_dict_cpp_obj['M'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[0]); // 'B'
    trans_p_dict_cpp_obj['E'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[0]); // 'B'
    trans_p_dict_cpp_obj['E'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[1]); // 'M'
    trans_p_dict_cpp_obj['S'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[3]); // 'E'
    trans_p_dict_cpp_obj['S'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[2]); // 'S'


    // Initialization for V[0] and path[0]
    for(int i=0; i<states_num; ++i)
    {
        py::dict t_dict = emip_p_dict_cpp[i]; // Already cast to dict
        double t_double_val = MIN_FLOAT_VAL;
        py::object ttemp_obj = obs[0]; // obs[0]
        py::object item_obj = get_dict_item_safe(t_dict, ttemp_obj); // Corrected

        if(!item_obj.is_none())
            t_double_val = get_double_from_py_object(item_obj);

        py::object start_p_item_obj = get_dict_item_safe(start_p, py_states_cpp[i]); // Corrected
        double t_double_2_val = MIN_FLOAT_VAL; // Default if not found
        if (!start_p_item_obj.is_none()) {
            t_double_2_val = get_double_from_py_object(start_p_item_obj);
        }

        V[0][states[i]-'B'] = t_double_val + t_double_2_val;
        path[0][states[i]-'B'] = states[i];
    }

    // Main Viterbi loop
    for(py::ssize_t i=1; i<obs_len; ++i)
    {
        py::object t_obs_obj = obs[i]; // obs[i]
        for(int j=0; j<states_num; ++j)
        {
            double em_p_val = MIN_FLOAT_VAL;
            char y_char = states[j];
            py::object item_obj = get_dict_item_safe(emip_p_dict_cpp[j], t_obs_obj); // Corrected
            if(!item_obj.is_none())
                em_p_val = get_double_from_py_object(item_obj);

            double max_prob_val = MIN_FLOAT_VAL;
            char best_state_char = '\0';

            for(int p = 0; p < 2; ++p)
            {
                double prob_val = em_p_val;
                char y0_char = PrevStatus_str_cpp[y_char-'B'][p];
                prob_val += V[i - 1][y0_char-'B'];

                py::object trans_p_item_obj = get_dict_item_safe(trans_p_dict_cpp_obj[y_char-'B'][p], py_states_cpp[j]); // Corrected
                if (trans_p_item_obj.is_none())
                    prob_val += MIN_FLOAT_VAL;
                else
                    prob_val += get_double_from_py_object(trans_p_item_obj);

                if (prob_val > max_prob_val)
                {
                    max_prob_val = prob_val;
                    best_state_char = y0_char;
                }
            }
            // Original C code had a fallback if best_state was still '\0'
            // This part seems to ensure best_state is set even if all probs are MIN_FLOAT
            if(best_state_char == '\0')
            {
                for(int p = 0; p < 2; p++)
                {
                    char y0_char_fallback = PrevStatus_str_cpp[y_char-'B'][p];
                    if(y0_char_fallback > best_state_char) // This comparison is character-based
                        best_state_char = y0_char_fallback;
                }
            }
            V[i][y_char-'B'] = max_prob_val;
            path[i][y_char-'B'] = best_state_char;
        }
    }

    // Final path reconstruction
    double max_prob_final = V[obs_len-1]['E'-'B'];
    char best_state_final = 'E';

    if (V[obs_len-1]['S'-'B'] > max_prob_final)
    {
        max_prob_final = V[obs_len-1]['S'-'B'];
        best_state_final = 'S';
    }

    py::list t_list_final; // Resulting list of states
    char now_state_char = best_state_final;

    for(py::ssize_t i = obs_len - 1; i >= 0; --i)
    {
        t_list_final.insert(0, py::str(std::string(1, now_state_char))); // Corrected py::str constructor
        now_state_char = path[i][now_state_char-'B'];
    }

    // Return a tuple (max_prob, list_of_states)
    return py::make_tuple(max_prob_final, t_list_final);
}

int _get_trie_pybind(DatTrie& trie, const std::string& filename, size_t offset = 0) {
    return trie.open(filename, offset);
}


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
        return word < other.word;
    }

    bool operator==(const Pair& other) const {
        return word == other.word && flag == other.flag;
    }
};

// Helper struct for Viterbi result
struct ViterbiResult {
    double prob;
    std::vector<Pair> word_tags;
};

// Helper to convert u32string to UTF-8 string
std::string u32_to_utf8(const std::u32string& s) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.to_bytes(s);
}

ViterbiResult posseg_viterbi_impl(const std::u32string& obs) {
    size_t obs_len = obs.length();
    if (obs_len == 0) {
        return {0.0, {}};
    }

    size_t num_states = HMM::NUM_STATES;
    // V[t * num_states + state] = prob
    std::vector<double> V(obs_len * num_states, HMM::MIN_INF);
    std::vector<int> mem_path(obs_len * num_states, -1);

    // Initialization
    char32_t first_char = obs[0];
    const std::vector<int>* initial_states;
    std::vector<int> all_states_vec;
    if (HMM::char_state_tab_P.count(first_char)) {
        initial_states = &HMM::char_state_tab_P.at(first_char);
    } else {
        all_states_vec.reserve(num_states);
        for(size_t i=0; i< num_states; ++i) all_states_vec.push_back(static_cast<int>(i));
        initial_states = &all_states_vec;
    }

    for (int y : *initial_states) {
        double emit = HMM::MIN_FLOAT;
        if (static_cast<size_t>(y) < HMM::emit_P.size() && HMM::emit_P[y].count(first_char)) {
             emit = HMM::emit_P[y].at(first_char);
        }
        V[y] = HMM::start_P[y] + emit;
    }

    // Reuse vectors to avoid re-allocation
    std::vector<int> prev_states;
    std::vector<int> obs_states;
    std::vector<bool> states_mask(num_states, false);
    prev_states.reserve(num_states);
    obs_states.reserve(num_states);

    // Recursion
    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];
        prev_states.clear();
        for(size_t i = 0; i < num_states; ++i) {
            if (V[(t-1) * num_states + i] > HMM::MIN_INF) {
                prev_states.push_back(static_cast<int>(i));
            }
        }

        if (prev_states.empty()) break;

        // Determine candidate states for current char
        obs_states.clear();
        std::fill(states_mask.begin(), states_mask.end(), false);

        if (HMM::char_state_tab_P.count(current_char)) {
            const std::vector<int>& char_states = HMM::char_state_tab_P.at(current_char);
            for (int y : char_states) {
                states_mask[y] = true;
            }
            // Only keep states that can be reached from prev_states
            for (int x : prev_states) {
                for (int y_next : HMM::trans_P_keys[x]) {
                    if (states_mask[y_next]) {
                        obs_states.push_back(y_next);
                        states_mask[y_next] = false; // Avoid duplicates
                    }
                }
            }
        } else {
            for (int x : prev_states) {
                for (int y_next : HMM::trans_P_keys[x]) {
                    if (!states_mask[y_next]) {
                        obs_states.push_back(y_next);
                        states_mask[y_next] = true;
                    }
                }
            }
        }

        if (obs_states.empty()) {
            for (int x : prev_states) {
                for (int y_next : HMM::trans_P_keys[x]) {
                    if (!states_mask[y_next]) {
                        obs_states.push_back(y_next);
                        states_mask[y_next] = true;
                    }
                }
            }
        }

        for (int y : obs_states) {
            double max_prob = HMM::MIN_INF;
            int best_prev_state = -1;

            double em_p = HMM::MIN_FLOAT;
            if (static_cast<size_t>(y) < HMM::emit_P.size() && HMM::emit_P[y].count(current_char)) {
                em_p = HMM::emit_P[y].at(current_char);
            }

            for (int y0 : prev_states) {
                double trans = HMM::get_trans_P(y0, y);
                if (trans == HMM::MIN_INF) continue;

                double current_prob = V[(t - 1) * num_states + y0] + trans;
                if (current_prob > max_prob) {
                    max_prob = current_prob;
                    best_prev_state = y0;
                }
            }
            V[t * num_states + y] = max_prob + em_p;
            mem_path[t * num_states + y] = best_prev_state;
        }
    }

    // Termination
    double final_max_prob = HMM::MIN_INF;
    int last_state = -1;
    size_t last_idx_base = (obs_len - 1) * num_states;

    for (size_t y = 0; y < num_states; ++y) {
        if (V[last_idx_base + y] > final_max_prob) {
            final_max_prob = V[last_idx_base + y];
            last_state = static_cast<int>(y);
        }
    }

    if (last_state == -1) {
        return {0.0, {}};
    }

    // Path backtracking
    std::vector<int> path_ids;
    path_ids.reserve(obs_len);
    int curr = last_state;
    for (int t = static_cast<int>(obs_len) - 1; t >= 0; --t) {
        path_ids.push_back(curr);
        curr = mem_path[t * num_states + curr];
    }
    std::reverse(path_ids.begin(), path_ids.end());

    // Word reconstruction
    std::vector<Pair> word_pos_tags_route;
    size_t begin = 0;
    for (size_t i = 0; i < obs_len; ++i) {
        int state_id = path_ids[i];
        int pos_tag_id = state_id / 4;
        char state_char = HMM::reverse_state_map[state_id % 4];
        std::string pos_tag = HMM::reverse_pos_tag_map[pos_tag_id];

        if (state_char == 'B') {
            begin = i;
        } else if (state_char == 'E') {
            word_pos_tags_route.emplace_back(u32_to_utf8(obs.substr(begin, i + 1 - begin)), pos_tag);
        } else if (state_char == 'S') {
            word_pos_tags_route.emplace_back(u32_to_utf8(obs.substr(i, 1)), pos_tag);
        }
    }

    return {final_max_prob, word_pos_tags_route};
}

py::tuple _posseg_viterbi_cpp(std::u32string obs) {
    ViterbiResult result = posseg_viterbi_impl(obs);

    py::list word_pos_tags_route;
    for (auto& item : result.word_tags) {
        word_pos_tags_route.append(std::move(item));
    }

    return py::make_tuple(result.prob, word_pos_tags_route);
}

void load_finalseg_hmm_model(py::dict start_p_dict, py::dict trans_p_dict, py::dict emit_p_dict) {
    FinalHMM::start_P.assign(4, HMM::MIN_FLOAT);
    FinalHMM::trans_P.assign(4, std::vector<double>(4, HMM::MIN_FLOAT));
    FinalHMM::emit_P.assign(4, std::unordered_map<char32_t, double>());

    for (auto item : start_p_dict) {
        std::string state_str = item.first.cast<std::string>();
        double prob = item.second.cast<double>();
        if (FinalHMM::state_map.count(state_str[0])) {
            FinalHMM::start_P[FinalHMM::state_map.at(state_str[0])] = prob;
        }
    }

    for (auto from_item : trans_p_dict) {
        char from_state = from_item.first.cast<std::string>()[0];
        int from_id = FinalHMM::state_map.at(from_state);
        py::dict to_dict = from_item.second.cast<py::dict>();
        for (auto to_item : to_dict) {
            char to_state = to_item.first.cast<std::string>()[0];
            int to_id = FinalHMM::state_map.at(to_state);
            FinalHMM::trans_P[from_id][to_id] = to_item.second.cast<double>();
        }
    }

    for (auto item : emit_p_dict) {
        char state = item.first.cast<std::string>()[0];
        int id = FinalHMM::state_map.at(state);
        py::dict char_prob_dict = item.second.cast<py::dict>();
        for (auto char_item : char_prob_dict) {
            std::u32string ch_str = char_item.first.cast<std::u32string>();
            if (!ch_str.empty()) {
                FinalHMM::emit_P[id][ch_str[0]] = char_item.second.cast<double>();
            }
        }
    }
    FinalHMM::initialized = true;
}

// Internal finalseg Viterbi implementation returning a vector of strings
std::vector<std::string> finalseg_viterbi_internal(const std::u32string& obs) {
    size_t obs_len = obs.length();
    if (obs_len == 0) return {};

    if (!FinalHMM::initialized) {
        // Fallback: return single characters if models are not loaded
        std::vector<std::string> words;
        words.reserve(obs_len);
        for (char32_t ch : obs) {
            words.push_back(u32_to_utf8(std::u32string(1, ch)));
        }
        return words;
    }

    std::vector<std::array<double, 4>> V(obs_len);
    std::vector<std::array<int, 4>> path(obs_len);

    for (int i = 0; i < 4; ++i) {
        double emit = HMM::MIN_FLOAT;
        if (FinalHMM::emit_P[i].count(obs[0])) {
            emit = FinalHMM::emit_P[i].at(obs[0]);
        }
        V[0][i] = FinalHMM::start_P[i] + emit;
        path[0][i] = i;
    }

    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];
        for (int y = 0; y < 4; ++y) {
            double em_p = HMM::MIN_FLOAT;
            if (FinalHMM::emit_P[y].count(current_char)) {
                em_p = FinalHMM::emit_P[y].at(current_char);
            }

            double max_prob = HMM::MIN_INF;
            int best_prev = -1;

            for (int y0 : FinalHMM::prev_states[y]) {
                double prob = V[t - 1][y0] + FinalHMM::trans_P[y0][y] + em_p;
                if (prob > max_prob) {
                    max_prob = prob;
                    best_prev = y0;
                }
            }
            V[t][y] = max_prob;
            path[t][y] = best_prev;
        }
    }

    double max_prob_final = V[obs_len - 1][2]; // 'E'
    int best_state_final = 2;
    if (V[obs_len - 1][3] > max_prob_final) { // 'S'
        max_prob_final = V[obs_len - 1][3];
        best_state_final = 3;
    }

    std::string res_states = "";
    int curr = best_state_final;
    for (int t = static_cast<int>(obs_len) - 1; t >= 0; --t) {
        res_states += FinalHMM::reverse_state_map[curr];
        curr = path[t][curr];
    }
    std::reverse(res_states.begin(), res_states.end());

    std::vector<std::string> words;
    size_t begin = 0;
    for (size_t i = 0; i < obs_len; ++i) {
        char pos = res_states[i];
        if (pos == 'B') {
            begin = i;
        } else if (pos == 'E') {
            words.push_back(u32_to_utf8(obs.substr(begin, i + 1 - begin)));
        } else if (pos == 'S') {
            words.push_back(u32_to_utf8(obs.substr(i, 1)));
        }
    }
    return words;
}

py::tuple _finalseg_viterbi_cpp(std::u32string obs) {
    size_t obs_len = obs.length();
    if (obs_len == 0 || !FinalHMM::initialized) {
        return py::make_tuple(0.0, py::list());
    }

    // Re-calculate for probability if needed for public API, or just return 0.0
    // To be efficient and accurate, we re-run the Viterbi core but return (prob, path_str)
    std::vector<std::array<double, 4>> V(obs_len);
    std::vector<std::array<int, 4>> path(obs_len);

    for (int i = 0; i < 4; ++i) {
        double emit = HMM::MIN_FLOAT;
        if (FinalHMM::emit_P[i].count(obs[0])) {
            emit = FinalHMM::emit_P[i].at(obs[0]);
        }
        V[0][i] = FinalHMM::start_P[i] + emit;
        path[0][i] = i;
    }

    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];
        for (int y = 0; y < 4; ++y) {
            double em_p = HMM::MIN_FLOAT;
            if (FinalHMM::emit_P[y].count(current_char)) {
                em_p = FinalHMM::emit_P[y].at(current_char);
            }
            double max_prob = HMM::MIN_INF;
            int best_prev = -1;
            for (int y0 : FinalHMM::prev_states[y]) {
                double prob = V[t - 1][y0] + FinalHMM::trans_P[y0][y] + em_p;
                if (prob > max_prob) {
                    max_prob = prob;
                    best_prev = y0;
                }
            }
            V[t][y] = max_prob;
            path[t][y] = best_prev;
        }
    }

    double max_prob_final = V[obs_len - 1][2]; // 'E'
    int best_state_final = 2;
    if (V[obs_len - 1][3] > max_prob_final) { // 'S'
        max_prob_final = V[obs_len - 1][3];
        best_state_final = 3;
    }

    std::string res_states = "";
    int curr = best_state_final;
    for (int t = static_cast<int>(obs_len) - 1; t >= 0; --t) {
        res_states += FinalHMM::reverse_state_map[curr];
        curr = path[t][curr];
    }
    std::reverse(res_states.begin(), res_states.end());

    return py::make_tuple(max_prob_final, py::cast(res_states));
}


double load_userdict_from_path_pybind(DatTrie& trie, const std::string& filename, py::dict& user_word_tag_tab_py, py::object batch_add_force_split_func) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open dictionary file: " + filename);
    }
    // Collect all words and their frequencies, including those currently in the trie and from user dict
    // Use std::map to ensure keys are sorted, required for cedar::da::build
    std::map<std::string, int> all_words;
    std::unordered_map<std::string, std::string> tags_from_user_dict_file; // Collect tags explicitly from the user dict file
    std::vector<std::string> force_split_words_to_add; // Collect words for batch force_split

    // Phase 1: Extract all existing words from the trie
    std::vector<std::pair<std::string, int>> existing_trie_words;
    trie.extract_words(existing_trie_words);
    for (const auto& pair : existing_trie_words) {
        all_words[pair.first] = pair.second;
    }

    // Phase 2: Process user dictionary lines from file
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        // Handle BOM on first line if present
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }

        // Trim whitespace and skip comments
        size_t first = line.find_first_not_of(" \t\r\n");
        size_t last = line.find_last_not_of(" \t\r\n");
        if (std::string::npos == first || std::string::npos == last) {
            continue; // Empty or all-whitespace line
        }
        line = line.substr(first, (last - first + 1));
        if (line.empty() || line[0] == '#') continue;

        std::string word_str;
        int freq = 3; // Default frequency based on jieba's default for add_word
        std::string tag = "x"; // Default tag: 'x' for unknown/other

        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) {
            parts.push_back(part);
        }

        if (parts.empty()) continue;

        word_str = parts[0];
        if (parts.size() > 1) {
            bool is_digit = !parts[1].empty() && std::all_of(parts[1].begin(), parts[1].end(), ::isdigit);
            if (is_digit) {
                freq = std::stoi(parts[1]);
            } else {
                tag = parts[1];
            }
        }
        if (parts.size() > 2) {
            tag = parts[2];
        }

        all_words[word_str] = freq;
        tags_from_user_dict_file[word_str] = tag;

        size_t len_word = word_str.length();
        const char* str_ptr = word_str.c_str();
        for (size_t i = 0; i < len_word; ) {
            size_t char_len = 1;
            unsigned char c = static_cast<unsigned char>(str_ptr[i]);
            if (c < 0x80) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            else { char_len = 1; }

            if (i + char_len > len_word) break;

            std::string wfrag = word_str.substr(0, i + char_len);

            if (wfrag.length() < word_str.length()) {
                all_words.insert({wfrag, 0});
            }
            i += char_len;
        }

        if (freq == 0) {
            force_split_words_to_add.push_back(word_str);
        }
    }

    file.close();

    // Phase 3: Prepare data for rebuilding
    std::vector<const char*> keys;
    std::vector<size_t> lengths;
    std::vector<int> freqs;
    keys.reserve(all_words.size());
    lengths.reserve(all_words.size());
    freqs.reserve(all_words.size());

    double new_total_freq = 0.0;
    for (const auto& pair : all_words) {
        keys.push_back(pair.first.c_str());
        lengths.push_back(pair.first.length());
        freqs.push_back(pair.second);
        new_total_freq += pair.second;
    }

    // Phase 4: Rebuild DatTrie
    trie.build(keys.size(), keys.data(), lengths.data(), freqs.data());

    // Phase 5: Update Python and C++ word_tag_tab
    user_word_tag_tab_py.clear();
    for (const auto& pair : tags_from_user_dict_file) {
        py::str py_word = py::str(pair.first);
        py::str py_tag = py::str(pair.second);
        user_word_tag_tab_py[py_word] = py_tag;
        trie.word_tag_tab[pair.first] = pair.second;
    }

    // Phase 6: Batch call batch_add_force_split_func
    if (!force_split_words_to_add.empty() && batch_add_force_split_func.ptr() != nullptr && PyCallable_Check(batch_add_force_split_func.ptr())) {
        py::list py_force_split_words = py::cast(force_split_words_to_add);
        batch_add_force_split_func(py_force_split_words);
    }

    return new_total_freq;
}



double load_main_dict_from_path_pybind(DatTrie& trie, const std::string& filename, py::dict& main_word_tag_tab_py) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open dictionary file: " + filename);
    }
    std::map<std::string, int> all_words;
    std::unordered_map<std::string, std::string> tags_from_main_dict_file;

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
        if (std::string::npos == first || std::string::npos == last) {
            continue;
        }
        line = line.substr(first, (last - first + 1));
        if (line.empty() || line[0] == '#') continue;

        std::string word_str;
        int freq = 3;
        std::string tag = "x";

        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) {
            parts.push_back(part);
        }

        if (parts.empty()) continue;

        word_str = parts[0];
        if (parts.size() > 1) {
            bool is_digit = !parts[1].empty() && std::all_of(parts[1].begin(), parts[1].end(), ::isdigit);
            if (is_digit) {
                freq = std::stoi(parts[1]);
            } else {
                tag = parts[1];
            }
        }
        if (parts.size() > 2) {
            tag = parts[2];
        }

        all_words[word_str] = freq;
        tags_from_main_dict_file[word_str] = tag;

        size_t len_word = word_str.length();
        const char* str_ptr = word_str.c_str();
        for (size_t i = 0; i < len_word; ) {
            size_t char_len = 1;
            unsigned char c = static_cast<unsigned char>(str_ptr[i]);
            if (c < 0x80) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            else { char_len = 1; }

            if (i + char_len > len_word) break;

            std::string wfrag = word_str.substr(0, i + char_len);

            if (wfrag.length() < word_str.length()) {
                all_words.insert({wfrag, 0});
            }
            i += char_len;
        }
    }

    file.close();

    std::vector<const char*> keys;
    std::vector<size_t> lengths;
    std::vector<int> freqs;
    keys.reserve(all_words.size());
    lengths.reserve(all_words.size());
    freqs.reserve(all_words.size());

    double new_total_freq = 0.0;
    for (const auto& pair : all_words) {
        keys.push_back(pair.first.c_str());
        lengths.push_back(pair.first.length());
        freqs.push_back(pair.second);
        new_total_freq += pair.second;
    }

    trie.build(keys.size(), keys.data(), lengths.data(), freqs.data());

    main_word_tag_tab_py.clear();
    for (const auto& pair : tags_from_main_dict_file) {
        py::str py_word = py::str(pair.first);
        py::str py_tag = py::str(pair.second);
        main_word_tag_tab_py[py_word] = py_tag;
        trie.word_tag_tab[pair.first] = pair.second;
    }

    return new_total_freq;
}

void load_hmm_model(py::dict start_p_dict, py::dict trans_p_dict, py::dict emit_p_dict, py::dict char_state_tab_p_dict) {
    // Clear previous data
    HMM::pos_tag_map.clear();
    HMM::reverse_pos_tag_map.clear();
    HMM::start_P.clear();
    HMM::trans_P_flat.clear();
    HMM::emit_P.clear();
    HMM::char_state_tab_P.clear();
    HMM::trans_P_keys.clear();

    // Build pos_tag maps from start_p keys
    int tag_id_counter = 0;
    for (auto item : start_p_dict) { // Iterate over dict items directly
        py::tuple state_tag = item.first.cast<py::tuple>(); // key part: (char, str) tuple
        std::string tag = state_tag[1].cast<std::string>();
        if (HMM::pos_tag_map.find(tag) == HMM::pos_tag_map.end()) {
            HMM::pos_tag_map[tag] = tag_id_counter;
            HMM::reverse_pos_tag_map.push_back(tag);
            tag_id_counter++;
        }
    }

    HMM::NUM_STATES = HMM::pos_tag_map.size() * 4;
    HMM::start_P.assign(HMM::NUM_STATES, HMM::MIN_FLOAT);
    HMM::trans_P_flat.assign(HMM::NUM_STATES * HMM::NUM_STATES, HMM::MIN_INF);
    HMM::emit_P.assign(HMM::NUM_STATES, std::unordered_map<char32_t, double>());
    HMM::trans_P_keys.assign(HMM::NUM_STATES, std::vector<int>());

    // Populate start_P
    for (auto item : start_p_dict) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        char state = state_tag[0].cast<std::string>()[0];
        std::string tag = state_tag[1].cast<std::string>();
        double prob = item.second.cast<double>(); // Access value directly
        int id = HMM::get_state_tag_id(tag, state);
        if (id != -1) {
            HMM::start_P[id] = prob;
        }
    }

    // Populate trans_P and trans_P_keys
    for (auto from_item : trans_p_dict) {
        py::tuple from_state_tag = from_item.first.cast<py::tuple>();
        char from_state = from_state_tag[0].cast<std::string>()[0];
        std::string from_tag = from_state_tag[1].cast<std::string>();
        int from_id = HMM::get_state_tag_id(from_tag, from_state);
        if (from_id == -1) continue;

        py::dict to_dict = from_item.second.cast<py::dict>(); // Inner dict
        for (auto to_item : to_dict) {
            py::tuple to_state_tag = to_item.first.cast<py::tuple>();
            char to_state = to_state_tag[0].cast<std::string>()[0];
            std::string to_tag = to_state_tag[1].cast<std::string>();
            double prob = to_item.second.cast<double>();
            int to_id = HMM::get_state_tag_id(to_tag, to_state);
            if (to_id != -1) {
                HMM::trans_P_flat[from_id * HMM::NUM_STATES + to_id] = prob;
                HMM::trans_P_keys[from_id].push_back(to_id);
            }
        }
    }

    // Populate emit_P
    for (auto item : emit_p_dict) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        char state = state_tag[0].cast<std::string>()[0];
        std::string tag = state_tag[1].cast<std::string>();
        int id = HMM::get_state_tag_id(tag, state);
        if (id == -1) continue;

        py::dict char_prob_dict = item.second.cast<py::dict>(); // Inner dict
        for (auto char_item : char_prob_dict) {
            std::u32string ch_str = char_item.first.cast<std::u32string>();
            if (!ch_str.empty()) {
                 char32_t ch = ch_str[0];
                 double prob = char_item.second.cast<double>();
                 HMM::emit_P[id][ch] = prob;
            }
        }
    }

    // Populate char_state_tab_P
    for (auto item : char_state_tab_p_dict) {
        std::u32string ch_str = item.first.cast<std::u32string>();
        if (!ch_str.empty()) {
            char32_t ch = ch_str[0];
            py::list state_tag_list = item.second.cast<py::list>(); // Value is a Python list
            std::vector<int> state_ids;
            for(py::handle state_tag_item_handle : state_tag_list) { // Iterate over list items
                py::tuple state_tag = state_tag_item_handle.cast<py::tuple>();
                char state = state_tag[0].cast<std::string>()[0];
                std::string tag = state_tag[1].cast<std::string>();
                int id = HMM::get_state_tag_id(tag, state);
                if (id != -1) {
                    state_ids.push_back(id);
                }
            }
            HMM::char_state_tab_P[ch] = state_ids;
        }
    }
}

py::dict _get_DAG(DatTrie& trie, const std::string& sentence) {
    py::dict DAG;
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;

    for (size_t k = 0; k < N; k++) {
        py::list tmplist;
        for (size_t i = k; i < N; i++) {
            size_t start = offsets[k];
            size_t len = offsets[i + 1] - start;
            std::string word(sentence.data() + start, len);

            int freq = trie.search(word);
            if (freq == -1) break;
            if (freq > 0) {
                tmplist.append(i);
            }
        }
        if (py::len(tmplist) == 0) {
            tmplist.append(k);
        }
        DAG[py::cast(k)] = tmplist;
    }
    return DAG;
}

int _get_freq(DatTrie& trie, py::object word) {
    std::string word_str = word.cast<std::string>();
    int freq = trie.search(word_str);
    if (freq != -1) {
        return freq;
    }
    return 0;
}

// Helper function to convert std::u32string to py::str
py::str u32string_to_pystr(const std::u32string& s) {
    return py::cast(s);
}

// Helper function to check if a u32string matches a number pattern
bool is_number(const std::u32string& s) {
    if (s.empty()) return false;
    for (char32_t ch : s) {
        if (!((ch >= U'0' && ch <= U'9') || ch == U'.')) {
            return false;
        }
    }
    return true;
}

// Helper function to check if a u32string matches an english pattern (must have at least one letter)
bool is_english(const std::u32string& s) {
    if (s.empty()) return false;
    bool has_alpha = false;
    for (char32_t ch : s) {
        if ((ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z')) {
            has_alpha = true;
        } else if (ch >= U'0' && ch <= U'9') {
            // digit allowed
        } else {
            return false;
        }
    }
    return has_alpha;
}

// Optimized posseg __cut_DAG
py::list _posseg_cut_DAG_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total
) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return py::list();

    std::vector<std::vector<size_t>> DAG(N);
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);

    for(size_t k = 0; k < N; k++) {
        for(size_t i = k; i < N; i++) {
            size_t start = offsets[k];
            size_t len = offsets[i + 1] - start;
            std::string word(sentence.data() + start, len);
            int freq = trie.search(word);
            if (freq == -1) break;
            if (freq > 0) DAG[k].push_back(i);
        }
        if(DAG[k].empty()) DAG[k].push_back(k);
    }

    _route[N][0] = 0.0;
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
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + _route[x_val + 1][0];
            if(fq_last_val >= max_freq_val) {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = (double)max_x_val;
    }

    py::list result;
    size_t x = 0;
    std::string buf;

    auto process_buffer = [&](const std::string& buffer) {
        if (buffer.empty()) return;

        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        std::u32string buf_u32 = conv.from_bytes(buffer);

        if (buf_u32.length() == 1) {
            result.append(Pair(buffer, trie.get_tag(buffer)));
            return;
        }

        // Mixed buffer: split into blocks of Han, Alphanumeric, and Others
        std::u32string current_block;
        enum CharType { HAN, ALPHANUM, OTHER };
        CharType last_type = OTHER;

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
                for (auto& word_tag : viterbi_result.word_tags) {
                    result.append(std::move(word_tag));
                }
            } else if (last_type == ALPHANUM) {
                if (is_english(current_block)) result.append(Pair(block_utf8, "eng"));
                else result.append(Pair(block_utf8, "m"));
            } else {
                result.append(Pair(block_utf8, "x"));
            }
            current_block.clear();
        };

        for (char32_t ch : buf_u32) {
            CharType current_type = get_type(ch);
            if (current_block.empty()) {
                last_type = current_type;
            } else if (current_type != last_type) {
                flush_block();
                last_type = current_type;
            }
            current_block += ch;
        }
        flush_block();
    };

    while (x < N) {
        size_t y = static_cast<size_t>(_route[x][1]) + 1;
        size_t start = offsets[x];
        size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);

        if (y - x == 1) {
            buf += word;
        } else {
            if (!buf.empty()) { process_buffer(buf); buf.clear(); }
            result.append(Pair(word, trie.get_tag(word)));
        }
        x = y;
    }
    if (!buf.empty()) process_buffer(buf);
    return result;
}

// Optimized posseg __cut_DAG_NO_HMM
py::list _posseg_cut_DAG_NO_HMM_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total
) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return py::list();

    std::vector<std::vector<size_t>> DAG(N);
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);

    for(size_t k = 0; k < N; k++) {
        for(size_t i = k; i < N; i++) {
            size_t start = offsets[k];
            size_t len = offsets[i + 1] - start;
            std::string word(sentence.data() + start, len);
            int freq = trie.search(word);
            if (freq == -1) break;
            if (freq > 0) DAG[k].push_back(i);
        }
        if(DAG[k].empty()) DAG[k].push_back(k);
    }

    _route[N][0] = 0.0;
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
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + _route[x_val + 1][0];
            if(fq_last_val >= max_freq_val) {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = (double)max_x_val;
    }

    py::list result;
    size_t x = 0;
    std::string buf;

    while (x < N) {
        size_t y = static_cast<size_t>(_route[x][1]) + 1;
        size_t start = offsets[x];
        size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);

        if (y - x == 1) {
            unsigned char c = static_cast<unsigned char>(word[0]);
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
                buf += word;
            } else {
                if (!buf.empty()) { result.append(Pair(buf, "eng")); buf.clear(); }
                result.append(Pair(word, trie.get_tag(word)));
            }
        } else {
            if (!buf.empty()) { result.append(Pair(buf, "eng")); buf.clear(); }
            result.append(Pair(word, trie.get_tag(word)));
        }
        x = y;
    }



                    if (!buf.empty()) result.append(Pair(buf, "eng"));



                    return result;



                }










// C++ implementation for _load_word_tag_pybind
void _load_word_tag_pybind(const std::string& filename, py::dict word_tag_tab_py) {
    word_tag_tab_py.clear(); // Clear existing content before populating

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw py::value_error("Could not open dictionary file: " + filename);
    }

    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        // Handle BOM on first line if present (Python should typically handle this, but for robustness)
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }

        // Trim whitespace
        size_t first = line.find_first_not_of(" \t\r\n");
        size_t last = line.find_last_not_of(" \t\r\n");
        if (std::string::npos == first || std::string::npos == last) {
            continue; // Empty or all-whitespace line
        }
        line = line.substr(first, (last - first + 1));

        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments

        std::string word_str;
        std::string tag_str = "x"; // Default tag

        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) {
            parts.push_back(part);
        }

        if (parts.empty()) continue;

        word_str = parts[0];
        if (parts.size() > 1) {
            // Check if the second part is a number (frequency)
            // If it's not a number, it's a tag
            bool is_digit = !parts[1].empty() && std::all_of(parts[1].begin(), parts[1].end(), ::isdigit);
            if (!is_digit) {
                tag_str = parts[1];
            }
        }
        if (parts.size() > 2) {
            tag_str = parts[2];
        }

        word_tag_tab_py[py::str(word_str)] = py::str(tag_str);
    }
    file.close();
}


// Optimized C++ implementation of __cut_DAG
py::list _cut_DAG_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total
) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return py::list();

    struct DAGNode {
        size_t end_idx;
        int freq;
    };
    std::vector<std::vector<DAGNode>> DAG(N);
    std::vector<std::pair<double, size_t>> route(N + 1);
    double logtotal = log(total);

    // One pass to build DAG with frequencies
    for (size_t k = 0; k < N; ++k) {
        size_t start = offsets[k];
        const char* ptr = sentence.data() + start;
        size_t remain_len = sentence.size() - start;

        // cedar's commonPrefixSearch
        // We use a fixed-size buffer for results to avoid heap allocation
        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(
            ptr, results, 64, remain_len
        );

        // Find which character index each prefix length corresponds to
        size_t current_res_idx = 0;
        for (size_t i = k; i < N && current_res_idx < num; ++i) {
            size_t prefix_len = offsets[i + 1] - start;
            if (prefix_len == results[current_res_idx].length) {
                if (results[current_res_idx].value > 0) {
                    DAG[k].push_back({i, results[current_res_idx].value});
                }
                current_res_idx++;
            }
        }

        if (DAG[k].empty()) {
            DAG[k].push_back({k, 1}); // Default freq 1 for single char
        }
    }

    // Viterbi route calculation
    route[N] = {0.0, 0};
    for (int i = (int)N - 1; i >= 0; --i) {
        double max_prob = -1e100;
        size_t best_x = 0;
        for (const auto& node : DAG[i]) {
            double prob = log(node.freq > 0 ? node.freq : 1) - logtotal + route[node.end_idx + 1].first;
            if (prob > max_prob) {
                max_prob = prob;
                best_x = node.end_idx;
            }
        }
        route[i] = {max_prob, best_x};
    }

    // Output segmentation
    py::list result;
    size_t x = 0;
    std::string buf;
    size_t buf_char_count = 0;

    auto process_buffer = [&]() {
        if (buf.empty()) return;
        if (buf_char_count == 1) {
            result.append(buf);
        } else {
            if (trie.search(buf) <= 0) {
                // Split buffer into Han and non-Han blocks
                std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
                std::u32string buf_u32 = conv.from_bytes(buf);
                std::u32string current_block;
                bool is_han = false;

                auto flush_sub_block = [&]() {
                    if (current_block.empty()) return;
                    if (is_han) {
                        std::vector<std::string> words = finalseg_viterbi_internal(current_block);
                        for (const auto& w : words) result.append(w);
                    } else {
                        // Further split non-Han block into alphanumeric and symbols
                        std::u32string sub;
                        bool is_alnum = false;
                        auto flush_alnum = [&]() {
                            if (sub.empty()) return;
                            result.append(u32_to_utf8(sub));
                            sub.clear();
                        };

                        for (char32_t ch : current_block) {
                            bool ch_alnum = (ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9');
                            if (sub.empty()) {
                                is_alnum = ch_alnum;
                            } else if (ch_alnum != is_alnum) {
                                flush_alnum();
                                is_alnum = ch_alnum;
                            }
                            sub += ch;
                        }
                        flush_alnum();
                    }
                    current_block.clear();
                };

                for (char32_t ch : buf_u32) {
                    bool ch_is_han = (ch >= 0x4E00 && ch <= 0x9FD5);
                    if (current_block.empty()) {
                        is_han = ch_is_han;
                    } else if (ch_is_han != is_han) {
                        flush_sub_block();
                        is_han = ch_is_han;
                    }
                    current_block += ch;
                }
                flush_sub_block();
            } else {
                std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
                std::u32string buf_u32 = conv.from_bytes(buf);
                for (char32_t ch : buf_u32) {
                    result.append(u32_to_utf8(std::u32string(1, ch)));
                }
            }
        }
        buf.clear();
        buf_char_count = 0;
    };

    while (x < N) {
        size_t y = route[x].second + 1;
        size_t start = offsets[x];
        size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);

        if (y - x == 1) {
            buf += word;
            buf_char_count++;
        } else {
            process_buffer();
            result.append(word);
        }
        x = y;
    }
    process_buffer();
    return result;
}

// Optimized C++ implementation of __cut_DAG_NO_HMM
py::list _cut_DAG_NO_HMM_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total
) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return py::list();

    std::vector<std::pair<double, size_t>> route(N + 1);
    double logtotal = log(total);

    // Calculate route directly
    route[N] = {0.0, 0};
    for (int i = (int)N - 1; i >= 0; --i) {
        double max_prob = -1e100;
        size_t best_x = i;

        size_t start = offsets[i];
        const char* ptr = sentence.data() + start;
        size_t remain_len = sentence.size() - start;

        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(
            ptr, results, 64, remain_len
        );

        // Single character fallback probability
        max_prob = log(1.0) - logtotal + route[i + 1].first;
        best_x = i;

        if (num > 0) {
            size_t current_res_idx = 0;
            for (size_t j = i; j < N && current_res_idx < num; ++j) {
                size_t prefix_len = offsets[j + 1] - start;
                if (prefix_len == results[current_res_idx].length) {
                    int freq = results[current_res_idx].value;
                    if (freq > 0) {
                        double prob = log((double)freq) - logtotal + route[j + 1].first;
                        if (prob > max_prob) {
                            max_prob = prob;
                            best_x = j;
                        }
                    }
                    current_res_idx++;
                }
            }
        }
        route[i] = {max_prob, best_x};
    }

    py::list result;
    size_t x = 0;
    std::string buf;

    while (x < N) {
        size_t y = route[x].second + 1;
        size_t start = offsets[x];
        size_t len = offsets[y] - start;
        std::string word(sentence.data() + start, len);

        if (y - x == 1 && ((word[0] >= 'a' && word[0] <= 'z') || (word[0] >= 'A' && word[0] <= 'Z') || (word[0] >= '0' && word[0] <= '9'))) {
            buf += word;
        } else {
            if (!buf.empty()) {
                result.append(buf);
                buf.clear();
            }
            result.append(word);
        }
        x = y;
    }
    if (!buf.empty()) result.append(buf);
    return result;
}

// Helper to get next char32_t from UTF-8 string
inline char32_t utf8_next_char(const char*& p, const char* end) {
    if (p >= end) return 0;
    unsigned char c = static_cast<unsigned char>(*p++);
    if (c < 0x80) return c;
    if ((c & 0xE0) == 0xC0) {
        if (p >= end) return c;
        char32_t res = (c & 0x1F) << 6;
        res |= (*p++ & 0x3F);
        return res;
    }
    if ((c & 0xF0) == 0xE0) {
        if (p + 1 >= end) return c;
        char32_t res = (c & 0x0F) << 12;
        res |= (*p++ & 0x3F) << 6;
        res |= (*p++ & 0x3F);
        return res;
    }
    if ((c & 0xF8) == 0xF0) {
        if (p + 2 >= end) return c;
        char32_t res = (c & 0x07) << 18;
        res |= (*p++ & 0x3F) << 12;
        res |= (*p++ & 0x3F) << 6;
        res |= (*p++ & 0x3F);
        return res;
    }
    return c;
}

inline bool is_han_alnum_fast(char32_t ch) {
    return (ch >= 0x4E00 && ch <= 0x9FD5) ||
           (ch >= U'a' && ch <= U'z') ||
           (ch >= U'A' && ch <= U'Z') ||
           (ch >= U'0' && ch <= U'9') ||
           ch == U'+' || ch == U'#' || ch == U'&' || ch == U'.' || ch == U'_' || ch == U'-' || ch == U'%';
}

inline bool is_eng_fast(char32_t ch) {
    return (ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9');
}

// Global internal cut dispatcher in C++ to avoid Python loop overhead
py::list _cut_internal_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total,
    bool HMM
) {
    py::list result;
    const char* p = sentence.data();
    const char* end = p + sentence.size();
    const char* block_start = nullptr;

    while (p < end) {
        const char* current_p = p;
        char32_t ch = utf8_next_char(p, end);
        if (is_han_alnum_fast(ch)) {
            if (!block_start) block_start = current_p;
        } else {
            if (block_start) {
                std::string block(block_start, current_p - block_start);
                py::list words = HMM ? _cut_DAG_cpp(trie, block, total) : _cut_DAG_NO_HMM_cpp(trie, block, total);
                for (auto w : words) result.append(w);
                block_start = nullptr;
            }
            // Handle separators (re_skip)
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

// Helper for cut_all to process a block
void _cut_all_block(
    DatTrie& trie,
    const std::string& sentence,
    py::list& result
) {
    const std::vector<size_t> offsets = get_utf8_offsets(sentence);
    const size_t N = offsets.size() - 1;
    if (N == 0) return;

    std::vector<std::vector<size_t>> DAG(N);
    for (size_t k = 0; k < N; k++) {
        size_t start = offsets[k];
        const char* ptr = sentence.data() + start;
        size_t remain_len = sentence.size() - start;

        cedar::da<int>::result_pair_type results[64];
        size_t num = trie.trie_ref().commonPrefixSearch<cedar::da<int>::result_pair_type>(
            ptr, results, 64, remain_len
        );

        if (num > 0) {
            for (size_t i = 0; i < num; ++i) {
                if (results[i].value > 0) {
                    size_t match_len = results[i].length;
                    for (size_t j = k; j < N; ++j) {
                        if (offsets[j+1] - start == match_len) {
                            DAG[k].push_back(j);
                            break;
                        }
                    }
                }
            }
        }
        if (DAG[k].empty()) DAG[k].push_back(k);
    }

    int old_j = -1;
    int eng_scan = 0;
    std::string eng_buf = "";

    for (size_t k = 0; k < N; k++) {
        const auto& L = DAG[k];

        // Character at k
        size_t start_k = offsets[k];
        size_t len_k = offsets[k+1] - start_k;
        std::string char_k = sentence.substr(start_k, len_k);

        // Check if char_k is english (only if it's 1 byte or specific ASCII)
        bool is_eng = (char_k.size() == 1 && ((char_k[0] >= 'a' && char_k[0] <= 'z') || (char_k[0] >= 'A' && char_k[0] <= 'Z') || (char_k[0] >= '0' && char_k[0] <= '9')));

        if (eng_scan == 1 && !is_eng) {
            eng_scan = 0;
            result.append(eng_buf);
            eng_buf = "";
        }

        if (L.size() == 1 && (int)k > old_j) {
            size_t start = offsets[k];
            size_t len = offsets[L[0] + 1] - start;
            std::string word = sentence.substr(start, len);

            if (is_eng && word.size() == len_k) { // Single english character
                if (eng_scan == 0) {
                    eng_scan = 1;
                    eng_buf = word;
                } else {
                    eng_buf += word;
                }
            } else {
                if (eng_scan == 0) {
                    result.append(word);
                }
            }
            old_j = (int)L[0];
        } else {
            for (size_t j : L) {
                if (j > k) {
                    size_t start = offsets[k];
                    size_t len = offsets[j + 1] - start;
                    result.append(sentence.substr(start, len));
                    old_j = (int)j;
                }
            }
        }
    }
    if (eng_scan == 1) result.append(eng_buf);
}

// Optimized C++ implementation of cut_all with block splitting
py::list _cut_all_internal_cpp(
    DatTrie& trie,
    const std::string& sentence
) {
    py::list result;
    const char* p = sentence.data();
    const char* end = p + sentence.size();
    const char* block_start = nullptr;
    bool last_is_han_alnum = false;

    while (p < end) {
        const char* current_p = p;
        char32_t ch = utf8_next_char(p, end);
        bool is_han_alnum = is_han_alnum_fast(ch);

        if (block_start == nullptr) {
            block_start = current_p;
            last_is_han_alnum = is_han_alnum;
        } else if (is_han_alnum != last_is_han_alnum) {
            std::string block(block_start, current_p - block_start);
            if (last_is_han_alnum) {
                _cut_all_block(trie, block, result);
            } else {
                // Non-han-alnum block: split by whitespace
                const char* sp = block.data();
                const char* send = sp + block.size();
                const char* s_start = nullptr;
                while (sp < send) {
                    const char* cur_sp = sp;
                    char32_t sch = utf8_next_char(sp, send);
                    bool is_space = (sch == U' ' || sch == U'\t' || sch == U'\r' || sch == U'\n' || sch == 0x3000);
                    if (is_space) {
                        if (s_start) {
                            result.append(std::string(s_start, cur_sp - s_start));
                            s_start = nullptr;
                        }
                    } else {
                        if (!s_start) s_start = cur_sp;
                    }
                }
                if (s_start) result.append(std::string(s_start, send - s_start));
            }
            block_start = current_p;
            last_is_han_alnum = is_han_alnum;
        }
    }
    if (block_start) {
        std::string block(block_start, end - block_start);
        if (last_is_han_alnum) {
            _cut_all_block(trie, block, result);
        } else {
            const char* sp = block.data();
            const char* send = sp + block.size();
            const char* s_start = nullptr;
            while (sp < send) {
                const char* cur_sp = sp;
                char32_t sch = utf8_next_char(sp, send);
                bool is_space = (sch == U' ' || sch == U'\t' || sch == U'\r' || sch == U'\n' || sch == 0x3000);
                if (is_space) {
                    if (s_start) {
                        result.append(std::string(s_start, cur_sp - s_start));
                        s_start = nullptr;
                    }
                } else {
                    if (!s_start) s_start = cur_sp;
                }
            }
            if (s_start) result.append(std::string(s_start, send - s_start));
        }
    }
    return result;
}

// C++ implementation of cut_for_search
py::list _cut_for_search_internal_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total,
    bool HMM
) {
    py::list words = _cut_internal_cpp(trie, sentence, total, HMM);
    py::list result;

    for (auto w_handle : words) {
        std::string w = w_handle.cast<std::string>();
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        std::u32string w_u32 = conv.from_bytes(w);
        size_t w_len = w_u32.length();

        if (w_len > 2) {
            for (size_t i = 0; i < w_len - 1; i++) {
                std::u32string gram2_u32 = w_u32.substr(i, 2);
                std::string gram2 = conv.to_bytes(gram2_u32);
                if (trie.search(gram2) > 0) result.append(gram2);
            }
        }
        if (w_len > 3) {
            for (size_t i = 0; i < w_len - 2; i++) {
                std::u32string gram3_u32 = w_u32.substr(i, 3);
                std::string gram3 = conv.to_bytes(gram3_u32);
                if (trie.search(gram3) > 0) result.append(gram3);
            }
        }
        result.append(w);
    }
    return result;
}

py::list _posseg_cut_internal_cpp(
    DatTrie& trie,
    const std::string& sentence,
    double total,
    bool HMM
) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    std::u32string sentence_u32 = conv.from_bytes(sentence);
    py::list result;

    std::u32string block;
    for (char32_t ch : sentence_u32) {
        bool is_han_alnum = (ch >= 0x4E00 && ch <= 0x9FD5) ||
                            (ch >= U'a' && ch <= U'z') ||
                            (ch >= U'A' && ch <= U'Z') ||
                            (ch >= U'0' && ch <= U'9') ||
                            ch == U'+' || ch == U'#' || ch == U'&' || ch == U'.' || ch == U'_';

        if (is_han_alnum) {
            block += ch;
        } else {
            if (!block.empty()) {
                std::string block_utf8 = conv.to_bytes(block);
                py::list words = HMM ? _posseg_cut_DAG_cpp(trie, block_utf8, total) : _posseg_cut_DAG_NO_HMM_cpp(trie, block_utf8, total);
                for (auto w : words) result.append(w);
                block.clear();
            }
            std::string ch_utf8 = conv.to_bytes(std::u32string(1, ch));
            result.append(Pair(ch_utf8, "x"));
        }
    }
    if (!block.empty()) {
        std::string block_utf8 = conv.to_bytes(block);
        py::list words = HMM ? _posseg_cut_DAG_cpp(trie, block_utf8, total) : _posseg_cut_DAG_NO_HMM_cpp(trie, block_utf8, total);
        for (auto w : words) result.append(w);
    }
    return result;
}

PYBIND11_MODULE(_jieba_fast_dat_functions_py3, m) {
    m.doc() = "pybind11 plugin for jieba_fast_dat C functions";

    py::class_<Pair>(m, "pair")
        .def(py::init<std::string, std::string>())
        .def_readwrite("word", &Pair::word)
        .def_readwrite("flag", &Pair::flag)
        .def("__str__", &Pair::toString)
        .def("__repr__", &Pair::repr)
        .def("__lt__", &Pair::operator<)
        .def("__eq__", &Pair::operator==)
        .def("__iter__", [](const Pair& p) {
            return py::make_iterator(&p.word, (&p.flag) + 1);
        }, py::keep_alive<0, 1>())
        .def(py::pickle(
            [](const Pair& p) { // __getstate__
                return py::make_tuple(p.word, p.flag);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                return Pair(t[0].cast<std::string>(), t[1].cast<std::string>());
            }
        ));

    py::class_<DatTrie>(m, "DatTrie")
        .def(py::init<>())
        .def("build", static_cast<double (DatTrie::*)(py::iterable)>(&DatTrie::build), py::arg("word_freqs_iterable"), "Builds the DatTrie from an iterable of (word, freq) pairs and returns the total frequency.")
        .def("clear", &DatTrie::clear)
        .def("search", static_cast<int (DatTrie::*)(const std::string&) const>(&DatTrie::search), py::arg("word"))
        .def("open", &_get_trie_pybind, py::arg("filename"), py::arg("offset") = 0)
        .def("save", &DatTrie::save, py::arg("filename"))
        .def("save_to_bytes", &DatTrie::save_to_bytes, "Saves the DatTrie to a byte string.")
        .def("load_from_bytes", &DatTrie::load_from_bytes, py::arg("data"), "Loads the DatTrie from a byte string.")
        .def("num_keys", &DatTrie::num_keys)
        .def("extract_words", &DatTrie::extract_words, py::arg("words_with_freqs"))
        .def("update_word_tag_tab", &DatTrie::update_word_tag_tab, py::arg("new_tab"), "Updates the word-tag tab from a Python dict.")
        .def("add_word", &DatTrie::add_word, py::arg("word"), py::arg("freq"), py::arg("tag") = "x", "Adds a word to the DatTrie with a given frequency.")
        .def("del_word", &DatTrie::del_word, py::arg("word"), "Deletes a word from the DatTrie.");

    m.def("_viterbi", &_viterbi_pybind,
          py::arg("obs"), py::arg("_states_py"), py::arg("start_p"), py::arg("trans_p"), py::arg("emip_p"));

    m.def("_calc", &_calc_pybind,
          py::arg("trie"), py::arg("sentence"), py::arg("DAG"), py::arg("route"), py::arg("total"));

    m.def("load_main_dict_from_path_pybind", &load_main_dict_from_path_pybind,
          py::arg("trie"), py::arg("filename"), py::arg("main_word_tag_tab"),
          "Loads the main dictionary from a file, builds the DatTrie and populates the word-tag tab.");

    m.def("load_hmm_model", &load_hmm_model,
          py::arg("start_p_dict"), py::arg("trans_p_dict"), py::arg("emit_p_dict"), py::arg("char_state_tab_p_dict"));

    m.def("load_finalseg_hmm_model", &load_finalseg_hmm_model,
          py::arg("start_p_dict"), py::arg("trans_p_dict"), py::arg("emit_p_dict"));

    m.def("_posseg_viterbi_cpp", &_posseg_viterbi_cpp, py::arg("obs"));

    m.def("_finalseg_viterbi_cpp", &_finalseg_viterbi_cpp, py::arg("obs"));

    m.def("_get_DAG_and_calc", &_get_DAG_and_calc_pybind,
          py::arg("trie"), py::arg("sentence"), py::arg("route"), py::arg("total"));

    m.def("_get_DAG", &_get_DAG,
          py::arg("trie"), py::arg("sentence"));

    m.def("_get_freq", &_get_freq,
          py::arg("trie"), py::arg("word"));

    m.def("_posseg_cut_DAG_cpp", &_posseg_cut_DAG_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"));

    m.def("_posseg_cut_DAG_NO_HMM_cpp", &_posseg_cut_DAG_NO_HMM_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"));

    m.def("_cut_internal_cpp", &_cut_internal_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"), py::arg("HMM"));

    m.def("_posseg_cut_internal_cpp", &_posseg_cut_internal_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"), py::arg("HMM"));

    m.def("_cut_all_internal_cpp", &_cut_all_internal_cpp,
          py::arg("trie"), py::arg("sentence"));

    m.def("_cut_for_search_internal_cpp", &_cut_for_search_internal_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"), py::arg("HMM"));

    m.def("_cut_DAG_cpp", &_cut_DAG_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"));

    m.def("_cut_DAG_NO_HMM_cpp", &_cut_DAG_NO_HMM_cpp,
          py::arg("trie"), py::arg("sentence"),
          py::arg("total"));

    m.def("load_userdict_pybind", &load_userdict_from_path_pybind,
          py::arg("trie"), py::arg("filename"),
          py::arg("user_word_tag_tab"), py::arg("batch_add_force_split_func"));

    m.def("_load_word_tag_pybind", &_load_word_tag_pybind,
          py::arg("filename"), py::arg("word_tag_tab_py"),
          "Loads word-tag pairs from a dictionary file into a Python dict in C++.");
}
