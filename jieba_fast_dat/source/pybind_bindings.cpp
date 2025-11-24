#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <math.h>
#include <stdlib.h>
#include <limits> // For std::numeric_limits
#include <array> // For std::array
#include <string> // For std::string
#include "cedar.h"
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

    // HMM parameters (optimized with vectors)
    std::vector<double> start_P;
    std::vector<std::vector<double>> trans_P;
    std::vector<std::unordered_map<char32_t, double>> emit_P;
    std::unordered_map<char32_t, std::vector<int>> char_state_tab_P;
    // For pruning, to replicate original logic
    std::vector<std::vector<int>> trans_P_keys;
}


namespace py = pybind11;

class DatTrie {
public:
    DatTrie() {}

    void build(const std::vector<std::pair<std::string, int>>& word_freqs) {
        trie_.clear();
        for (const auto& pair : word_freqs) {
            trie_.update(pair.first.c_str(), pair.first.length(), pair.second);
        }
    }

    int search(const std::string& word) {
        return trie_.exactMatchSearch<int>(word.c_str(), word.length());
    }

    int search(const char* s, size_t len) {
        return trie_.exactMatchSearch<int>(s, len);
    }

    int open(const std::string& filename, size_t offset = 0) {
        return trie_.open(filename.c_str(), "rb", offset);
    }

    int save(const std::string& filename) {
        return trie_.save(filename.c_str());
    }

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


int _calc_pybind(DatTrie& trie, py::sequence sentence, py::dict DAG, py::dict& route, py::object total_obj, py::dict user_freq)
{
    double total;
    if (py::isinstance<py::float_>(total_obj) || py::isinstance<py::int_>(total_obj)) {
        total = total_obj.cast<double>();
    } else {
        throw py::type_error("Expected a float or int object for 'total'.");
    }

    const py::ssize_t N = py::len(sentence);
    const double logtotal = log(total);
    double max_freq_val, fq_val, fq_2_val, fq_last_val;
    py::ssize_t max_x_val, idx, i, t_list_len, x_val;

    py::tuple temp_tuple = py::make_tuple(0, 0);
    route[py::cast(N)] = temp_tuple;

    for(idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq_val = std::numeric_limits<double>::lowest(); // Use lowest() for smallest possible double
        max_x_val = 0;

        py::object idx_key = py::cast(idx);

        if (!DAG.contains(idx_key)) {
            throw py::key_error("DAG does not contain key " + std::to_string(idx));
        }

        py::list t_list = DAG[idx_key].cast<py::list>();
        t_list_len = py::len(t_list);

        for(i = 0; i < t_list_len; i++)
        {
            fq_val = 1;
            x_val = get_long_from_py_object(t_list[i]);

            // PySequence_GetSlice(sentence, idx, x+1)
            py::slice slice_obj(py::cast(idx), py::cast(x_val + 1), py::none()); // Corrected slice constructor
            py::object slice_of_sentence_obj = sentence[slice_obj];
            std::string slice_of_sentence = slice_of_sentence_obj.cast<std::string>();

            // Check user_freq first
            if (user_freq.contains(slice_of_sentence_obj)) {
                fq_val = get_long_from_py_object(user_freq[slice_of_sentence_obj]);
            } else {
                fq_val = trie.search(slice_of_sentence);
                if (fq_val == -1) fq_val = 0;
            }
            if (fq_val == 0) fq_val = 1;


            // PyDict_GetItem(route, PyInt_FromLong((long)x + 1))
            py::object route_key = py::cast(x_val + 1);
            py::object t_tuple_obj = get_dict_item_safe(route, route_key);
            if (t_tuple_obj.is_none()) {
                throw py::key_error("route does not contain key " + std::to_string(x_val + 1));
            }
            py::tuple t_tuple = t_tuple_obj.cast<py::tuple>();

            // PyFloat_AsDouble(PyTuple_GetItem(t_tuple, 0))
            fq_2_val = get_double_from_py_object(t_tuple[0]);
            fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if(fq_last_val > max_freq_val)
            {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
            // pybind11 handles reference counting, no need for Py_DecRef
        }
        py::tuple tuple_last = py::make_tuple(max_freq_val, max_x_val);
        route[py::cast(idx)] = tuple_last;
    }
    return 1;
}

int _get_DAG_pybind(py::dict DAG, py::dict FREQ, py::sequence sentence)
{
    const py::ssize_t N = py::len(sentence);
    py::object frag; // Use py::object for frag to handle its changing type (item vs slice)
    py::ssize_t i, k;

    for(k = 0; k < N; k++)
    {
        py::list tmplist; // pybind11 list
        i = k;
        frag = sentence[k]; // Get item at k

        // Loop while i < N and FREQ contains frag
        while(i < N && FREQ.contains(frag))
        {
            // Check if FREQ[frag] is truthy (non-zero long)
            py::object freq_item = FREQ[frag];
            if (!freq_item.is_none() && get_long_from_py_object(freq_item))
            {
                tmplist.append(i);
            }
            i++;
            // Update frag to be a slice from k to i+1
            py::slice slice_obj(py::cast(k), py::cast(i + 1), py::none()); // Corrected slice constructor
            frag = sentence[slice_obj];
        }

        if (py::len(tmplist) == 0) {
            tmplist.append(k);
        }
        DAG[py::cast(k)] = tmplist;
    }
    return 1;
}

int _get_DAG_and_calc_pybind(DatTrie& trie, py::dict FREQ, py::sequence sentence, py::list route, double total)
{
    const py::ssize_t N = py::len(sentence);
    // Using std::vector for dynamic arrays
    std::vector<std::vector<py::ssize_t>> DAG(N);

    py::ssize_t k, i, idx, max_x_val;
    long fq_val;
    py::ssize_t x_val;
    py::object frag;
    py::object t_f_obj;
    py::object o_freq_obj;

    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);
    double max_freq_val;
    double fq_2_val, fq_last_val;

    _route[N][0] = 0;
    _route[N][1] = 0;

    for(k = 0; k < N; k++)
    {
        i = k;
        while(i < N)
        {
            py::slice slice_obj(py::cast(k), py::cast(i + 1), py::none());
            frag = sentence[slice_obj];

            bool found = false;
            int freq = 0;

            if (FREQ.contains(frag)) {
                 found = true;
                 freq = get_long_from_py_object(FREQ[frag]);
            } else {
                 std::string frag_str = frag.cast<std::string>();
                 freq = trie.search(frag_str);
                 if (freq != -1) {
                     found = true;
                 }
            }

            if (!found) {
                break;
            }

            if (freq > 0) {
                DAG[k].push_back(i);
            }
            i++;
        }
        if(DAG[k].empty())
        {
            DAG[k].push_back(k);
        }
    }


    for(idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq_val = std::numeric_limits<double>::lowest();
        max_x_val = 0;
        py::ssize_t t_list_len = DAG[idx].size();

        for(i = 0; i < t_list_len; i++)
        {
            fq_val = 1;
            x_val = DAG[idx][i];

            py::slice slice_obj(py::cast(idx), py::cast(x_val + 1), py::none());
            py::object slice_of_sentence = sentence[slice_obj];

            if (FREQ.contains(slice_of_sentence)) {
                fq_val = get_long_from_py_object(FREQ[slice_of_sentence]);
            } else {
                std::string slice_str = slice_of_sentence.cast<std::string>();
                int trie_freq = trie.search(slice_str);
                if (trie_freq != -1) {
                    fq_val = trie_freq;
                } else {
                    fq_val = 0;
                }
            }
            if (fq_val == 0) fq_val = 1;

            fq_2_val = _route[x_val + 1][0];
            fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if(fq_last_val >= max_freq_val)
            {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = static_cast<double>(max_x_val);
    }
    for(i = 0; i <= N; i++)
    {
        route.append(static_cast<long>(_route[i][1]));
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


// Helper struct for Viterbi result
struct ViterbiResult {
    double prob;
    std::vector<std::pair<std::u32string, std::string>> word_tags;
};

ViterbiResult posseg_viterbi_impl(const std::u32string& obs) {
    size_t obs_len = obs.length();
    if (obs_len == 0) {
        return {0.0, {}};
    }

    // Optimized: Use vector instead of unordered_map for V and mem_path
    // V[t][state] = prob
    std::vector<std::vector<double>> V(obs_len, std::vector<double>(HMM::NUM_STATES, HMM::MIN_INF));
    std::vector<std::vector<int>> mem_path(obs_len, std::vector<int>(HMM::NUM_STATES, -1));

    // Initialization
    char32_t first_char = obs[0];
    const std::vector<int>* initial_states;
    std::vector<int> all_states_vec;
    if (HMM::char_state_tab_P.count(first_char)) {
        initial_states = &HMM::char_state_tab_P.at(first_char);
    } else {
        for(size_t i=0; i< HMM::NUM_STATES; ++i) all_states_vec.push_back(static_cast<int>(i));
        initial_states = &all_states_vec;
    }

    for (int y : *initial_states) {
        double emit = HMM::MIN_FLOAT;
        if (static_cast<size_t>(y) < HMM::emit_P.size() && HMM::emit_P[y].count(first_char)) {
             emit = HMM::emit_P[y].at(first_char);
        }
        V[0][y] = HMM::start_P[y] + emit;
        mem_path[0][y] = -1; // Represents empty path
    }

    // Recursion
    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];

        std::vector<int> prev_states;
        // Find active states from previous step
        for(int i = 0; static_cast<size_t>(i) < HMM::NUM_STATES; ++i) {
            if (V[t-1][i] > HMM::MIN_INF && !HMM::trans_P_keys[i].empty()) {
                prev_states.push_back(i);
            }
        }

        std::set<int> prev_states_expect_next;
        for (int x : prev_states) {
            for (int y_ : HMM::trans_P_keys[x]) {
                prev_states_expect_next.insert(y_);
            }
        }

        std::set<int> obs_states;
        if (HMM::char_state_tab_P.count(current_char)) {
            const std::vector<int>& char_states = HMM::char_state_tab_P.at(current_char);
            std::set<int> char_states_set(char_states.begin(), char_states.end());
            std::set_intersection(prev_states_expect_next.begin(), prev_states_expect_next.end(),
                                  char_states_set.begin(), char_states_set.end(),
                                  std::inserter(obs_states, obs_states.begin()));
        } else {
             obs_states = prev_states_expect_next;
        }

        if (obs_states.empty()) {
             if (!prev_states_expect_next.empty()) {
                obs_states = prev_states_expect_next;
             } else {
                for(size_t i=0; i< HMM::NUM_STATES; ++i) obs_states.insert(i);
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
                double trans = HMM::trans_P[y0][y];
                if (trans == HMM::MIN_INF) continue;

                double current_prob = V[t - 1][y0] + trans;
                if (current_prob > max_prob) {
                    max_prob = current_prob;
                    best_prev_state = y0;
                }
            }
            V[t][y] = max_prob + em_p;
            mem_path[t][y] = best_prev_state;
        }
    }

    // Termination
    double final_max_prob = HMM::MIN_INF;
    int last_state = -1;

    for (int y = 0; static_cast<size_t>(y) < HMM::NUM_STATES; ++y) {
        if (V.back()[y] > final_max_prob) {
            final_max_prob = V.back()[y];
            last_state = y;
        }
    }

    if (last_state == -1) {
        return {0.0, {}};
    }

    // Path backtracking for states
    std::vector<std::pair<char, std::string>> states_route;
    int current_state = last_state;
    for (int t = obs_len - 1; t >= 0; --t) {
        if (current_state == -1) {
             break; // Should not happen in a valid path
        }
        int pos_tag_id = current_state / 4;
        std::string pos_tag = HMM::reverse_pos_tag_map[pos_tag_id];
        char state_char = HMM::reverse_state_map[current_state % 4];

        states_route.push_back({state_char, pos_tag});

        current_state = mem_path[t][current_state];
    }
    std::reverse(states_route.begin(), states_route.end());

    // Now, reconstruct words and their POS tags based on states_route and obs
    std::vector<std::pair<std::u32string, std::string>> word_pos_tags_route;
    size_t begin = 0;
    size_t nexti = 0;

    for (size_t i = 0; i < obs_len; ++i) {
        char state_char = states_route[i].first;
        std::string pos_tag = states_route[i].second;

        if (state_char == 'B') {
            begin = i;
        } else if (state_char == 'E') {
            word_pos_tags_route.push_back({obs.substr(begin, i + 1 - begin), pos_tag});
            nexti = i + 1;
        } else if (state_char == 'S') {
            word_pos_tags_route.push_back({obs.substr(i, 1), pos_tag});
            nexti = i + 1;
        }
    }
    // Handle any remaining part of the sentence
    if (nexti < obs_len) {
        std::u32string word_u32 = obs.substr(nexti);
        if (!states_route.empty()) {
            std::string pos_tag = states_route[nexti].second;
            word_pos_tags_route.push_back({word_u32, pos_tag});
        } else {
            word_pos_tags_route.push_back({word_u32, "x"});
        }
    }

    return {final_max_prob, word_pos_tags_route};
}

py::tuple _posseg_viterbi_cpp(std::u32string obs) {
    ViterbiResult result = posseg_viterbi_impl(obs);

    py::list word_pos_tags_route;
    for (const auto& item : result.word_tags) {
        word_pos_tags_route.append(py::make_tuple(item.first, item.second));
    }

    return py::make_tuple(result.prob, word_pos_tags_route);
}

// Helper to convert UTF-8 string to u32string
std::u32string utf8_to_u32(const std::string& s) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.from_bytes(s);
}

// Helper to convert u32string to UTF-8 string
std::string u32_to_utf8(const std::u32string& s) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.to_bytes(s);
}

void load_userdict_pybind(DatTrie& trie, py::dict user_freq, py::dict user_word_tag_tab, const std::string& filename, py::object add_force_split_func) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        // Handle BOM on first line
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && line[0] == '\xEF' && line[1] == '\xBB' && line[2] == '\xBF') {
                line = line.substr(3);
            }
        }

        // Trim whitespace (simple version, mostly for newline)
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
        size_t start = 0;
        while (start < line.size() && line[start] == ' ') {
            start++;
        }
        if (start < line.size()) {
            line = line.substr(start);
        } else {
            continue; // Empty line
        }

        if (line.empty()) continue;

        std::vector<std::string> parts;
        // Manual split to match Python's split(" ", 2)
        size_t pos = 0;
        int count = 0;
        while (pos < line.length() && count < 2) {
            size_t next_pos = line.find(' ', pos);
            if (next_pos == std::string::npos) {
                parts.push_back(line.substr(pos));
                pos = line.length();
            } else {
                parts.push_back(line.substr(pos, next_pos - pos));
                pos = next_pos + 1;
            }
            count++;
        }
        if (pos < line.length()) {
            parts.push_back(line.substr(pos));
        }

        if (parts.empty()) continue;

        const std::string& word_str = parts[0];
        int freq = 1000; // Default
        std::string tag = "x"; // Default

        if (parts.size() > 1) {
            // Check if parts[1] is digit
            bool is_digit = !parts[1].empty() && std::all_of(parts[1].begin(), parts[1].end(), ::isdigit);
            if (is_digit) {
                freq = std::stoi(parts[1]);
            } else {
                tag = parts[1];
            }
        }

        if (parts.size() == 3) {
            tag = parts[2];
        }

        py::str py_word = py::str(word_str);
        user_freq[py_word] = freq;
        if (!tag.empty()) {
            user_word_tag_tab[py_word] = py::str(tag);
        }

        // Add prefixes
        size_t len = word_str.length();
        const char* str_ptr = word_str.c_str();
        for (size_t i = 0; i < len; ) {
            // Move to next char
            size_t char_len = 1;
            unsigned char c = static_cast<unsigned char>(str_ptr[i]);
            if (c < 0x80) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;

            i += char_len;

            // Optimization: Check main dict (trie) using pointer and length
            if (trie.search(str_ptr, i) != -1) {
                continue;
            }

            // Now check user_freq
            py::str py_wfrag = py::str(str_ptr, i);
            if (user_freq.contains(py_wfrag)) {
                continue;
            }

            user_freq[py_wfrag] = 0;
        }

        if (freq == 0) {
            add_force_split_func(py_word);
        }
    }
}

void load_hmm_model(py::dict start_p, py::dict trans_p, py::dict emit_p, py::dict char_state_tab_p) {
    // Clear previous data
    HMM::pos_tag_map.clear();
    HMM::reverse_pos_tag_map.clear();
    HMM::start_P.clear();
    HMM::trans_P.clear();
    HMM::emit_P.clear();
    HMM::char_state_tab_P.clear();
    HMM::trans_P_keys.clear();

    // Build pos_tag maps from start_p keys
    int tag_id_counter = 0;
    for (auto item : start_p) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        std::string tag = state_tag[1].cast<std::string>();
        if (HMM::pos_tag_map.find(tag) == HMM::pos_tag_map.end()) {
            HMM::pos_tag_map[tag] = tag_id_counter;
            HMM::reverse_pos_tag_map.push_back(tag);
            tag_id_counter++;
        }
    }

    HMM::NUM_STATES = HMM::pos_tag_map.size() * 4;
    HMM::start_P.assign(HMM::NUM_STATES, HMM::MIN_FLOAT);
    HMM::trans_P.assign(HMM::NUM_STATES, std::vector<double>(HMM::NUM_STATES, HMM::MIN_INF));
    HMM::emit_P.assign(HMM::NUM_STATES, std::unordered_map<char32_t, double>());
    HMM::trans_P_keys.assign(HMM::NUM_STATES, std::vector<int>());

    // Populate start_P
    for (auto item : start_p) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        char state = state_tag[0].cast<std::string>()[0];
        std::string tag = state_tag[1].cast<std::string>();
        double prob = item.second.cast<double>();
        int id = HMM::get_state_tag_id(tag, state);
        if (id != -1) {
            HMM::start_P[id] = prob;
        }
    }

    // Populate trans_P and trans_P_keys
    for (auto from_item : trans_p) {
        py::tuple from_state_tag = from_item.first.cast<py::tuple>();
        char from_state = from_state_tag[0].cast<std::string>()[0];
        std::string from_tag = from_state_tag[1].cast<std::string>();
        int from_id = HMM::get_state_tag_id(from_tag, from_state);
        if (from_id == -1) continue;

        py::dict to_dict = from_item.second.cast<py::dict>();
        for (auto to_item : to_dict) {
            py::tuple to_state_tag = to_item.first.cast<py::tuple>();
            char to_state = to_state_tag[0].cast<std::string>()[0];
            std::string to_tag = to_state_tag[1].cast<std::string>();
            double prob = to_item.second.cast<double>();
            int to_id = HMM::get_state_tag_id(to_tag, to_state);
            if (to_id != -1) {
                HMM::trans_P[from_id][to_id] = prob;
                HMM::trans_P_keys[from_id].push_back(to_id);
            }
        }
    }

    // Populate emit_P
    for (auto item : emit_p) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        char state = state_tag[0].cast<std::string>()[0];
        std::string tag = state_tag[1].cast<std::string>();
        int id = HMM::get_state_tag_id(tag, state);
        if (id == -1) continue;

        py::dict char_prob_dict = item.second.cast<py::dict>();
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
    for (auto item : char_state_tab_p) {
        std::u32string ch_str = item.first.cast<std::u32string>();
        if (!ch_str.empty()) {
            char32_t ch = ch_str[0];
            py::tuple state_tag_tuple = item.second.cast<py::tuple>();
            std::vector<int> state_ids;
            for(auto state_tag_item : state_tag_tuple) {
                py::tuple state_tag = state_tag_item.cast<py::tuple>();
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

py::dict _get_DAG(DatTrie& trie, py::sequence sentence, py::dict user_freq) {
    py::dict DAG;
    const py::ssize_t N = py::len(sentence);

    for (py::ssize_t k = 0; k < N; k++) {
        py::list tmplist;
        py::ssize_t i = k;

        while (i < N) {
            py::slice slice_obj(py::cast(k), py::cast(i + 1), py::none());
            py::object frag = sentence[slice_obj];

            bool found = false;
            int freq = 0;

            if (user_freq.contains(frag)) {
                 found = true;
                 freq = get_long_from_py_object(user_freq[frag]);
            } else {
                 std::string frag_str = frag.cast<std::string>();
                 freq = trie.search(frag_str);
                 if (freq != -1) {
                     found = true;
                 }
            }

            if (!found) {
                break;
            }

            if (freq > 0) {
                tmplist.append(i);
            }

            i++;
        }

        if (py::len(tmplist) == 0) {
            tmplist.append(k);
        }
        DAG[py::cast(k)] = tmplist;
    }
    return DAG;
}

int _get_freq(DatTrie& trie, py::dict user_freq, py::object word) {
    if (user_freq.contains(word)) {
        return get_long_from_py_object(user_freq[word]);
    }
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

// Helper function to check if a u32string matches an english pattern
bool is_english(const std::u32string& s) {
    if (s.empty()) return false;
    for (char32_t ch : s) {
        if (!((ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9'))) {
            return false;
        }
    }
    return true;
}

// C++ implementation of posseg __cut_DAG
py::list _posseg_cut_DAG_cpp(
    DatTrie& trie,
    py::dict user_freq,
    const std::u32string& sentence,
    py::dict word_tag_tab,
    double total
) {
    size_t N = sentence.length();
    if (N == 0) {
        return py::list();
    }

    // Get DAG and route using the existing C++ function
    std::vector<std::vector<py::ssize_t>> DAG(N);
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);

    // Build DAG
    for (size_t k = 0; k < N; ++k) {
        py::ssize_t i = k;
        while (i < static_cast<py::ssize_t>(N)) {
            std::u32string frag = sentence.substr(k, i - k + 1);
            py::str frag_py = u32string_to_pystr(frag);

            bool found = false;
            int freq = 0;

            if (user_freq.contains(frag_py)) {
                found = true;
                freq = py::cast<int>(user_freq[frag_py]);
            } else {
                std::string frag_str = py::cast<std::string>(frag_py);
                freq = trie.search(frag_str);
                if (freq != -1) {
                    found = true;
                }
            }

            if (!found) {
                break;
            }

            if (freq > 0) {
                DAG[k].push_back(i);
            }
            i++;
        }
        if (DAG[k].empty()) {
            DAG[k].push_back(k);
        }
    }

    // Calculate route
    _route[N][0] = 0;
    _route[N][1] = 0;

    for (py::ssize_t idx = N - 1; idx >= 0; --idx) {
        double max_freq_val = std::numeric_limits<double>::lowest();
        py::ssize_t max_x_val = 0;

        for (py::ssize_t x_val : DAG[idx]) {
            std::u32string slice = sentence.substr(idx, x_val - idx + 1);
            py::str slice_py = u32string_to_pystr(slice);

            long fq_val = 1;
            if (user_freq.contains(slice_py)) {
                fq_val = py::cast<long>(user_freq[slice_py]);
            } else {
                std::string slice_str = py::cast<std::string>(slice_py);
                int freq = trie.search(slice_str);
                if (freq != -1) {
                    fq_val = freq;
                } else {
                    fq_val = 0;
                }
            }
            if (fq_val == 0) fq_val = 1;

            double fq_2_val = _route[x_val + 1][0];
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if (fq_last_val >= max_freq_val) {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = static_cast<double>(max_x_val);
    }

    // Now process the route to generate pairs
    py::list result;
    size_t x = 0;
    std::u32string buf;

    auto get_tag = [&word_tag_tab](const std::u32string& word) -> std::string {
        py::str word_py = u32string_to_pystr(word);
        if (word_tag_tab.contains(word_py)) {
            return py::cast<std::string>(word_tag_tab[word_py]);
        }
        return "x";
    };

    auto process_buffer = [&](const std::u32string& buffer) -> py::list {
        py::list buf_result;
        if (buffer.length() == 1) {
            std::string tag = get_tag(buffer);
            buf_result.append(py::make_tuple(u32string_to_pystr(buffer), tag));
        } else {
            // Check if buffer has frequency
            py::str buf_py = u32string_to_pystr(buffer);
            int buf_freq = 0;
            if (user_freq.contains(buf_py)) {
                buf_freq = py::cast<int>(user_freq[buf_py]);
            } else {
                std::string buf_str = py::cast<std::string>(buf_py);
                buf_freq = trie.search(buf_str);
                if (buf_freq == -1) buf_freq = 0;
            }

            if (!buf_freq) {  // No frequency found
                if (is_number(buffer)) {
                    buf_result.append(py::make_tuple(u32string_to_pystr(buffer), "m"));
                } else if (is_english(buffer)) {
                    buf_result.append(py::make_tuple(u32string_to_pystr(buffer), "eng"));
                } else {
                    // Use HMM to cut
                    ViterbiResult viterbi_result = posseg_viterbi_impl(buffer);
                    for (const auto& word_tag : viterbi_result.word_tags) {
                        buf_result.append(py::make_tuple(u32string_to_pystr(word_tag.first), word_tag.second));
                    }
                }
            } else {  // Has frequency - split into single chars
                for (char32_t ch : buffer) {
                    std::u32string ch_str(1, ch);
                    std::string tag = get_tag(ch_str);
                    buf_result.append(py::make_tuple(u32string_to_pystr(ch_str), tag));
                }
            }
        }
        return buf_result;
    };

    while (x < N) {
        size_t y = static_cast<size_t>(_route[x][1]) + 1;
        std::u32string l_word = sentence.substr(x, y - x);

        if (y - x == 1) {
            buf += l_word;
        } else {
            if (!buf.empty()) {
                py::list buf_pairs = process_buffer(buf);
                for (auto item : buf_pairs) {
                    result.append(item);
                }
                buf.clear();
            }
            std::string tag = get_tag(l_word);
            result.append(py::make_tuple(u32string_to_pystr(l_word), tag));
        }
        x = y;
    }

    if (!buf.empty()) {
        py::list buf_pairs = process_buffer(buf);
        for (auto item : buf_pairs) {
            result.append(item);
        }
    }

    return result;
}


// C++ implementation of posseg __cut_DAG_NO_HMM
py::list _posseg_cut_DAG_NO_HMM_cpp(
    DatTrie& trie,
    py::dict user_freq,
    const std::u32string& sentence,
    py::dict word_tag_tab,
    double total
) {
    size_t N = sentence.length();
    if (N == 0) {
        return py::list();
    }

    // Get DAG and route using the existing C++ function
    std::vector<std::vector<py::ssize_t>> DAG(N);
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);

    // Build DAG
    for (size_t k = 0; k < N; ++k) {
        py::ssize_t i = k;
        while (i < static_cast<py::ssize_t>(N)) {
            std::u32string frag = sentence.substr(k, i - k + 1);
            py::str frag_py = u32string_to_pystr(frag);

            bool found = false;
            int freq = 0;

            if (user_freq.contains(frag_py)) {
                 found = true;
                 freq = py::cast<int>(user_freq[frag_py]);
            } else {
                 std::string frag_str = py::cast<std::string>(frag_py);
                 freq = trie.search(frag_str);
                 if (freq != -1) {
                     found = true;
                 }
            }

            if (!found) {
                break;
            }

            if (freq > 0) {
                DAG[k].push_back(i);
            }
            i++;
        }
        if (DAG[k].empty()) {
            DAG[k].push_back(k);
        }
    }

    // Calculate route
    _route[N][0] = 0;
    _route[N][1] = 0;

    for (py::ssize_t idx = N - 1; idx >= 0; --idx) {
        double max_freq_val = std::numeric_limits<double>::lowest();
        py::ssize_t max_x_val = 0;

        for (py::ssize_t x_val : DAG[idx]) {
            std::u32string slice = sentence.substr(idx, x_val - idx + 1);
            py::str slice_py = u32string_to_pystr(slice);

            long fq_val = 1;
            if (user_freq.contains(slice_py)) {
                fq_val = py::cast<long>(user_freq[slice_py]);
            } else {
                std::string slice_str = py::cast<std::string>(slice_py);
                int freq = trie.search(slice_str);
                if (freq != -1) {
                    fq_val = freq;
                } else {
                    fq_val = 0;
                }
            }
            if (fq_val == 0) fq_val = 1;

            double fq_2_val = _route[x_val + 1][0];
            double fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if (fq_last_val >= max_freq_val) {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = static_cast<double>(max_x_val);
    }

    // Now process the route to generate pairs, mirroring Python's __cut_DAG_NO_HMM
    py::list result;
    size_t x = 0;
    std::u32string buf;

    auto get_tag_from_word_tag_tab = [&word_tag_tab](const std::u32string& word_u32) -> std::string {
        py::str word_py = u32string_to_pystr(word_u32);
        if (word_tag_tab.contains(word_py)) {
            return py::cast<std::string>(word_tag_tab[word_py]);
        }
        return "x";
    };

    while (x < N) {
        size_t y = static_cast<size_t>(_route[x][1]) + 1;
        std::u32string l_word_u32 = sentence.substr(x, y - x);

        if (l_word_u32.length() == 1) {
            char32_t ch = l_word_u32[0];
            if ((ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9')) {
                buf += l_word_u32;
            } else {
                if (!buf.empty()) {
                    result.append(py::make_tuple(u32string_to_pystr(buf), "eng"));
                    buf.clear();
                }
                result.append(py::make_tuple(u32string_to_pystr(l_word_u32), get_tag_from_word_tag_tab(l_word_u32)));
            }
        } else { // l_word_u32.length() > 1
            if (!buf.empty()) {
                result.append(py::make_tuple(u32string_to_pystr(buf), "eng"));
                buf.clear();
            }
            result.append(py::make_tuple(u32string_to_pystr(l_word_u32), get_tag_from_word_tag_tab(l_word_u32)));
        }
        x = y;
    }

    if (!buf.empty()) {
        result.append(py::make_tuple(u32string_to_pystr(buf), "eng"));
    }

    return result;
}




PYBIND11_MODULE(_jieba_fast_dat_functions_py3, m) {
    m.doc() = "pybind11 plugin for jieba_fast_dat C functions";

    py::class_<DatTrie>(m, "DatTrie")
        .def(py::init<>())
        .def("build", &DatTrie::build, py::arg("word_freqs"))
        .def("search", static_cast<int (DatTrie::*)(const std::string&)>(&DatTrie::search), py::arg("word"))
        .def("open", &_get_trie_pybind, py::arg("filename"), py::arg("offset") = 0)
        .def("save", &DatTrie::save, py::arg("filename"));

    m.def("_viterbi", &_viterbi_pybind,
          py::arg("obs"), py::arg("_states_py"), py::arg("start_p"), py::arg("trans_p"), py::arg("emip_p"));

    m.def("_calc", &_calc_pybind,
          py::arg("trie"), py::arg("sentence"), py::arg("DAG"), py::arg("route"), py::arg("total_obj"), py::arg("user_freq"));

    m.def("load_hmm_model", &load_hmm_model,
          py::arg("start_p"), py::arg("trans_p"), py::arg("emit_p"), py::arg("char_state_tab_p"));

    m.def("_posseg_viterbi_cpp", &_posseg_viterbi_cpp, py::arg("obs"));

    m.def("_get_DAG_and_calc", &_get_DAG_and_calc_pybind,
          py::arg("trie"), py::arg("user_freq"), py::arg("sentence"), py::arg("route"), py::arg("total"));

    m.def("_get_DAG", &_get_DAG,
          py::arg("trie"), py::arg("sentence"), py::arg("user_freq"));

    m.def("_get_freq", &_get_freq,
          py::arg("trie"), py::arg("user_freq"), py::arg("word"));

    m.def("_posseg_cut_DAG_cpp", &_posseg_cut_DAG_cpp,
          py::arg("trie"), py::arg("user_freq"), py::arg("sentence"),
          py::arg("word_tag_tab"), py::arg("total"));

    m.def("_posseg_cut_DAG_NO_HMM_cpp", &_posseg_cut_DAG_NO_HMM_cpp,
          py::arg("trie"), py::arg("user_freq"), py::arg("sentence"),
          py::arg("word_tag_tab"), py::arg("total"));

    m.def("load_userdict_pybind", &load_userdict_pybind,
          py::arg("trie"), py::arg("user_freq"), py::arg("user_word_tag_tab"),
          py::arg("filename"), py::arg("add_force_split_func"));
}
