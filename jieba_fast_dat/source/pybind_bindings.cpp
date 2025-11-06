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

// HMM model data structures
namespace HMM {
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

    // Combined ID for (state, pos_tag)
    // id = pos_tag_id * 4 + state_id
    int get_state_tag_id(const std::string& pos_tag, char state) {
        if (pos_tag_map.find(pos_tag) == pos_tag_map.end() || state_map.find(state) == state_map.end()) {
            return -1; // Not found
        }
        return pos_tag_map[pos_tag] * 4 + state_map.at(state);
    }

    // HMM parameters
    std::unordered_map<int, double> start_P;
    std::unordered_map<int, std::unordered_map<int, double>> trans_P;
    std::unordered_map<int, std::unordered_map<char32_t, double>> emit_P;
    std::unordered_map<char32_t, std::vector<int>> char_state_tab_P;
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


int _calc_pybind(DatTrie& trie, py::sequence sentence, py::dict DAG, py::dict& route, py::object total_obj)
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


            fq_val = trie.search(slice_of_sentence);
            if (fq_val == -1) fq_val = 0;
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

int _get_DAG_and_calc_pybind(py::dict FREQ, py::sequence sentence, py::list route, double total)
{
    const py::ssize_t N = py::len(sentence);
    // Using std::vector for dynamic arrays, similar to malloc behavior
    // DAG: Py_ssize_t (*DAG)[20] -> std::vector<std::array<py::ssize_t, 20>>
    std::vector<std::array<py::ssize_t, 20>> DAG(N);
    // points: Py_ssize_t *points -> std::vector<py::ssize_t>
    std::vector<py::ssize_t> points(N, 0); // Initialize with 0

    py::ssize_t k, i, idx, max_x_val;
    long fq_val; // Use long for fq
    py::ssize_t x_val;
    py::object frag; // Use py::object for frag
    py::object t_f_obj; // Use py::object for t_f
    py::object o_freq_obj; // Use py::object for o_freq

    // _route: double (*_route)[2] -> std::vector<std::array<double, 2>>
    std::vector<std::array<double, 2>> _route(N + 1);
    double logtotal = log(total);
    double max_freq_val; // Use double for max_freq
    double fq_2_val, fq_last_val;

    _route[N][0] = 0;
    _route[N][1] = 0;

    // points initialization is already done by std::vector constructor

    for(k = 0; k < N; k++)
    {
        i = k;
        frag = sentence[k]; // sentence[k]
        // while(i < N && (t_f = PyDict_GetItem(FREQ, frag)) && (points[k] < 12))
        while(i < N && !(t_f_obj = get_dict_item_safe(FREQ, frag)).is_none() && (points[k] < 12))
        {
            if(get_long_from_py_object(t_f_obj)) // get_long_from_py_object(t_f)
            {
                DAG[k][points[k]] = i;
                points[k]++;
            }
            i++;
            // pybind11 handles reference counting for frag
            py::slice slice_obj(py::cast(k), py::cast(i + 1), py::none()); // Corrected slice constructor
            frag = sentence[slice_obj]; // sentence.slice(k, i+1)
        }
        // pybind11 handles reference counting for frag
        if(points[k] == 0)
        {
            DAG[k][0] = k;
            points[k] = 1;
        }
    }


    for(idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq_val = std::numeric_limits<double>::lowest(); // Use lowest() for smallest possible double
        max_x_val = 0;
        py::ssize_t t_list_len = points[idx]; // points[idx]

        for(i = 0; i < t_list_len; i++)
        {
            fq_val = 1;
            x_val = DAG[idx][i]; // DAG[idx][i]

            py::slice slice_obj(py::cast(idx), py::cast(x_val + 1), py::none()); // Corrected slice constructor
            py::object slice_of_sentence = sentence[slice_obj]; // sentence.slice(idx, x+1)

            o_freq_obj = get_dict_item_safe(FREQ, slice_of_sentence);
            if (!o_freq_obj.is_none())
            {
                fq_val = get_long_from_py_object(o_freq_obj);
                if (fq_val == 0) fq_val = 1;
            }
            fq_2_val = _route[x_val + 1][0]; // _route[x+1][0]
            fq_last_val = log(static_cast<double>(fq_val)) - logtotal + fq_2_val;

            if(fq_last_val >= max_freq_val)
            {
                max_freq_val = fq_last_val;
                max_x_val = x_val;
            }
            // pybind11 handles reference counting
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = static_cast<double>(max_x_val); // Cast to double
    }
    for(i = 0; i <= N; i++)
    {
        route.append(static_cast<long>(_route[i][1])); // Append long to py::list
    }
    // No need to free std::vector
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

py::tuple _posseg_viterbi_cpp(py::sequence obs_py) {
    std::u32string obs;
    for (auto item : obs_py) {
        obs += item.cast<std::u32string>();
    }

    size_t obs_len = obs.length();
    if (obs_len == 0) {
        return py::make_tuple(0.0, py::list());
    }

    std::vector<std::unordered_map<int, double>> V(obs_len);
    std::vector<std::unordered_map<int, int>> path(obs_len);

    // Initialization
    char32_t first_char = obs[0];
    const std::vector<int>* states_to_check;
    std::vector<int> all_states;
    if (HMM::char_state_tab_P.find(first_char) != HMM::char_state_tab_P.end()) {
        states_to_check = &HMM::char_state_tab_P.at(first_char);
    } else {
        for(auto const& [key, val] : HMM::start_P) {
            all_states.push_back(key);
        }
        states_to_check = &all_states;
    }

    for (int y : *states_to_check) {
        double em_p = (HMM::emit_P.count(y) && HMM::emit_P.at(y).count(first_char)) ? HMM::emit_P.at(y).at(first_char) : MIN_FLOAT_VAL;
        double start_p = HMM::start_P.count(y) ? HMM::start_P.at(y) : MIN_FLOAT_VAL;
        V[0][y] = start_p + em_p;
        path[0][y] = -1;
    }

    // Recursion
    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];
        const std::vector<int>* next_states_to_check;
        std::vector<int> all_next_states;
        if (HMM::char_state_tab_P.find(current_char) != HMM::char_state_tab_P.end()) {
            next_states_to_check = &HMM::char_state_tab_P.at(current_char);
        } else {
            if (all_states.empty()) {
                 for(auto const& [key, val] : HMM::start_P) {
                    all_states.push_back(key);
                }
            }
            next_states_to_check = &all_states;
        }

        for (int y : *next_states_to_check) {
            double em_p = (HMM::emit_P.count(y) && HMM::emit_P.at(y).count(current_char)) ? HMM::emit_P.at(y).at(current_char) : MIN_FLOAT_VAL;
            double max_prob = MIN_FLOAT_VAL;
            int best_prev_state = -1;

            for (auto const& [y0, prob0] : V[t-1]) {
                 char current_state_char = HMM::reverse_state_map[y % 4];
                 char prev_state_char = HMM::reverse_state_map[y0 % 4];

                 bool valid_transition = false;
                 if ((current_state_char == 'B' || current_state_char == 'S') && (prev_state_char == 'E' || prev_state_char == 'S')) valid_transition = true;
                 if ((current_state_char == 'M' || current_state_char == 'E') && (prev_state_char == 'M' || prev_state_char == 'B')) valid_transition = true;

                 if (valid_transition) {
                    double trans_p = (HMM::trans_P.count(y0) && HMM::trans_P.at(y0).count(y)) ? HMM::trans_P.at(y0).at(y) : MIN_FLOAT_VAL;
                    double current_prob = prob0 + trans_p;
                    if (current_prob > max_prob) {
                        max_prob = current_prob;
                        best_prev_state = y0;
                    }
                 }
            }
            V[t][y] = max_prob + em_p;
            path[t][y] = best_prev_state;
        }
    }

    // Termination
    double final_max_prob = MIN_FLOAT_VAL;
    int last_state = -1;

    for (auto const& [pos_tag, tag_id] : HMM::pos_tag_map) {
        // Check 'S' state
        int s_state_id = HMM::get_state_tag_id(pos_tag, 'S');
        if (s_state_id != -1) {
            if (V[obs_len - 1].count(s_state_id)) {
                double prob = V[obs_len - 1].at(s_state_id);
                if (prob > final_max_prob) {
                    final_max_prob = prob;
                    last_state = s_state_id;
                }
            }
        }
        // Check 'E' state
        int e_state_id = HMM::get_state_tag_id(pos_tag, 'E');
        if (e_state_id != -1) {
            if (V[obs_len - 1].count(e_state_id)) {
                double prob = V[obs_len - 1].at(e_state_id);
                if (prob > final_max_prob) {
                    final_max_prob = prob;
                    last_state = e_state_id;
                }
            }
        }
    }

    if (last_state == -1) {
        // Fallback: if no E or S state, find any max
        for (auto const& [y, prob] : V[obs_len - 1]) {
            if (prob > final_max_prob) {
                final_max_prob = prob;
                last_state = y;
            }
        }
    }

    if (last_state == -1) {
        // All states have MIN_FLOAT probability.
        // Pick a default state.
        if (!states_to_check->empty()) {
            last_state = (*states_to_check)[0];
        } else {
            // This should be impossible.
            return py::make_tuple(0.0, py::list());
        }
    }

    // Path backtracking
    py::list result_path;
    for (int t = obs_len - 1; t >= 0; --t) {
        int pos_tag_id = last_state / 4;
        std::string pos_tag = HMM::reverse_pos_tag_map[pos_tag_id];
        char state_char = HMM::reverse_state_map[last_state % 4];
        py::tuple state_and_tag = py::make_tuple(std::string(1, state_char), pos_tag);
        result_path.insert(0, state_and_tag);

        if (path[t].find(last_state) == path[t].end()) break; // Should not happen
        last_state = path[t][last_state];
    }

    return py::make_tuple(final_max_prob, result_path);
}

void load_hmm_model(py::dict start_p, py::dict trans_p, py::dict emit_p, py::dict char_state_tab_p) {
    // Clear previous data
    HMM::pos_tag_map.clear();
    HMM::reverse_pos_tag_map.clear();
    HMM::start_P.clear();
    HMM::trans_P.clear();
    HMM::emit_P.clear();
    HMM::char_state_tab_P.clear();

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

    // Populate trans_P
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

PYBIND11_MODULE(_jieba_fast_dat_functions_py3, m) {
    m.doc() = "pybind11 plugin for jieba_fast_dat C functions";

    py::class_<DatTrie>(m, "DatTrie")
        .def(py::init<>())
        .def("build", &DatTrie::build, py::arg("word_freqs"))
        .def("search", &DatTrie::search, py::arg("word"))
        .def("open", &_get_trie_pybind, py::arg("filename"), py::arg("offset") = 0)
        .def("save", &DatTrie::save, py::arg("filename"));

    m.def("_viterbi", &_viterbi_pybind,
          py::arg("obs"), py::arg("_states_py"), py::arg("start_p"), py::arg("trans_p"), py::arg("emip_p"));

    m.def("_calc", &_calc_pybind,
          py::arg("trie"), py::arg("sentence"), py::arg("DAG"), py::arg("route"), py::arg("total_obj"));

    m.def("load_hmm_model", &load_hmm_model,
          py::arg("start_p"), py::arg("trans_p"), py::arg("emit_p"), py::arg("char_state_tab_p"));

    m.def("_posseg_viterbi_cpp", &_posseg_viterbi_cpp, py::arg("obs"));
}
