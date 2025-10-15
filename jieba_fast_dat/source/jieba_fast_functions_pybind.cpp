#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of STL containers
#include <math.h>
#include <float.h> // For DBL_MAX
#include <vector>
#include <string>
#include <map>
#include <fstream> // For file operations
#include <sstream> // For string stream operations
#include <algorithm> // For std::sort

#include "darts.h" // Include the darts-clone library

namespace py = pybind11;

// Define a struct to hold word and frequency for DAT building
struct DictWord {
    std::string word;
    double frequency;
    int pos; // Part of speech, if needed

    // Custom comparison for sorting, required by Darts
    bool operator<(const DictWord& other) const {
        return word < other.word;
    }
};

class JiebaDict {
public:
    JiebaDict() : is_loaded(false) {}

    bool load_dict(const std::string& dict_path) {
        std::ifstream ifs(dict_path);
        if (!ifs.is_open()) {
            py::print("Error: Could not open dictionary file:", dict_path);
            return false;
        }

        std::vector<DictWord> dict_words;
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string word;
            double freq;
            std::string pos_str; // To consume the part of speech if present

            if (!(iss >> word >> freq)) {
                // Handle lines with only word or other formats
                freq = 3; // Default frequency
                // Try to read just the word
                iss.clear();
                iss.seekg(0);
                if (!(iss >> word)) {
                    continue; // Skip malformed lines
                }
            }
            // Consume the rest of the line (part of speech)
            std::getline(iss, pos_str);

            dict_words.push_back({word, freq});
        }
        ifs.close();

        if (dict_words.empty()) {
            py::print("Warning: Dictionary file is empty or malformed:", dict_path);
            return false;
        }

        // Sort words alphabetically, required by Darts
        std::sort(dict_words.begin(), dict_words.end());

        std::vector<const char*> keys;
        // Fix: Use std::size_t for key_lengths
        std::vector<std::size_t> key_lengths;
        std::vector<int> darts_values(dict_words.size()); // Darts stores int values, we'll map frequencies to indices

        for (size_t i = 0; i < dict_words.size(); ++i) {
            keys.push_back(dict_words[i].word.c_str());
            key_lengths.push_back(dict_words[i].word.length());
            darts_values[i] = i; // Store index as value
            word_frequencies[dict_words[i].word] = dict_words[i].frequency;
        }

        // Fix: Correct argument type for key_lengths.data()
        if (trie.build(dict_words.size(), keys.data(), key_lengths.data(), darts_values.data()) != 0) {
            py::print("Error: Failed to build Double-Array Trie.");
            return false;
        }

        is_loaded = true;
        py::print("Dictionary loaded successfully from:", dict_path, "with", dict_words.size(), "words.");
        return true;
    }

    // Lookup word frequency
    double get_word_frequency(const std::string& word) const {
        auto it = word_frequencies.find(word);
        if (it != word_frequencies.end()) {
            return it->second;
        }
        return 0.0; // Or a default frequency if not found
    }

    // Check if a word exists in the dictionary
    bool contains_word(const std::string& word) const {
        // Fix: Pass word.length() to exactMatchSearch
        return trie.exactMatchSearch<int>(word.c_str(), word.length()) >= 0;
    }

    // Get all words that are prefixes of the given text
    // Returns a vector of pairs: {end_position, word_id_in_trie}
    std::vector<std::pair<size_t, size_t>> common_prefix_search(const std::string& text) const {
        std::vector<std::pair<size_t, size_t>> result;
        Darts::DoubleArray::result_pair_type ret[1024]; // Max 1024 matches
        size_t num_matches = trie.commonPrefixSearch(text.c_str(), ret, 1024);

        for (size_t i = 0; i < num_matches; ++i) {
            result.push_back({ret[i].length, ret[i].value});
        }
        return result;
    }

    bool is_dict_loaded() const {
        return is_loaded;
    }

    double get_total_frequency() const {
        double total = 0.0;
        for (const auto& pair : word_frequencies) {
            total += pair.second;
        }
        return total;
    }

private:
    Darts::DoubleArray trie;
    std::map<std::string, double> word_frequencies; // To store frequencies
    bool is_loaded;
};

// Global instance of the dictionary
JiebaDict global_jieba_dict;

int _calc(py::object sentence, py::dict DAG, py::dict route, double total)
{
    const py::ssize_t N = py::len(sentence);
    const double logtotal = log(total);
    double max_freq, fq_last; // Declared fq_last here
    py::ssize_t max_x, idx, i, t_list_len, x;

    route[py::int_(N)] = py::make_tuple(py::int_(0), py::int_(0));

    for(idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq = -DBL_MAX; // Equivalent to INT_MIN for double
        max_x = 0;
        py::list t_list = DAG[py::int_(idx)];
        t_list_len = py::len(t_list);
        for(i = 0; i < t_list_len; i++)
        {
            double fq = 1;
            x = py::cast<py::ssize_t>(t_list[i]);
            py::object slice_of_sentence_obj = sentence.attr("__getitem__")(py::slice(idx, x + 1, 1));
            std::string slice_of_sentence_str = py::cast<std::string>(slice_of_sentence_obj);

            fq = global_jieba_dict.get_word_frequency(slice_of_sentence_str);

            if (fq == 0) fq = 1; // Still handle fq being 0
            py::tuple t_tuple = route[py::int_(x + 1)];
            double fq_2 = py::cast<double>(t_tuple[0]);
            fq_last = log(fq) - logtotal + fq_2;
            if(fq_last > max_freq)
            {
                max_freq = fq_last;
                max_x = x;
            }
        }
        route[py::int_(idx)] = py::make_tuple(max_freq, py::int_(max_x));
    }
    return 1;
}

int _get_DAG(py::dict DAG, py::object sentence)
{
    const py::ssize_t N = py::len(sentence);
    py::object frag_obj;
    std::string frag_str;
    py::ssize_t i, k;
    for(k = 0; k < N; k++)
    {
        py::list tmplist; // Equivalent to PyList_New(0)
        i = k;
        frag_obj = sentence.attr("__getitem__")(k); // Explicitly get item as py::object
        frag_str = py::cast<std::string>(frag_obj);
        while(i < N && global_jieba_dict.contains_word(frag_str)) // Use global_jieba_dict
        {
            if (global_jieba_dict.get_word_frequency(frag_str) > 0) // Check if frequency is non-zero
            {
                tmplist.append(py::int_(i)); // Equivalent to PyList_Append(tmplist, PyInt_FromLong((long)i))
            }
            i += 1;
            frag_obj = sentence.attr("__getitem__")(py::slice(k, i + 1, 1)); // Explicitly get slice as py::object
            frag_str = py::cast<std::string>(frag_obj);
        }
        if (py::len(tmplist) == 0) // Equivalent to PyList_Size(tmplist) == 0
            tmplist.append(py::int_(k)); // Equivalent to PyList_Append(tmplist, PyInt_FromLong((long)k))
        DAG[py::int_(k)] = tmplist; // Equivalent to PyDict_SetItem(DAG, PyInt_FromLong((long)k), tmplist)
    }
    return 1;
}

int _get_DAG_and_calc(py::object sentence, py::list route, double total)
{
    const py::ssize_t N = py::len(sentence);
    std::vector<std::vector<py::ssize_t>> DAG(N, std::vector<py::ssize_t>(20));
    std::vector<py::ssize_t> points(N, 0);
    std::vector<std::vector<double>> _route(N + 1, std::vector<double>(2));

    double logtotal = log(total);
    double max_freq_val;
    double fq_2, fq_last;

    _route[N][0] = 0;
    _route[N][1] = 0;

    for(py::ssize_t k = 0; k < N; k++)
    {
        py::ssize_t i = k;
        py::object frag_obj = sentence.attr("__getitem__")(k); // Explicitly get item as py::object
        std::string frag_str = py::cast<std::string>(frag_obj);

        while(i < N && (points[k] < 12))
        {
            double fq = global_jieba_dict.get_word_frequency(frag_str);
            if (fq > 0)
            {
                DAG[k][points[k]] = i;
                points[k]++;
            }
            i++;
            frag_obj = sentence.attr("__getitem__")(py::slice(k, i + 1, 1)); // Explicitly get slice as py::object
            frag_str = py::cast<std::string>(frag_obj);
        }
        if (points[k] == 0)
        {
            DAG[k][0] = k;
            points[k] = 1;
        }
    }

    for(py::ssize_t idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq_val = -DBL_MAX;
        py::ssize_t max_x = 0;
        py::ssize_t t_list_len = points[idx];
        for(py::ssize_t i = 0; i < t_list_len; i++)
        {
            double fq = 1;
            py::ssize_t x = DAG[idx][i];
            py::object slice_of_sentence_obj = sentence.attr("__getitem__")(py::slice(idx, x + 1, 1));
            std::string slice_of_sentence_str = py::cast<std::string>(slice_of_sentence_obj);

            fq = global_jieba_dict.get_word_frequency(slice_of_sentence_str);

            if (fq == 0) fq = 1;
            fq_2 = _route[x + 1][0];
            fq_last = log(fq) - logtotal + fq_2;
            if(fq_last >= max_freq_val)
            {
                max_freq_val = fq_last;
                max_x = x;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = (double)max_x;
    }
    for(py::ssize_t i = 0; i <= N; i++)
    {
        route.append(py::int_((long)_route[i][1]));
    }
    return 1;
}

py::tuple _viterbi(py::object obs, py::str _states, py::dict start_p, py::dict trans_p, py::dict emip_p)
{
    const py::ssize_t obs_len = py::len(obs);
    const int states_num = 4;

    std::vector<std::vector<double>> V(obs_len, std::vector<double>(22));
    std::vector<std::vector<char>> path(obs_len, std::vector<char>(22));

    std::string states_str = _states;
    const char* states = states_str.c_str();

    char y, best_state, y0, now_state;
    int p;

    std::map<char, std::string> PrevStatus_map;
    PrevStatus_map['B'] = "ES";
    PrevStatus_map['M'] = "MB";
    PrevStatus_map['S'] = "SE";
    PrevStatus_map['E'] = "BM";

    std::vector<py::object> emip_p_dict(states_num);
    std::vector<std::vector<py::object>> trans_p_dict(22, std::vector<py::object>(2));
    std::vector<py::object> py_states(states_num);

    for(int i = 0; i < states_num; i++)
        py_states[i] = py::str(std::string(1, states[i])); // Fix: Convert char to std::string

    emip_p_dict[0] = emip_p[py_states[0]];
    emip_p_dict[1] = emip_p[py_states[1]];
    emip_p_dict[2] = emip_p[py_states[2]];
    emip_p_dict[3] = emip_p[py_states[3]];

    trans_p_dict['B' - 'B'][0] = trans_p[py_states[2]];
    trans_p_dict['B' - 'B'][1] = trans_p[py_states[3]];
    trans_p_dict['M' - 'B'][0] = trans_p[py_states[1]];
    trans_p_dict['M' - 'B'][1] = trans_p[py_states[0]];
    trans_p_dict['E' - 'B'][0] = trans_p[py_states[0]];
    trans_p_dict['E' - 'B'][1] = trans_p[py_states[1]];
    trans_p_dict['S' - 'B'][0] = trans_p[py_states[3]];
    trans_p_dict['S' - 'B'][1] = trans_p[py_states[2]];

    for(int i = 0; i < states_num; i++)
    {
        py::object t_dict = emip_p[py_states[i]];
        double t_double = -DBL_MAX;
        py::object ttemp = obs.attr("__getitem__")(0); // Explicitly get item as py::object
        py::object item = t_dict.attr("get")(ttemp);
        if(!item.is_none())
            t_double = py::cast<double>(item);
        double t_double_2 = py::cast<double>(start_p[py_states[i]]);
        V[0][states[i] - 'B'] = t_double + t_double_2;
        path[0][states[i] - 'B'] = states[i];
    }

    for(py::ssize_t i = 1; i < obs_len; i++)
    {
        py::object t_obs = obs.attr("__getitem__")(i); // Explicitly get item as py::object
        for(int j = 0; j < states_num; j++)
        {
            double em_p = -DBL_MAX;
            y = states[j];
            py::object item = emip_p_dict[j].attr("get")(t_obs);
            if(!item.is_none())
                em_p = py::cast<double>(item);
            double max_prob = -DBL_MAX;
            best_state = ' ';
            for(p = 0; p < 2; p++)
            {
                double prob = em_p;
                y0 = PrevStatus_map[y][p];
                prob += V[i - 1][y0 - 'B'];
                item = trans_p_dict[y - 'B'][p].attr("get")(py_states[j]);
                if (item.is_none())
                    prob += -DBL_MAX;
                else
                    prob += py::cast<double>(item);
                if (prob > max_prob)
                {
                    max_prob = prob;
                    best_state = y0;
                }
            }
            if(best_state == ' ')
            {
                for(p = 0; p < 2; p++)
                {
                    y0 = PrevStatus_map[y][p];
                    if(y0 > best_state)
                        best_state = y0;
                }
            }
            V[i][y - 'B'] = max_prob;
            path[i][y - 'B'] = best_state;
        }
    }

    double max_prob_final = V[obs_len - 1]['E' - 'B'];
    best_state = 'E';

    if (V[obs_len - 1]['S' - 'B'] > max_prob_final)
    {
        max_prob_final = V[obs_len - 1]['S' - 'B'];
        best_state = 'S';
    }

    py::list t_list;
    now_state = best_state;

    for(py::ssize_t i = obs_len - 1; i >= 0; i--)
    {
        t_list.insert(0, py::str(std::string(1, now_state))); // Fix: Convert char to std::string
        now_state = path[i][now_state - 'B'];
    }

    return py::make_tuple(max_prob_final, t_list);
}

PYBIND11_MODULE(_jieba_fast_functions_pybind, m) {
    m.doc() = "pybind11 plugin for jieba_fast_dat"; // optional module docstring

    m.def("_calc", &_calc, "A function to calculate word frequencies");
    m.def("_get_DAG", &_get_DAG, "A function to get DAG");
    m.def("_get_DAG_and_calc", &_get_DAG_and_calc, "A function to get DAG and calculate");
    m.def("_viterbi", &_viterbi, "A function to perform Viterbi algorithm");
    m.def("load_dict", [](const std::string& dict_path) {
        return global_jieba_dict.load_dict(dict_path);
    }, "Load the dictionary from a given path.");
    m.def("get_total_frequency", []() {
        return global_jieba_dict.get_total_frequency();
    }, "Get the total frequency of words in the dictionary.");
    m.def("get_word_frequency", [](const std::string& word) {
        return global_jieba_dict.get_word_frequency(word);
    }, "Get the frequency of a specific word.");
}
