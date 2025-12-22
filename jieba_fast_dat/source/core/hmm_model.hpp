#ifndef JIEBA_FAST_DAT_HMM_MODEL_HPP
#define JIEBA_FAST_DAT_HMM_MODEL_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jieba_fast_dat {

struct HMMModel {
    const double MIN_FLOAT = -3.14e100;
    const double MIN_INF = -std::numeric_limits<double>::infinity();

    const std::unordered_map<char, int> state_map = {
        {'B', 0}, {'M', 1}, {'E', 2}, {'S', 3}
    };
    const std::vector<char> reverse_state_map = {'B', 'M', 'E', 'S'};

    std::unordered_map<std::string, int> pos_tag_map;
    std::vector<std::string> reverse_pos_tag_map;
    size_t num_states = 0;

    std::vector<double> start_P;
    std::vector<double> trans_P_flat;
    std::vector<std::unordered_map<char32_t, double>> emit_P;
    std::unordered_map<char32_t, std::vector<int>> char_state_tab_P;
    std::vector<std::vector<int>> trans_P_keys;

    inline int get_state_tag_id(const std::string& pos_tag, char state) const {
        auto it = pos_tag_map.find(pos_tag);
        if (it == pos_tag_map.end()) return -1;
        return it->second * 4 + state_map.at(state);
    }

    inline double get_trans_P(int from, int to) const {
        return trans_P_flat[from * num_states + to];
    }
};

struct FinalHMMModel {
    std::vector<double> start_P;
    std::vector<std::vector<double>> trans_P;
    std::vector<std::unordered_map<char32_t, double>> emit_P;
    bool initialized = false;

    const std::unordered_map<char, int> state_map = {
        {'B', 0}, {'M', 1}, {'E', 2}, {'S', 3}
    };
    const std::vector<char> reverse_state_map = {'B', 'M', 'E', 'S'};
    const std::vector<std::vector<int>> prev_states = {
        {2, 3}, {1, 0}, {0, 1}, {3, 2}
    };
};

// Global shared pointers for thread-safe access
extern std::shared_ptr<HMMModel> global_hmm_model;
extern std::shared_ptr<FinalHMMModel> global_final_hmm_model;

void load_hmm_model(py::dict start_p_dict, py::dict trans_p_dict, py::dict emit_p_dict, py::dict char_state_tab_p_dict);
void load_finalseg_hmm_model(py::dict start_p_dict, py::dict trans_p_dict, py::dict emit_p_dict);

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_HMM_MODEL_HPP
