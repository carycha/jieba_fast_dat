#ifndef JIEBA_FAST_DAT_VITERBI_ENGINE_HPP
#define JIEBA_FAST_DAT_VITERBI_ENGINE_HPP

#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <atomic>
#include "types.hpp"
#include "utils.hpp"
#include "hmm_model.hpp"

namespace jieba_fast_dat {

struct ViterbiPool {
    std::vector<double> V;
    std::vector<int> path;
    std::vector<int> prev_states;
    std::vector<int> obs_states;
    std::vector<bool> states_mask;

    void reserve(size_t n, size_t num_states) {
        if (V.size() < n * num_states) V.resize(n * num_states);
        if (path.size() < n * num_states) path.resize(n * num_states);
        if (states_mask.size() < num_states) states_mask.resize(num_states);
        prev_states.reserve(num_states);
        obs_states.reserve(num_states);
    }
};

extern thread_local ViterbiPool posseg_pool;

inline ViterbiResult posseg_viterbi_impl(const std::u32string& obs) {
    size_t obs_len = obs.length();
    if (obs_len == 0) return {0.0, {}};

    // Atomically load current model
    auto model = std::atomic_load(&global_hmm_model);
    size_t num_states = model->num_states;
    if (num_states == 0) return {0.0, {}};

    posseg_pool.reserve(obs_len, num_states);
    double* V = posseg_pool.V.data();
    int* mem_path = posseg_pool.path.data();

    std::fill(V, V + obs_len * num_states, model->MIN_INF);

    char32_t first_char = obs[0];
    const std::vector<int>* initial_states;
    std::vector<int> all_states_vec;
    auto it_tab = model->char_state_tab_P.find(first_char);
    if (it_tab != model->char_state_tab_P.end()) {
        initial_states = &it_tab->second;
    } else {
        all_states_vec.reserve(num_states);
        for(size_t i=0; i< num_states; ++i) all_states_vec.push_back(static_cast<int>(i));
        initial_states = &all_states_vec;
    }

    for (int y : *initial_states) {
        double emit = model->MIN_FLOAT;
        if (static_cast<size_t>(y) < model->emit_P.size()) {
            auto it_emit = model->emit_P[y].find(first_char);
            if (it_emit != model->emit_P[y].end()) emit = it_emit->second;
        }
        V[y] = model->start_P[y] + emit;
    }

    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];
        posseg_pool.prev_states.clear();
        for(size_t i = 0; i < num_states; ++i) {
            if (V[(t-1) * num_states + i] > model->MIN_INF) {
                posseg_pool.prev_states.push_back(static_cast<int>(i));
            }
        }
        if (posseg_pool.prev_states.empty()) break;

        posseg_pool.obs_states.clear();
        std::fill(posseg_pool.states_mask.begin(), posseg_pool.states_mask.end(), false);

        auto it_tab_curr = model->char_state_tab_P.find(current_char);
        if (it_tab_curr != model->char_state_tab_P.end()) {
            const std::vector<int>& char_states = it_tab_curr->second;
            for (int y : char_states) posseg_pool.states_mask[y] = true;
            for (int x : posseg_pool.prev_states) {
                for (int y_next : model->trans_P_keys[x]) {
                    if (posseg_pool.states_mask[y_next]) {
                        posseg_pool.obs_states.push_back(y_next);
                        posseg_pool.states_mask[y_next] = false;
                    }
                }
            }
        } else {
            for (int x : posseg_pool.prev_states) {
                for (int y_next : model->trans_P_keys[x]) {
                    if (!posseg_pool.states_mask[y_next]) {
                        posseg_pool.obs_states.push_back(y_next);
                        posseg_pool.states_mask[y_next] = true;
                    }
                }
            }
        }

        if (posseg_pool.obs_states.empty()) {
            for (int x : posseg_pool.prev_states) {
                for (int y_next : model->trans_P_keys[x]) {
                    if (!posseg_pool.states_mask[y_next]) {
                        posseg_pool.obs_states.push_back(y_next);
                        posseg_pool.states_mask[y_next] = true;
                    }
                }
            }
        }

        for (int y : posseg_pool.obs_states) {
            double max_prob = model->MIN_INF;
            int best_prev_state = -1;
            double em_p = model->MIN_FLOAT;
            if (static_cast<size_t>(y) < model->emit_P.size()) {
                auto it_emit = model->emit_P[y].find(current_char);
                if (it_emit != model->emit_P[y].end()) em_p = it_emit->second;
            }
            for (int y0 : posseg_pool.prev_states) {
                double trans = model->get_trans_P(y0, y);
                if (trans == model->MIN_INF) continue;
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

    double final_max_prob = model->MIN_INF;
    int last_state = -1;
    size_t last_idx_base = (obs_len - 1) * num_states;
    for (size_t y = 0; y < num_states; ++y) {
        if (V[last_idx_base + y] > final_max_prob) {
            final_max_prob = V[last_idx_base + y];
            last_state = static_cast<int>(y);
        }
    }
    if (last_state == -1) return {0.0, {}};

    std::vector<int> path_ids(obs_len);
    int curr = last_state;
    for (int t = static_cast<int>(obs_len) - 1; t >= 0; --t) {
        path_ids[t] = curr;
        curr = mem_path[t * num_states + curr];
    }

    std::vector<Pair> word_pos_tags_route;
    size_t begin = 0;
    for (size_t i = 0; i < obs_len; ++i) {
        int state_id = path_ids[i];
        int pos_tag_id = state_id / 4;
        char state_char = model->reverse_state_map[state_id % 4];
        std::string pos_tag = model->reverse_pos_tag_map[pos_tag_id];

        if (state_char == 'B') begin = i;
        else if (state_char == 'E') word_pos_tags_route.emplace_back(u32_to_utf8(obs.substr(begin, i + 1 - begin)), pos_tag);
        else if (state_char == 'S') word_pos_tags_route.emplace_back(u32_to_utf8(obs.substr(i, 1)), pos_tag);
    }
    return {final_max_prob, word_pos_tags_route};
}

inline std::vector<std::string> finalseg_viterbi_internal(const std::u32string& obs) {
    size_t obs_len = obs.length();
    if (obs_len == 0) return {};

    auto model = std::atomic_load(&global_final_hmm_model);
    if (!model->initialized) {
        std::vector<std::string> words;
        words.reserve(obs_len);
        for (char32_t ch : obs) words.push_back(u32_to_utf8(std::u32string(1, ch)));
        return words;
    }

    std::vector<std::array<double, 4>> V(obs_len);
    std::vector<std::array<int, 4>> path(obs_len);

    for (int i = 0; i < 4; ++i) {
        double emit = -3.14e100;
        auto it_emit = model->emit_P[i].find(obs[0]);
        if (it_emit != model->emit_P[i].end()) emit = it_emit->second;
        V[0][i] = model->start_P[i] + emit;
        path[0][i] = i;
    }

    for (size_t t = 1; t < obs_len; ++t) {
        char32_t current_char = obs[t];
        for (int y = 0; y < 4; ++y) {
            double em_p = -3.14e100;
            auto it_emit = model->emit_P[y].find(current_char);
            if (it_emit != model->emit_P[y].end()) em_p = it_emit->second;
            double max_prob = -std::numeric_limits<double>::infinity();
            int best_prev = -1;
            for (int y0 : model->prev_states[y]) {
                double prob = V[t - 1][y0] + model->trans_P[y0][y] + em_p;
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
        res_states += model->reverse_state_map[curr];
        curr = path[t][curr];
    }
    std::reverse(res_states.begin(), res_states.end());

    std::vector<std::string> words;
    size_t begin = 0;
    for (size_t i = 0; i < obs_len; ++i) {
        char pos = res_states[i];
        if (pos == 'B') begin = i;
        else if (pos == 'E') words.push_back(u32_to_utf8(obs.substr(begin, i + 1 - begin)));
        else if (pos == 'S') words.push_back(u32_to_utf8(obs.substr(i, 1)));
    }
    return words;
}

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_VITERBI_ENGINE_HPP
