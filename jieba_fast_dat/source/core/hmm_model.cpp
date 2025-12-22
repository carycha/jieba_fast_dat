#include "hmm_model.hpp"
#include <algorithm>
#include <atomic>

namespace jieba_fast_dat {

std::shared_ptr<HMMModel> global_hmm_model = std::make_shared<HMMModel>();
std::shared_ptr<FinalHMMModel> global_final_hmm_model = std::make_shared<FinalHMMModel>();

void load_hmm_model(py::dict start_p_dict, py::dict trans_p_dict, py::dict emit_p_dict, py::dict char_state_tab_p_dict) {
    auto new_model = std::make_shared<HMMModel>();
    int tag_id_counter = 0;

    for (auto item : start_p_dict) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        std::string tag = state_tag[1].cast<std::string>();
        if (new_model->pos_tag_map.find(tag) == new_model->pos_tag_map.end()) {
            new_model->pos_tag_map[tag] = tag_id_counter;
            new_model->reverse_pos_tag_map.push_back(tag);
            tag_id_counter++;
        }
    }

    new_model->num_states = new_model->pos_tag_map.size() * 4;
    new_model->start_P.assign(new_model->num_states, new_model->MIN_FLOAT);
    new_model->trans_P_flat.assign(new_model->num_states * new_model->num_states, new_model->MIN_INF);
    new_model->emit_P.resize(new_model->num_states);
    new_model->trans_P_keys.resize(new_model->num_states);

    for (auto item : start_p_dict) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        char state = state_tag[0].cast<std::string>()[0];
        std::string tag = state_tag[1].cast<std::string>();
        double prob = item.second.cast<double>();
        int id = new_model->get_state_tag_id(tag, state);
        if (id != -1) new_model->start_P[id] = prob;
    }

    for (auto from_item : trans_p_dict) {
        py::tuple from_state_tag = from_item.first.cast<py::tuple>();
        char from_state = from_state_tag[0].cast<std::string>()[0];
        std::string from_tag = from_state_tag[1].cast<std::string>();
        int from_id = new_model->get_state_tag_id(from_tag, from_state);
        if (from_id == -1) continue;
        py::dict to_dict = from_item.second.cast<py::dict>();
        for (auto to_item : to_dict) {
            py::tuple to_state_tag = to_item.first.cast<py::tuple>();
            char to_state = to_state_tag[0].cast<std::string>()[0];
            std::string to_tag = to_state_tag[1].cast<std::string>();
            double prob = to_item.second.cast<double>();
            int to_id = new_model->get_state_tag_id(to_tag, to_state);
            if (to_id != -1) {
                new_model->trans_P_flat[from_id * new_model->num_states + to_id] = prob;
                new_model->trans_P_keys[from_id].push_back(to_id);
            }
        }
    }

    for (auto item : emit_p_dict) {
        py::tuple state_tag = item.first.cast<py::tuple>();
        char state = state_tag[0].cast<std::string>()[0];
        std::string tag = state_tag[1].cast<std::string>();
        int id = new_model->get_state_tag_id(tag, state);
        if (id == -1) continue;
        py::dict char_prob_dict = item.second.cast<py::dict>();
        for (auto char_item : char_prob_dict) {
            std::u32string ch_str = char_item.first.cast<std::u32string>();
            if (!ch_str.empty()) new_model->emit_P[id][ch_str[0]] = char_item.second.cast<double>();
        }
    }

    for (auto item : char_state_tab_p_dict) {
        std::u32string ch_str = item.first.cast<std::u32string>();
        if (!ch_str.empty()) {
            char32_t ch = ch_str[0];
            py::list state_tag_list = item.second.cast<py::list>();
            std::vector<int> state_ids;
            for(py::handle h : state_tag_list) {
                py::tuple state_tag = h.cast<py::tuple>();
                char state = state_tag[0].cast<std::string>()[0];
                std::string tag = state_tag[1].cast<std::string>();
                int id = new_model->get_state_tag_id(tag, state);
                if (id != -1) state_ids.push_back(id);
            }
            new_model->char_state_tab_P[ch] = state_ids;
        }
    }

    std::atomic_store(&global_hmm_model, new_model);
}

void load_finalseg_hmm_model(py::dict start_p_dict, py::dict trans_p_dict, py::dict emit_p_dict) {
    auto new_model = std::make_shared<FinalHMMModel>();
    new_model->start_P.assign(4, -3.14e100);
    new_model->trans_P.assign(4, std::vector<double>(4, -3.14e100));
    new_model->emit_P.resize(4);

    for (auto item : start_p_dict) {
        std::string state_str = item.first.cast<std::string>();
        double prob = item.second.cast<double>();
        if (new_model->state_map.count(state_str[0])) new_model->start_P[new_model->state_map.at(state_str[0])] = prob;
    }

    for (auto from_item : trans_p_dict) {
        char from_state = from_item.first.cast<std::string>()[0];
        int from_id = new_model->state_map.at(from_state);
        py::dict to_dict = from_item.second.cast<py::dict>();
        for (auto to_item : to_dict) {
            char to_state = to_item.first.cast<std::string>()[0];
            int to_id = new_model->state_map.at(to_state);
            new_model->trans_P[from_id][to_id] = to_item.second.cast<double>();
        }
    }

    for (auto item : emit_p_dict) {
        char state = item.first.cast<std::string>()[0];
        int id = new_model->state_map.at(state);
        py::dict char_prob_dict = item.second.cast<py::dict>();
        for (auto char_item : char_prob_dict) {
            std::u32string ch_str = char_item.first.cast<std::u32string>();
            if (!ch_str.empty()) new_model->emit_P[id][ch_str[0]] = char_item.second.cast<double>();
        }
    }
    new_model->initialized = true;
    std::atomic_store(&global_final_hmm_model, new_model);
}

} // namespace jieba_fast_dat
