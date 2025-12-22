#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <pybind11/detail/common.h>

#include "core/types.hpp"
#include "core/utils.hpp"
#include "core/trie.hpp"
#include "core/hmm_model.hpp"
#include "core/viterbi_engine.hpp"
#include "core/dictionary.hpp"
#include "core/segmenter.hpp"
#include <atomic>

namespace py = pybind11;
using namespace jieba_fast_dat;

// Helper to get long from py::object
long get_long_from_py_object(py::object obj) {
    if (py::isinstance<py::int_>(obj)) return obj.cast<long>();
    throw py::type_error("Expected an integer object.");
}

// Helper to get double from py::object
double get_double_from_py_object(py::object obj) {
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) return obj.cast<double>();
    throw py::type_error("Expected a float or integer object.");
}

// Helper to safely get an item from a dict, returning a default if not found
py::object get_dict_item_safe(py::dict d, py::object key, py::object default_val = py::none()) {
    if (d.contains(key)) return d[key];
    return default_val;
}

// Viterbi wrapper for legacy/compatibility if needed
py::tuple _viterbi_pybind(py::sequence obs, py::str _states_py, py::dict start_p, py::dict trans_p, py::dict emip_p) {
    auto model = std::atomic_load(&global_hmm_model);
    const py::ssize_t obs_len = py::len(obs);
    const int states_num = 4;
    std::string states_str = _states_py.cast<std::string>();
    const char* states = states_str.c_str();
    std::array<std::string, 22> PrevStatus_str_cpp;
    PrevStatus_str_cpp['B'-'B'] = "ES"; PrevStatus_str_cpp['M'-'B'] = "MB";
    PrevStatus_str_cpp['S'-'B'] = "SE"; PrevStatus_str_cpp['E'-'B'] = "BM";
    std::vector<std::array<double, 22>> V(obs_len);
    std::vector<std::array<char, 22>> path(obs_len);
    std::array<py::str, 4> py_states_cpp;
    for(int i=0; i<states_num; ++i) py_states_cpp[i] = py::str(std::string(1, states[i]));
    std::array<py::dict, 4> emip_p_dict_cpp;
    for(int i=0; i<states_num; ++i) emip_p_dict_cpp[i] = emip_p[py_states_cpp[i]].cast<py::dict>();
    std::array<std::array<py::object, 2>, 22> trans_p_dict_cpp_obj;
    trans_p_dict_cpp_obj['B'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[2]);
    trans_p_dict_cpp_obj['B'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[3]);
    trans_p_dict_cpp_obj['M'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[1]);
    trans_p_dict_cpp_obj['M'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[0]);
    trans_p_dict_cpp_obj['E'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[0]);
    trans_p_dict_cpp_obj['E'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[1]);
    trans_p_dict_cpp_obj['S'-'B'][0] = get_dict_item_safe(trans_p, py_states_cpp[3]);
    trans_p_dict_cpp_obj['S'-'B'][1] = get_dict_item_safe(trans_p, py_states_cpp[2]);

    for(int i=0; i<states_num; ++i) {
        py::dict t_dict = emip_p_dict_cpp[i];
        double t_double_val = model->MIN_FLOAT;
        py::object ttemp_obj = obs[0];
        py::object item_obj = get_dict_item_safe(t_dict, ttemp_obj);
        if(!item_obj.is_none()) t_double_val = get_double_from_py_object(item_obj);
        py::object start_p_item_obj = get_dict_item_safe(start_p, py_states_cpp[i]);
        double t_double_2_val = model->MIN_FLOAT;
        if (!start_p_item_obj.is_none()) t_double_2_val = get_double_from_py_object(start_p_item_obj);
        V[0][states[i]-'B'] = t_double_val + t_double_2_val;
        path[0][states[i]-'B'] = states[i];
    }

    for(py::ssize_t i=1; i<obs_len; ++i) {
        py::object t_obs_obj = obs[i];
        for(int j=0; j<states_num; ++j) {
            double em_p_val = model->MIN_FLOAT;
            char y_char = states[j];
            py::object item_obj = get_dict_item_safe(emip_p_dict_cpp[j], t_obs_obj);
            if(!item_obj.is_none()) em_p_val = get_double_from_py_object(item_obj);
            double max_prob_val = model->MIN_FLOAT;
            char best_state_char = '\0';
            for(int p = 0; p < 2; ++p) {
                double prob_val = em_p_val;
                char y0_char = PrevStatus_str_cpp[y_char-'B'][p];
                prob_val += V[i - 1][y0_char-'B'];
                py::object trans_p_item_obj = get_dict_item_safe(trans_p_dict_cpp_obj[y_char-'B'][p], py_states_cpp[j]);
                if (trans_p_item_obj.is_none()) prob_val += model->MIN_FLOAT;
                else prob_val += get_double_from_py_object(trans_p_item_obj);
                if (prob_val > max_prob_val) { max_prob_val = prob_val; best_state_char = y0_char; }
            }
            if(best_state_char == '\0') {
                for(int p = 0; p < 2; p++) {
                    char y0_char_fallback = PrevStatus_str_cpp[y_char-'B'][p];
                    if(y0_char_fallback > best_state_char) best_state_char = y0_char_fallback;
                }
            }
            V[i][y_char-'B'] = max_prob_val;
            path[i][y_char-'B'] = best_state_char;
        }
    }
    double max_prob_final = V[obs_len-1]['E'-'B']; char best_state_final = 'E';
    if (V[obs_len-1]['S'-'B'] > max_prob_final) { max_prob_final = V[obs_len-1]['S'-'B']; best_state_final = 'S'; }
    py::list t_list_final; char now_state_char = best_state_final;
    for(py::ssize_t i = obs_len - 1; i >= 0; --i) {
        t_list_final.insert(0, py::str(std::string(1, now_state_char)));
        now_state_char = path[i][now_state_char-'B'];
    }
    return py::make_tuple(max_prob_final, t_list_final);
}

// Wrapper for posseg_viterbi_impl
py::tuple _posseg_viterbi_cpp_wrapper(std::u32string obs) {
    ViterbiResult result = posseg_viterbi_impl(obs);
    py::list word_pos_tags_route;
    for (auto& item : result.word_tags) word_pos_tags_route.append(std::move(item));
    return py::make_tuple(result.prob, word_pos_tags_route);
}

// Wrapper for finalseg_viterbi_internal returning (prob, states_str)
py::tuple _finalseg_viterbi_cpp_wrapper(std::u32string obs) {
    size_t obs_len = obs.length();
    auto model = std::atomic_load(&global_final_hmm_model);
    if (obs_len == 0 || !model->initialized) return py::make_tuple(0.0, py::list());
    // Re-run Viterbi to get both probability and path string
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
                if (prob > max_prob) { max_prob = prob; best_prev = y0; }
            }
            V[t][y] = max_prob; path[t][y] = best_prev;
        }
    }
    double max_prob_final = V[obs_len - 1][2]; int best_state_final = 2;
    if (V[obs_len - 1][3] > max_prob_final) { max_prob_final = V[obs_len - 1][3]; best_state_final = 3; }
    std::string res_states = ""; int curr = best_state_final;
    for (int t = static_cast<int>(obs_len) - 1; t >= 0; --t) { res_states += model->reverse_state_map[curr]; curr = path[t][curr]; }
    std::reverse(res_states.begin(), res_states.end());
    return py::make_tuple(max_prob_final, py::cast(res_states));
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
            return py::iter(py::make_tuple(p.word, p.flag));
        })
        .def(py::pickle(
            [](const Pair& p) { return py::make_tuple(p.word, p.flag); },
            [](py::tuple t) {
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return Pair(t[0].cast<std::string>(), t[1].cast<std::string>());
            }
        ));

    py::class_<DatTrie>(m, "DatTrie")
        .def(py::init<>())
        .def("build", static_cast<double (DatTrie::*)(py::iterable)>(&DatTrie::build), py::arg("word_freqs_iterable"))
        .def("clear", &DatTrie::clear)
        .def("search", static_cast<int (DatTrie::*)(const std::string&) const>(&DatTrie::search), py::arg("word"))
        .def("open", [](DatTrie& trie, const std::string& filename, size_t offset) {
            return trie.open(filename, offset);
        }, py::arg("filename"), py::arg("offset") = 0)
        .def("save", &DatTrie::save, py::arg("filename"))
        .def("save_all", &DatTrie::save_all, py::arg("filename"))
        .def("load_all", &DatTrie::load_all, py::arg("filename"))
        .def("save_to_bytes", &DatTrie::save_to_bytes)
        .def("load_from_bytes", &DatTrie::load_from_bytes, py::arg("data"))
        .def("num_keys", &DatTrie::num_keys)
        .def("extract_words", &DatTrie::extract_words, py::arg("words_with_freqs"))
        .def("update_word_tag_tab", &DatTrie::update_word_tag_tab, py::arg("new_tab"))
        .def("add_word", &DatTrie::add_word, py::arg("word"), py::arg("freq"), py::arg("tag") = "x")
        .def("del_word", &DatTrie::del_word, py::arg("word"))
        .def_readwrite("total_freq", &DatTrie::total_freq)
        .def(py::pickle(
            [](const DatTrie& d) { return d.save_to_bytes(); },
            [](py::bytes t) { auto d = std::make_unique<DatTrie>(); d->load_from_bytes(t); return d.release(); }
        ));

    m.def("_viterbi", &_viterbi_pybind, py::arg("obs"), py::arg("_states_py"), py::arg("start_p"), py::arg("trans_p"), py::arg("emip_p"));
    m.def("_calc", &_calc_pybind, py::arg("trie"), py::arg("sentence"), py::arg("DAG"), py::arg("route"), py::arg("total"));
    m.def("load_main_dict_from_path_pybind", &load_main_dict_from_path_pybind, py::arg("trie"), py::arg("filename"), py::arg("main_word_tag_tab"));
    m.def("load_hmm_model", &load_hmm_model, py::arg("start_p_dict"), py::arg("trans_p_dict"), py::arg("emit_p_dict"), py::arg("char_state_tab_p_dict"));
    m.def("load_finalseg_hmm_model", &load_finalseg_hmm_model, py::arg("start_p_dict"), py::arg("trans_p_dict"), py::arg("emit_p_dict"));
    m.def("_posseg_viterbi_cpp", &_posseg_viterbi_cpp_wrapper, py::arg("obs"));
    m.def("_finalseg_viterbi_cpp", &_finalseg_viterbi_cpp_wrapper, py::arg("obs"));
    m.def("_get_DAG_and_calc", &_get_DAG_and_calc_pybind, py::arg("trie"), py::arg("sentence"), py::arg("route"), py::arg("total"));
    m.def("_get_DAG", &_get_DAG, py::arg("trie"), py::arg("sentence"));
    m.def("_get_freq", &_get_freq, py::arg("trie"), py::arg("word"));
    m.def("_posseg_cut_DAG_cpp", &_posseg_cut_DAG_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"));
    m.def("_posseg_cut_DAG_NO_HMM_cpp", &_posseg_cut_DAG_NO_HMM_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"));
    m.def("_cut_internal_cpp", &_cut_internal_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"), py::arg("HMM"));
    m.def("_posseg_cut_internal_cpp", &_posseg_cut_internal_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"), py::arg("HMM"));
    m.def("_cut_all_internal_cpp", &_cut_all_internal_cpp, py::arg("trie"), py::arg("sentence"));
    m.def("_cut_for_search_internal_cpp", &_cut_for_search_internal_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"), py::arg("HMM"));
    m.def("_cut_DAG_cpp", &_cut_DAG_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"));
    m.def("_cut_DAG_NO_HMM_cpp", &_cut_DAG_NO_HMM_cpp, py::arg("trie"), py::arg("sentence"), py::arg("total"));
    m.def("load_userdict_pybind", &load_userdict_from_path_pybind, py::arg("trie"), py::arg("filename"), py::arg("user_word_tag_tab"), py::arg("batch_add_force_split_func"));
    m.def("_load_word_tag_pybind", &_load_word_tag_pybind, py::arg("filename"), py::arg("word_tag_tab_py"));
}
