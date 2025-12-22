#ifndef JIEBA_FAST_DAT_SEGMENTER_HPP
#define JIEBA_FAST_DAT_SEGMENTER_HPP

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include "trie.hpp"
#include "types.hpp"

namespace py = pybind11;

namespace jieba_fast_dat {

int _calc_pybind(DatTrie& trie, const std::string& sentence, py::dict DAG, py::dict& route, double total);
int _get_DAG_pybind(py::dict DAG, py::dict FREQ, const std::string& sentence);
int _get_DAG_and_calc_pybind(DatTrie& trie, const std::string& sentence, py::list route, double total);
py::dict _get_DAG(DatTrie& trie, const std::string& sentence);
int _get_freq(DatTrie& trie, py::object word);

py::list _posseg_cut_DAG_cpp(DatTrie& trie, const std::string& sentence, double total);
py::list _posseg_cut_DAG_NO_HMM_cpp(DatTrie& trie, const std::string& sentence, double total);
py::list _cut_DAG_cpp(DatTrie& trie, const std::string& sentence, double total);
py::list _cut_DAG_NO_HMM_cpp(DatTrie& trie, const std::string& sentence, double total);

py::list _cut_internal_cpp(DatTrie& trie, const std::string& sentence, double total, bool HMM);
py::list _cut_all_internal_cpp(DatTrie& trie, const std::string& sentence);
py::list _cut_for_search_internal_cpp(DatTrie& trie, const std::string& sentence, double total, bool HMM);
py::list _posseg_cut_internal_cpp(DatTrie& trie, const std::string& sentence, double total, bool HMM);

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_SEGMENTER_HPP
