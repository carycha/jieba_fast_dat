import gc
from collections import Counter
import _jieba_fast_dat_functions_py3 as c_funcs


# Helper to get object counts
def get_object_counts():
    gc.collect()
    return Counter(type(o).__name__ for o in gc.get_objects())


# Test for DatTrie memory leaks
def test_dat_trie_build_leak():
    initial_counts = get_object_counts()

    for _ in range(100):  # Repeat build multiple times
        trie = c_funcs.DatTrie()
        word_freqs = [("word" + str(i), i) for i in range(100)]
        trie.build(word_freqs)
        # trie goes out of scope and should be garbage collected

    del trie
    del word_freqs

    final_counts = get_object_counts()

    # We expect the number of DatTrie objects to return to initial or close to it
    # This is a basic check, more sophisticated checks might be needed if false positives occur
    assert (
        final_counts["DatTrie"] <= initial_counts["DatTrie"] + 1
    )  # Allow for some minor overhead


def test_dat_trie_open_save_leak(tmp_path):
    initial_counts = get_object_counts()

    temp_file = tmp_path / "temp_trie.dat"

    # Create a trie and save it
    trie_build = c_funcs.DatTrie()
    word_freqs = [("test" + str(i), i) for i in range(50)]
    trie_build.build(word_freqs)
    trie_build.save(str(temp_file))

    for _ in range(100):  # Repeatedly open and close
        trie_open = c_funcs.DatTrie()
        trie_open.open(str(temp_file))
        # trie_open goes out of scope

    del trie_build
    del word_freqs
    del trie_open

    final_counts = get_object_counts()

    assert final_counts["DatTrie"] <= initial_counts["DatTrie"] + 1


# Commenting out these tests as the functions are not directly exposed via pybind11
# def test_calc_pybind_leak():
#     # This test requires a DatTrie, sentence, DAG, route, and total_obj
#     # We'll create minimal valid inputs
#     trie = c_funcs.DatTrie()
#     word_freqs = [("a", 1), ("b", 1), ("c", 1), ("ab", 1), ("bc", 1)]
#     trie.build(word_freqs)

#     sentence = "abc"
#     DAG = {0: [0, 1], 1: [1, 2], 2: [2]} # Example DAG
#     route = {}
#     total = 3.0 # Example total

#     initial_counts = get_object_counts()

#     for _ in range(100):
#         # Need to re-initialize route for each call if it's modified in-place
#         current_route = {}
#         c_funcs._calc_pybind(trie, sentence, DAG, current_route, total)
#         # current_route should be garbage collected

#     del trie
#     del sentence
#     del DAG
#     del route
#     del total
#     del current_route

#     final_counts = get_object_counts()

#     # Check for common Python types that might leak
#     # This is a general check, specific types might need to be added
#     assert final_counts['dict'] <= initial_counts['dict'] + 5 # Allow some dict overhead
#     assert final_counts['list'] <= initial_counts['list'] + 5 # Allow some list overhead
#     assert final_counts['tuple'] <= initial_counts['tuple'] + 5 # Allow some tuple overhead

# # Test for _get_DAG_pybind memory leaks
# def test_get_dag_pybind_leak():
#     FREQ = {"a": 1, "b": 1, "c": 1, "ab": 1, "bc": 1}
#     sentence = "abc"
#     DAG = {} # This will be populated by the function

#     initial_counts = get_object_counts()

#     for _ in range(100):
#         current_DAG = {}
#         c_funcs._get_DAG_pybind(current_DAG, FREQ, sentence)
#         # current_DAG should be garbage collected

#     del FREQ
#     del sentence
#     del DAG
#     del current_DAG

#     final_counts = get_object_counts()

#     assert final_counts['dict'] <= initial_counts['dict'] + 5
#     assert final_counts['list'] <= initial_counts['list'] + 5

# # Test for _get_DAG_and_calc_pybind memory leaks
# def test_get_dag_and_calc_pybind_leak():
#     FREQ = {"a": 1, "b": 1, "c": 1, "ab": 1, "bc": 1}
#     sentence = "abc"
#     route = [] # This will be populated by the function
#     total = 3.0

#     initial_counts = get_object_counts()

#     for _ in range(100):
#         current_route = []
#         c_funcs._get_DAG_and_calc_pybind(FREQ, sentence, current_route, total)
#         # current_route should be garbage collected

#     del FREQ
#     del sentence
#     del route
#     del total
#     del current_route

#     final_counts = get_object_counts()

#     assert final_counts['dict'] <= initial_counts['dict'] + 5
#     assert final_counts['list'] <= initial_counts['list'] + 5

# # Test for _viterbi_pybind memory leaks
# def test_viterbi_pybind_leak():
#     obs = "abc"
#     states_py = "BMSE"
#     start_p = {"B": -0.2, "M": -100, "S": -0.1, "E": -100}
#     trans_p = {
#         "B": {"M": -0.1, "E": -0.2},
#         "M": {"M": -0.1, "B": -0.2},
#         "S": {"E": -0.1, "S": -0.2},
#         "E": {"B": -0.1, "M": -0.2},
#     }
#     emip_p = {
#         "B": {"a": -0.1, "b": -0.2},
#         "M": {"b": -0.1, "c": -0.2},
#         "S": {"a": -0.1, "c": -0.2},
#         "E": {"c": -0.1, "a": -0.2},
#     }

#     initial_counts = get_object_counts()

#     for _ in range(100):
#         c_funcs._viterbi_pybind(obs, states_py, start_p, trans_p, emip_p)

#     del obs
#     del states_py
#     del start_p
#     del trans_p
#     del emip_p

#     final_counts = get_object_counts()

#     assert final_counts['dict'] <= initial_counts['dict'] + 5
#     assert final_counts['list'] <= initial_counts['list'] + 5
#     assert final_counts['tuple'] <= initial_counts['tuple'] + 5
#     assert final_counts['str'] <= initial_counts['str'] + 5
