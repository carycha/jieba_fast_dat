import jieba.posseg as original_posseg
import jieba_fast_dat._jieba_fast_dat_functions_py3 as fast_functions

# Corrected import: import the viterbi function directly
from jieba.posseg.viterbi import viterbi as original_python_viterbi_function

# Load HMM parameters from original jieba
original_start_P = original_posseg.start_P
original_trans_P = original_posseg.trans_P
original_emit_P = original_posseg.emit_P
original_char_state_tab_P = original_posseg.char_state_tab_P

print("Original jieba HMM parameters loaded.")

# Load HMM model into jieba_fast_dat's C++ backend
fast_functions.load_hmm_model(
    original_start_P, original_trans_P, original_emit_P, original_char_state_tab_P
)
print("HMM model loaded into fast_dat C++ backend.")

# Test sentence from the failing test case
test_sentence = "台灣的台北是一個充滿活力的城市，這裡有許多電腦和手機的程式設計師。"

print(f"\n--- Comparing Viterbi output for sentence: '{test_sentence}' ---")

# --- Original Python Viterbi ---
# Call the directly imported viterbi function
python_prob, python_pos_list = original_python_viterbi_function(
    test_sentence,
    original_char_state_tab_P,  # Passed as 'states'
    original_start_P,
    original_trans_P,
    original_emit_P,
)

print("\nOriginal Python Viterbi Result:")
print(f"  Probability: {python_prob}")
print(f"  POS List: {python_pos_list}")

# --- C++ Viterbi ---
# The C++ _posseg_viterbi_cpp function expects a sequence of characters.
# It internally uses the loaded HMM model.
cpp_prob, cpp_pos_list = fast_functions._posseg_viterbi_cpp(test_sentence)

print("\nC++ Viterbi Result:")
print(f"  Probability: {cpp_prob}")
print(f"  POS List: {cpp_pos_list}")

# --- Comparison ---
print("\n--- Detailed Comparison ---")

# Convert Python pos_list to a list of (state, tag) tuples for easier comparison
python_pairs = []
for _i, char_info in enumerate(python_pos_list):
    # char_info is a tuple (state_char, pos_tag)
    python_pairs.append(char_info)

# Convert C++ pos_list to a list of (state, tag) tuples for easier comparison
cpp_pairs = []
for _i, char_info in enumerate(cpp_pos_list):
    # char_info is a tuple (state_char, pos_tag)
    cpp_pairs.append(char_info)

mismatch_found = False
if len(python_pairs) != len(cpp_pairs):
    print(
        f"Length mismatch: Python has {len(python_pairs)} items, "
        f"C++ has {len(cpp_pairs)} items."
    )
    mismatch_found = True
else:
    for i in range(len(python_pairs)):
        if python_pairs[i] != cpp_pairs[i]:
            print(f"Mismatch at index {i}:")
            print(f"  Python: {python_pairs[i]}")
            print(f"  C++:    {cpp_pairs[i]}")
            mismatch_found = True

if not mismatch_found:
    print("No mismatches found in POS lists (word and tag).")
else:
    print("Mismatches found in POS lists.")

print("\n--- Comparison complete ---")
