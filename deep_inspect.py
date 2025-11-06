import pickle
import os


def inspect_model_file(filepath):
    print(f"--- Inspecting: {filepath} ---")
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        print(f"  Type: {type(data)}")

        if isinstance(data, dict):
            print(f"  Number of items: {len(data)}")
            # Inspect first few items to understand structure
            for i, (key, value) in enumerate(data.items()):
                if i >= 5:  # Limit to first 5 items for brevity
                    print("  ...")
                    break
                print(
                    f"    Key Type: {type(key)}, Key Value (first 50 chars): {str(key)[:50]}"
                )
                print(
                    f"    Value Type: {type(value)}, Value Value (first 50 chars): {str(value)[:50]}"
                )

                if isinstance(value, dict):
                    print("      Nested Dict Items (first 3):")
                    for j, (nested_key, nested_value) in enumerate(value.items()):
                        if j >= 3:
                            print("      ...")
                            break
                        print(
                            f"        Nested Key Type: {type(nested_key)}, Value: {str(nested_key)[:50]}"
                        )
                        print(
                            f"        Nested Value Type: {type(nested_value)}, Value: {str(nested_value)[:50]}"
                        )
                elif isinstance(value, (list, tuple)):
                    print("      Nested List/Tuple Items (first 3):")
                    for j, item in enumerate(value):
                        if j >= 3:
                            print("      ...")
                            break
                        print(
                            f"        Item Type: {type(item)}, Value: {str(item)[:50]}"
                        )
        elif isinstance(data, (list, tuple)):
            print(f"  Number of items: {len(data)}")
            for i, item in enumerate(data):
                if i >= 5:
                    print("  ...")
                    break
                print(f"    Item Type: {type(item)}, Value: {str(item)[:50]}")

    except Exception as e:
        print(f"  Error loading or inspecting file: {e}")
    print("\n")


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "jieba_fast_dat")

    # finalseg models
    finalseg_path = os.path.join(base_path, "finalseg")
    inspect_model_file(os.path.join(finalseg_path, "prob_emit.p"))
    inspect_model_file(os.path.join(finalseg_path, "prob_start.p"))
    inspect_model_file(os.path.join(finalseg_path, "prob_trans.p"))

    # posseg models
    posseg_path = os.path.join(base_path, "posseg")
    inspect_model_file(os.path.join(posseg_path, "char_state_tab.p"))
    inspect_model_file(os.path.join(posseg_path, "prob_emit.p"))
    inspect_model_file(os.path.join(posseg_path, "prob_start.p"))
    inspect_model_file(os.path.join(posseg_path, "prob_trans.p"))
