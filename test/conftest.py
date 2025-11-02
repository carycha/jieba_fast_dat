import os
import pytest
import jieba_fast_dat

# Define paths to our test resources
TEST_DICTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_dicts")
TEST_TEXTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_texts")


@pytest.fixture(scope="session")
def dicts_dir():
    return TEST_DICTS_DIR


@pytest.fixture(scope="session")
def texts_dir():
    return TEST_TEXTS_DIR


@pytest.fixture(scope="session")
def dict_base_path():
    return os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")


@pytest.fixture(scope="session")
def user_dict_base_path():
    return os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")


@pytest.fixture(scope="session")
def dict_add_path():
    return os.path.join(TEST_DICTS_DIR, "test_dict_add.txt")


@pytest.fixture(scope="session")
def main_test_text_path():
    return os.path.join(TEST_TEXTS_DIR, "main_test_text.txt")


@pytest.fixture
def main_test_text(main_test_text_path):
    with open(main_test_text_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def idf_base_path():
    return os.path.join(TEST_DICTS_DIR, "text_idf_base.txt")


@pytest.fixture(scope="session")
def stop_words_path():
    return os.path.join(TEST_DICTS_DIR, "test_stop_words.txt")


@pytest.fixture(scope="module")
def tfidf_extractor(dict_base_path, idf_base_path):
    """
    Provides a TFIDF extractor instance initialized with custom dictionaries.
    """
    custom_tokenizer = jieba_fast_dat.Tokenizer(dictionary=dict_base_path)

    # The original test had a weird override to force HMM=False.
    # We will replicate that here inside the fixture to keep tests clean.
    original_cut = custom_tokenizer.cut

    def new_cut(sentence, cut_all=False, HMM=True):
        return original_cut(sentence, cut_all=cut_all, HMM=False)  # Force HMM=False

    custom_tokenizer.cut = new_cut

    extractor = jieba_fast_dat.analyse.TFIDF(idf_path=idf_base_path)
    extractor.tokenizer = custom_tokenizer
    return extractor


@pytest.fixture
def tokenizer_base(dict_base_path, user_dict_base_path):
    """
    Provides a clean, initialized tokenizer with custom dictionaries.
    """
    tokenizer = jieba_fast_dat.Tokenizer()
    tokenizer.set_dictionary(dict_base_path)
    tokenizer.load_userdict(user_dict_base_path)
    return tokenizer


@pytest.fixture
def pos_tokenizer(dict_base_path, user_dict_base_path):
    """
    Provides a correctly initialized posseg tokenizer.
    The key is to create a Tokenizer instance, load dicts into IT,
    and then pass THAT instance to the POSTokenizer.
    """
    # 1. Create a custom tokenizer instance.
    custom_tokenizer = jieba_fast_dat.Tokenizer()

    # 2. Load dictionaries directly into this instance.
    custom_tokenizer.set_dictionary(dict_base_path)
    custom_tokenizer.load_userdict(user_dict_base_path)

    # 3. Create the POSTokenizer from the fully configured tokenizer instance.
    pos_tokenizer = jieba_fast_dat.posseg.POSTokenizer(tokenizer=custom_tokenizer)
    return pos_tokenizer


@pytest.fixture(autouse=True)
def check_memory_leaks(request):
    """
    Pytest fixture to check for Python memory leaks after each test function.
    This uses the gc module to detect objects that are created during a test
    and are not garbage collected afterwards.
    """
    import gc

    gc.collect()
    gc.disable()  # Disable garbage collection during test to track new objects

    initial_objects = {id(obj) for obj in gc.get_objects()}

    yield  # Run the test function

    gc.enable()  # Re-enable garbage collection
    gc.collect()  # Force a collection after the test

    all_final_objects = gc.get_objects()
    leaked_object_ids = {id(obj) for obj in all_final_objects} - initial_objects

    leaked_objects = []
    for obj in all_final_objects:
        if id(obj) in leaked_object_ids:
            # Basic filtering for common internal/fixture objects
            if (
                obj is check_memory_leaks
                or obj is request  # The fixture function itself
                or (  # Pytest request fixture
                    hasattr(obj, "__module__")
                    and obj.__module__ is not None
                    and any(obj.__module__.startswith(p) for p in ("_pytest", "gc"))
                )
            ):  # Pytest/GC internals
                continue

            # Explicitly ignore jieba_fast_dat.Tokenizer and POSTokenizer instances and their bound methods
            if (
                isinstance(obj, jieba_fast_dat.Tokenizer)
                or isinstance(obj, jieba_fast_dat.posseg.POSTokenizer)
                or (
                    hasattr(obj, "__self__")
                    and (
                        isinstance(obj.__self__, jieba_fast_dat.Tokenizer)
                        or isinstance(obj.__self__, jieba_fast_dat.posseg.POSTokenizer)
                    )
                )
            ):
                continue

            # Additional filtering for common Python types that might linger
            obj_type_name = type(obj).__name__
            if obj_type_name in (
                "function",
                "wrapper",
                "dict",
                "list",
                "tuple",
                "set",
                "str",
            ):
                if (
                    (obj_type_name == "dict" and not obj)
                    or (obj_type_name == "list" and not obj)
                    or (obj_type_name == "tuple" and not obj)
                    or (obj_type_name == "set" and not obj)
                ):
                    continue  # Ignore empty Python collections

            # Consider objects that are part of the test module or related modules
            # This helps to focus on application-level leaks
            if (
                hasattr(obj, "__module__")
                and obj.__module__ is not None
                and obj.__module__.startswith("jieba_fast_dat")
            ) or (
                hasattr(obj, "__file__")
                and obj.__file__ is not None
                and "test/" in obj.__file__
                and not obj.__file__.endswith("conftest.py")
            ):
                leaked_objects.append(obj)

            # If it's a pybind11-related object, include it
            elif (
                hasattr(obj, "__module__")
                and obj.__module__ is not None
                and "_jieba_fast_dat_functions_py3" in obj.__module__
            ):
                leaked_objects.append(obj)

    if leaked_objects:
        leak_details = []
        for obj in leaked_objects:
            obj_type = type(obj).__name__
            obj_repr = repr(obj)
            if len(obj_repr) > 100:
                obj_repr = obj_repr[:97] + "..."
            leak_details.append(f"  - Type: {obj_type}, Value: {obj_repr}")

        pytest.fail(
            f"Memory leak detected! {len(leaked_objects)} objects were not garbage collected:\n"
            + "\n".join(leak_details)
        )
