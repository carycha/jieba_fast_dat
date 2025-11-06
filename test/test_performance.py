import cProfile
import pstats
import pytest
import io
from pathlib import Path

# Assuming jieba_fast_dat is installed or available in the path
import jieba_fast_dat as jieba
import jieba_fast_dat.posseg as posseg

# Define paths
TEST_TEXT_PATH = Path(__file__).parent.parent / "extra_dict" / "profile_test"
CUT_PROF_PATH = Path("cut.prof")
POS_PROF_PATH = Path("pos.prof")
CUT_SUMMARY_PATH = Path("cut_summary.txt")
POS_SUMMARY_PATH = Path("pos_summary.txt")


@pytest.fixture(scope="module")
def test_text_content():
    """Fixture to load the test text content once per module."""
    if not TEST_TEXT_PATH.exists():
        pytest.fail(f"Test text file not found: {TEST_TEXT_PATH}")
    with open(TEST_TEXT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _run_profiling(func, *args, prof_path, summary_path, request):
    """Helper to run a function with cProfile, save results, and store a summary."""
    profiler = cProfile.Profile()
    profiler.enable()
    # Convert generator to list to ensure full execution during profiling
    list(func(*args))
    profiler.disable()
    profiler.dump_stats(prof_path)

    # Write pstats output to a file
    with open(summary_path, "w", encoding="utf-8") as f:
        stats = pstats.Stats(str(prof_path), stream=f)
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)

    # Capture pstats output to a string for internal use if needed
    s = io.StringIO()
    stats = pstats.Stats(str(prof_path), stream=s)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
    summary = s.getvalue()

    # Store the summary in the pytest config object
    if not hasattr(request.config, "_performance_summaries"):
        request.config._performance_summaries = []
    request.config._performance_summaries.append(
        f"\n--- Performance Summary for {prof_path.name} ---\n{summary}"
    )


def test_performance_jieba_cut(test_text_content, request, capsys):
    """Profiles jieba.cut performance."""
    _run_profiling(
        jieba.cut,
        test_text_content,
        prof_path=CUT_PROF_PATH,
        summary_path=CUT_SUMMARY_PATH,
        request=request,
    )
    assert CUT_PROF_PATH.exists()
    assert CUT_PROF_PATH.stat().st_size > 0  # Ensure file is not empty
    assert CUT_SUMMARY_PATH.exists()
    assert CUT_SUMMARY_PATH.stat().st_size > 0  # Ensure file is not empty


def test_performance_posseg_cut(test_text_content, request, capsys):
    """Profiles jieba.posseg.cut performance."""
    _run_profiling(
        posseg.cut,
        test_text_content,
        prof_path=POS_PROF_PATH,
        summary_path=POS_SUMMARY_PATH,
        request=request,
    )
    assert POS_PROF_PATH.exists()
    assert POS_PROF_PATH.stat().st_size > 0  # Ensure file is not empty
    assert POS_SUMMARY_PATH.exists()
    assert POS_SUMMARY_PATH.stat().st_size > 0  # Ensure file is not empty
