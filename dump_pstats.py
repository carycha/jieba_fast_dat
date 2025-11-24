import pstats
import sys


def dump_stats_to_file(prof_file: str, output_file: str) -> None:
    stats = pstats.Stats(prof_file)
    original_stdout = sys.stdout  # Store original stdout
    with open(output_file, "w") as f:
        sys.stdout = f  # Redirect stdout to the file
        try:
            stats.sort_stats("cumulative").print_stats()
        finally:
            sys.stdout = original_stdout  # Restore original stdout


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dump_pstats.py <input_prof_file> <output_txt_file>")
        sys.exit(1)

    input_prof_file = sys.argv[1]
    output_txt_file = sys.argv[2]
    dump_stats_to_file(input_prof_file, output_txt_file)
