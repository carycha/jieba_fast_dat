import re
import time

import jieba_fast_dat


def test_regex_behavior():
    text = "a   b\r\nc"

    re_skip_old = re.compile(r"(\r\n|\s)", re.U)
    re_skip_new = re.compile(r"(\s+)", re.U)

    print(f"Text: {repr(text)}")

    split_old = re_skip_old.split(text)
    print(f"Old split: {split_old}")

    split_new = re_skip_new.split(text)
    print(f"New split: {split_new}")

    # Simulate the loop in cut
    def process(split_result: list[str]):
        res = []
        for x_idx, x in enumerate(split_result):
            if x_idx % 2 == 1:
                res.append(x)
            else:
                res.append(
                    x
                )  # In cut, it yields from chars; here we only need to see structure.
        return res

    # Actual jieba behavior check
    tokenizer = jieba_fast_dat.Tokenizer()
    print(f"Jieba cut: {list(tokenizer.cut(text))}")


def benchmark_regex():
    text = "word " * 1000
    re_skip_old = re.compile(r"(\r\n|\s)", re.U)
    re_skip_new = re.compile(r"(\s+)", re.U)

    start = time.time()
    for _ in range(1000):
        re_skip_old.split(text)
    print(f"Old regex time: {time.time() - start:.4f}s")

    start = time.time()
    for _ in range(1000):
        re_skip_new.split(text)
    print(f"New regex time: {time.time() - start:.4f}s")


if __name__ == "__main__":
    test_regex_behavior()
    benchmark_regex()
