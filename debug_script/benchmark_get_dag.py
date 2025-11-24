import time

from jieba_fast_dat import Tokenizer


def benchmark_get_dag():
    tokenizer = Tokenizer()
    tokenizer.initialize()

    text = (
        "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作" * 100
    )

    start_time = time.time()
    for _ in range(100):
        tokenizer.get_DAG(text)
    end_time = time.time()

    print(f"Time taken for 100 iterations: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    benchmark_get_dag()
