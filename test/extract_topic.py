from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition

import jieba_fast_dat

def test_extract_topic_basic():
    # Create some dummy documents for testing
    docs = [
        "今天 天气 真好 适合 出去 玩",
        "明天 天气 不好 可能 会 下雨",
        "今天 晚上 吃 什么 呢 火锅 烧烤",
        "火锅 烧烤 都 好吃 难以 选择"
    ]

    # Tokenize the documents using jieba_fast_dat.cut (simulated here)
    # In a real scenario, you'd cut raw text. Here, we use pre-tokenized for simplicity.
    processed_docs = [" ".join(jieba_fast_dat.cut(doc)) for doc in docs]

    # Use CountVectorizer
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(processed_docs)

    # Use TfidfTransformer
    tfidf = TfidfTransformer().fit_transform(counts)

    # Perform NMF (simplified)
    n_topic = 2
    nmf = decomposition.NMF(n_components=n_topic).fit(tfidf)

    # Assertions
    assert counts.shape[0] == len(docs)
    assert tfidf.shape[0] == len(docs)
    assert nmf.components_.shape[0] == n_topic
    assert isinstance(nmf.components_, (list, tuple, type(None))) or nmf.components_.ndim == 2 # Check if it's a 2D array or similar