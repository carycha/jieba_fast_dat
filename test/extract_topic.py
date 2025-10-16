import sys
sys.path.append("../")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition

import jieba_fast_dat as jieba
import time
import glob
import sys
import os
import random
import logging

if __name__ == '__main__':
    if len(sys.argv)<2:
        logging.info("usage: extract_topic.py directory [n_topic] [n_top_words]")
        sys.exit(0)

    n_topic = 10
    n_top_words = 25

    if len(sys.argv)>2:
        n_topic = int(sys.argv[2])

    if len(sys.argv)>3:
        n_top_words = int(sys.argv[3])

    count_vect = CountVectorizer()
    docs = []

    pattern = os.path.join(sys.argv[1],"*.txt") 
    logging.info("read "+pattern)

    for f_name in glob.glob(pattern):
        with open(f_name) as f:
            logging.info("read file:", f_name)
            for line in f: #one line as a document
                words = " ".join(jieba.cut(line))
                docs.append(words)

    random.shuffle(docs)

    logging.info("read done.")

    logging.info("transform")
    counts = count_vect.fit_transform(docs)
    tfidf = TfidfTransformer().fit_transform(counts)
    logging.info(tfidf.shape)


    t0 = time.time()
    logging.info("training...")

    nmf = decomposition.NMF(n_components=n_topic).fit(tfidf)
    logging.info("done in %0.3fs." % (time.time() - t0))

    # Inverse the vectorizer vocabulary to be able
    feature_names = count_vect.get_feature_names()

    for topic_idx, topic in enumerate(nmf.components_):
        logging.info("Topic #%d:" % topic_idx)
        logging.info(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        logging.info("")
