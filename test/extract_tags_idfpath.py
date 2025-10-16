import sys
sys.path.append('../')
import logging

import jieba_fast_dat as jieba
import jieba_fast_dat.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags_idfpath.py [file name] -k [top k]"

if __name__ == '__main__':
    parser = OptionParser(USAGE)
    parser.add_option("-k", dest="topK")
    opt, args = parser.parse_args()


    if len(args) < 1:
        logging.info(USAGE)
        sys.exit(1)

    file_name = args[0]

    if opt.topK is None:
        topK = 10
    else:
        topK = int(opt.topK)

    content = open(file_name, 'rb').read()

    jieba.analyse.set_idf_path("../extra_dict/idf.txt.big");

    tags = jieba.analyse.extract_tags(content, topK=topK)

    logging.info(",".join(tags))
