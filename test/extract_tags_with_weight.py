import sys
import logging
sys.path.append('../')

import jieba_fast_dat as jieba
import jieba_fast_dat.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags_with_weight.py [file name] -k [top k] -w [with weight=1 or 0]"

if __name__ == '__main__':
    parser = OptionParser(USAGE)
    parser.add_option("-k", dest="topK")
    parser.add_option("-w", dest="withWeight")
    opt, args = parser.parse_args()


    if len(args) < 1:
        logging.info(USAGE)
        sys.exit(1)

    file_name = args[0]

    if opt.topK is None:
        topK = 10
    else:
        topK = int(opt.topK)

    if opt.withWeight is None:
        withWeight = False
    else:
        if int(opt.withWeight) is 1:
            withWeight = True
        else:
            withWeight = False

    content = open(file_name, 'rb').read()

    tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=withWeight)

    if withWeight is True:
        for tag in tags:
            logging.info(f"tag: {tag[0]}\t\t weight: {tag[1]:.6f}")
    else:
        logging.info(",".join(tags))
