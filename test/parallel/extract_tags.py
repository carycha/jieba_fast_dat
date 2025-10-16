import logging
import sys
sys.path.append('../../')

import jieba_fast_dat as jieba
jieba.enable_parallel(4)
import jieba_fast_dat.analyse
from optparse import OptionParser

USAGE ="usage:    python extract_tags.py [file name] -k [top k]"

parser = OptionParser(USAGE)
parser.add_option("-k",dest="topK")
opt, args = parser.parse_args()


if len(args) <1:
    logging.info(USAGE)
    sys.exit(1)

file_name = args[0]

if opt.topK==None:
    topK=10
else:
    topK = int(opt.topK)


content = open(file_name,'rb').read()

tags = jieba.analyse.extract_tags(content,topK=topK)

logging.info(",".join(tags))


