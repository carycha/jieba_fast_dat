import os
import shutil
import pytest
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

from jieba_fast_dat.analyse import ChineseAnalyzer

def test_whoosh_indexing():
    analyzer = ChineseAnalyzer()

    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
    
    tmp_dir = "tmp_whoosh_test"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    ix = create_in(tmp_dir, schema)
    writer = ix.writer()

    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(_cur_dir, 'test.txt') # Assuming test.txt is in the same directory

    with open(file_name,"rb") as inf:
        i=0
        for line in inf:
            i+=1
            writer.add_document(
                title="line"+str(i),
                path="/a",
                content=line.decode('gbk','ignore')
            )
    writer.commit()

    searcher = ix.searcher()
    parser = QueryParser("content", schema=ix.schema)

    found_results = False
    for keyword in ("水果小姐","你","first","中文","交换机","交换"):
        q = parser.parse(keyword)
        results = searcher.search(q)
        if len(results) > 0:
            found_results = True
            break
    
    assert found_results, "No results found for any keyword"

    searcher.close()
    shutil.rmtree(tmp_dir)
