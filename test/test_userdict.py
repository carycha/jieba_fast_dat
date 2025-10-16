#encoding=utf-8
import sys
import os
import pytest
sys.path.append("../")
import jieba_fast_dat as jieba
import jieba_fast_dat.posseg as pseg
import logging

@pytest.fixture(scope="module")
def userdict_file(tmp_path_factory):
    # Create a temporary userdict.txt for testing
    userdict_path = tmp_path_factory.mktemp("userdict_dir") / "userdict.txt"
    with open(userdict_path, "w", encoding="utf-8") as f:
        f.write("自定义词 100 n\n")
        f.write("石墨烯 50 n\n")
        f.write("easy_install 100 eng\n")
    return userdict_path

def test_userdict_loading(userdict_file):
    jieba.load_userdict(str(userdict_file))
    
    test_sent = "这是一个自定义词和石墨烯的测试。"
    words = jieba.lcut(test_sent)
    assert "自定义词" in words
    assert "石墨烯" in words

def test_userdict_modification_and_reload(userdict_file):
    # Initial load
    jieba.load_userdict(str(userdict_file))
    test_sent_initial = "这是一个自定义词的测试。"
    words_initial = jieba.lcut(test_sent_initial)
    assert "自定义词" in words_initial
    assert "新词" not in words_initial

    # Modify userdict.txt and reload
    with open(userdict_file, "a", encoding="utf-8") as f:
        f.write("新词 100 n\n")
    
    jieba.load_userdict(str(userdict_file))
    test_sent_modified = "这是一个新词的测试。"
    words_modified = jieba.lcut(test_sent_modified)
    assert "新词" in words_modified

# Original test_sent and related logging.infos (adapted)
test_sent_original = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)

def test_original_cut():
    words = jieba.cut(test_sent_original)
    # logging.info('/'.join(words)) # For manual inspection
    assert len(list(words)) > 0 # Just ensure it cuts something

def test_original_posseg():
    result = pseg.cut(test_sent_original)
    # for w in result:
    #     logging.info(w.word, "/", w.flag, ", ', end=' ') # For manual inspection
    assert len(list(result)) > 0 # Just ensure it processes something

def test_english_and_regex_cut():
    terms = jieba.cut('easy_install is great')
    # logging.info('/'.join(terms)) # For manual inspection
    assert "easy_install" in list(terms)
    terms = jieba.cut('python 的正则表达式是好用的')
    # logging.info('/'.join(terms)) # For manual inspection
    assert "python" in list(terms)