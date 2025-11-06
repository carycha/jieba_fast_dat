# import re
import jieba_fast_dat

import pickle
from jieba_fast_dat.utils import (
    get_module_res,
    _get_char_type,
    split_by_char_type,
    CHAR_TYPE_ZH,
    CHAR_TYPE_NUM,
    CHAR_TYPE_ALPHA,
    CHAR_TYPE_OTHER,
)

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"

# re_han_detail = re.compile(r"([\u4E00-\u9FD5]+)")
# re_skip_detail = re.compile(r"([\.0-9]+|[a-zA-Z0-9]+)")
# re_han_internal = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
# re_skip_internal = re.compile(r"(\r\n|\s)")

# re_eng = re.compile(r"[a-zA-Z0-9]+")
# re_num = re.compile(r"[\.0-9]+")

# re_eng1 = re.compile("^[a-zA-Z0-9]$", re.U)


def load_model():
    # For Jython
    start_p = pickle.load(get_module_res(__name__, PROB_START_P))
    trans_p = pickle.load(get_module_res(__name__, PROB_TRANS_P))
    emit_p = pickle.load(get_module_res(__name__, PROB_EMIT_P))
    state = pickle.load(get_module_res(__name__, CHAR_STATE_TAB_P))
    return state, start_p, trans_p, emit_p


char_state_tab_P, start_P, trans_P, emit_P = load_model()
jieba_fast_dat.load_hmm_model(start_P, trans_P, emit_P, char_state_tab_P)


class pair(object):
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __str__(self):
        return "%s/%s" % (self.word, self.flag)

    def __repr__(self):
        return "pair(%r, %r)" % (self.word, self.flag)

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return (
            isinstance(other, pair)
            and self.word == other.word
            and self.flag == other.flag
        )

    def __hash__(self):
        return hash(self.word)


class POSTokenizer(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or jieba_fast_dat.Tokenizer()
        self.load_word_tag(self.tokenizer.get_dict_file())

    def __repr__(self):
        return "<POSTokenizer tokenizer=%r>" % self.tokenizer

    def __getattr__(self, name):
        if name in ("cut_for_search", "lcut_for_search", "tokenize"):
            # may be possible?
            raise NotImplementedError
        return getattr(self.tokenizer, name)

    def initialize(self, dictionary=None):
        self.tokenizer.initialize(dictionary)
        self.load_word_tag(self.tokenizer.get_dict_file())

    def load_word_tag(self, f):
        self.word_tag_tab = {}
        f_name = f.name
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    "invalid POS dictionary entry in %s at Line %s: %s"
                    % (f_name, lineno, line)
                )
        f.close()

    def makesure_userdict_loaded(self):
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence):
        prob, pos_list = jieba_fast_dat._posseg_viterbi_cpp(sentence)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            if pos == "B":
                begin = i
            elif pos == "E":
                yield pair(sentence[begin : i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == "S":
                yield pair(char, pos_list[i][1])
                nexti = i + 1
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    def _split_non_chinese_block_efficient(self, text_block):
        n = len(text_block)
        if n == 0:
            return  # empty generator

        start = 0
        while start < n:
            current_char = text_block[start]
            current_char_code = ord(current_char)
            current_char_type_id = _get_char_type(current_char_code)

            block_end = start + 1
            block_tag = "x"  # Default tag

            # Determine the initial block type and tag
            if current_char_type_id == CHAR_TYPE_NUM or current_char == ".":
                block_tag = "m"
            elif current_char_type_id == CHAR_TYPE_ALPHA:
                block_tag = "eng"
            # else block_tag remains 'x'

            # Extend the block as long as the type matches the current block_tag logic
            # AND for 'eng' blocks, split if char type changes between ALPHA and NUM
            while block_end < n:
                next_char = text_block[block_end]
                next_char_code = ord(next_char)
                next_char_type_id = _get_char_type(next_char_code)

                should_extend = False
                if block_tag == "m":
                    if next_char_type_id == CHAR_TYPE_NUM or next_char == ".":
                        should_extend = True
                elif block_tag == "eng":
                    # 'eng' blocks can contain both alpha and numeric characters,
                    # but we need to split if type changes from ALPHA to NUM or vice-versa
                    if (
                        current_char_type_id == CHAR_TYPE_ALPHA
                        and next_char_type_id == CHAR_TYPE_NUM
                    ) or (
                        current_char_type_id == CHAR_TYPE_NUM
                        and next_char_type_id == CHAR_TYPE_ALPHA
                    ):
                        # Type changed within an 'eng' block, so break and yield current
                        break
                    elif (
                        next_char_type_id == CHAR_TYPE_ALPHA
                        or next_char_type_id == CHAR_TYPE_NUM
                    ):
                        should_extend = True
                # 'x' blocks are single characters, so no extension

                if should_extend:
                    block_end += 1
                else:
                    break

            yield text_block[start:block_end], block_tag
            start = block_end

    def __cut_detail(self, sentence):
        for blk_str, blk_type in split_by_char_type(sentence):
            if not blk_str:
                continue
            if blk_type == CHAR_TYPE_ZH:
                for word in self.__cut(blk_str):
                    yield word
            else:
                # For non-Chinese blocks, use the efficient custom splitter
                for sub_blk_str, sub_blk_tag in self._split_non_chinese_block_efficient(
                    blk_str
                ):
                    yield pair(sub_blk_str, sub_blk_tag)

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            first_char_type = _get_char_type(ord(l_word[0]))
            if (
                (first_char_type == CHAR_TYPE_ALPHA or first_char_type == CHAR_TYPE_NUM)
            ) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    if buf.isdigit():
                        yield pair(buf, "m")
                    else:
                        yield pair(buf, "eng")
                    buf = ""
                yield pair(l_word, self.word_tag_tab.get(l_word, "x"))
                x = y
        if buf:
            yield pair(buf, "eng")
            buf = ""

    def __cut_DAG(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}

        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ""
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1 or not self.tokenizer.get_freq(buf):
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, "x"))
                    buf = ""
                yield pair(l_word, self.word_tag_tab.get(l_word, "x"))
            x = y

        if buf:
            if len(buf) == 1 or not self.tokenizer.get_freq(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, "x"))

    def __cut_internal(self, sentence, HMM=True):
        self.makesure_userdict_loaded()
        sentence = sentence
        # Use split_by_char_type to get blocks with their types
        blocks_with_types = split_by_char_type(sentence)

        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        for blk_str, blk_type in blocks_with_types:
            if not blk_str:
                continue

            if blk_type == CHAR_TYPE_ZH:  # Chinese block
                for word in cut_blk(blk_str):
                    yield word
            elif blk_type == CHAR_TYPE_NUM:  # Numeric block
                yield pair(blk_str, "m")
            elif blk_type == CHAR_TYPE_ALPHA:  # Alpha block
                yield pair(blk_str, "eng")
            else:  # Other characters (punctuation, symbols)
                for c in blk_str:
                    yield pair(c, "x")

    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence):
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence, HMM=True):
        for w in self.__cut_internal(sentence, HMM=HMM):
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))


# default Tokenizer instance

dt = POSTokenizer(jieba_fast_dat.dt)

# global functions

initialize = dt.initialize


def _lcut_internal(s):
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)


def cut(sentence, HMM=True):
    """
    Global `cut` function that supports parallel processing.

    Note that this only works using dt, custom POSTokenizer
    instances are not supported.
    """
    global dt
    if jieba_fast_dat.pool is None:
        for w in dt.cut(sentence, HMM=HMM):
            yield w
    else:
        parts = sentence.splitlines(True)
        if HMM:
            result = jieba_fast_dat.pool.map(_lcut_internal, parts)
        else:
            result = jieba_fast_dat.pool.map(_lcut_internal_no_hmm, parts)
        for r in result:
            for w in r:
                yield w


def lcut(sentence, HMM=True):
    return list(cut(sentence, HMM))
