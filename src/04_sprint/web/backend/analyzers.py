"""
Custom Whoosh analyzers for Chinese text segmentation using jieba.
Separated into its own module so that Whoosh's schema pickling works
correctly with uvicorn's --reload (multiprocessing spawn).
"""

import jieba
from whoosh.analysis import Tokenizer, Token, LowercaseFilter, StopFilter


class JiebaTokenizer(Tokenizer):
    """Custom Whoosh tokenizer using jieba for Chinese text segmentation."""

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, tokenize=True,
                 mode="", **kwargs):
        t = Token(positions, chars, removestops=removestops)
        pos = start_pos
        for word in jieba.cut(value, cut_all=False):
            word = word.strip()
            if not word:
                continue
            t.original = t.text = word
            t.pos = pos
            t.startchar = value.find(word, start_char)
            t.endchar = t.startchar + len(word)
            start_char = t.endchar
            pos += 1
            yield t


def ChineseAnalyzer():
    return JiebaTokenizer() | LowercaseFilter() | StopFilter()
