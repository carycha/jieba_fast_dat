# -*- coding: utf-8 -*-
import os
import sys

# New helper function to get the path string
def get_module_res_path(*res):
    try:
        import importlib.resources
        # For Python 3.7+, return the path as a string
        return str(importlib.resources.files(__name__).joinpath(*res))
    except (ImportError, AttributeError):
        # Fallback for older Python versions, already returns a string path
        return os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), *res))

# Original get_module_res to return an opened file object (for compatibility with existing code that expects it)
try:
    import importlib.resources
    get_module_res = lambda *res: importlib.resources.files(__name__).joinpath(*res).open('rb')
except (ImportError, AttributeError):
        get_module_res = lambda *res: open(os.path.normpath(os.path.join(
                                os.getcwd(), os.path.dirname(__file__), *res)), 'rb')

default_encoding = sys.getfilesystemencoding()

text_type = str
string_types = (str,)
xrange = range

iterkeys = lambda d: iter(d.keys())
itervalues = lambda d: iter(d.values())
iteritems = lambda d: iter(d.items())

def strdecode(sentence):
    if isinstance(sentence, bytes):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence

def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)