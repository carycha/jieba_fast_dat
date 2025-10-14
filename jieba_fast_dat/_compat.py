# -*- coding: utf-8 -*-
import os
import sys

try:
    import pkg_resources
    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
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
