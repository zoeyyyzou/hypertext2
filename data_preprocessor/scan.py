import ply.lex as lex
import sys

tokens = (
    # LABEL: the three common labels ($DOC, $TITLE, and $TEXT) used to indicate the
    # structure of a document.
    'LABEL',

    # WORD: strings of letters and digits separated by spaces, tabs, newlines, and most of the
    # punctuation marks. For example, “John”, “computer”, “mp3”, “123abc” are all treated
    # as WORD tokens.
    'WORD',

    # NUMBER: integers and real numbers, with possible positive and negative signs.
    'NUMBER',

    # APOSTROPHIZED: words like “John’s”, “O’Reily”, and “O’Reily’s” should be
    # treated as single tokens. However, “world’cup” and “this’is’just’a’models” are likely
    # to be typos and should be split further. Due to the longest possible match in JFlex,
    # however, it’s hard to separate these two cases in a scanner. As a result, you can treat
    # strings like “world’cup” and “this’is’just’a’models” as APOSTROPHIZED for now.
    'APOSTROPHIZED',

    # HYPHENATED: words such as “data-base” and “father-in-law” should be treated as
    # single tokens. However, “---” should be treated as a sequence of punctuation marks.
    # Note that when a hyphenated token is ended with an apostrophized suffix, it will be
    # classified as APOSTROPHIZED such as “father-in-law’s”. Again, due to the longest
    # possible match, you can write a general pattern so that strings like “this-is-just-atest”
    # can be treated as HYPHENATED for now.
    'HYPHENATED',

    # DELIMITERS: capture all sequences made of spaces, tabs, and newlines so that we can
    # filter them out from output.
    'DELIMITERS',

    # PUNCTUATION: Any symbol that does not contribute to the tokens above should be
    # treated as punctuation marks.
    'PUNCTUATION',
)


def t_APOSTROPHIZED(t):
    # For an APOSTROPHIZED token, it should only contain two or three parts when separated by
    # the apostrophes. If there are two parts, either the first part contains a single character and the
    # second part contains more than two characters, or the last part contains the character ‘s’. If there
    # are three parts, the first part should contain only one character and the last part contains the
    # character ‘s’. Otherwise, the token should be split into a sequence of tokens. For example,
    # “John’s”, “O’Reily”, and “O’Reily’s” will be treated as single tokens. When splitting an
    # APOSTROPHIZED token with two parts, we need to keep the apostrophe with the second part if
    # it has one or two characters; otherwise, add spaces on both sides of the apostrophe. For example,
    # we will split “You’re” and “I’ve” to “You ’re” and “I ’ve”, but “world’cup” will be split
    # into a sequence of three tokens “world ’ cup”. Note that the above two rules may be tested
    # together so that “father-in-law’s” is treated as one token, but “this-is-just-a-models’s”
    # should be split into a sequence of tokens “this - is - just - a - models ’ s”.
    r'\w+(-\w+)*(\'\w+)+'
    items = t.value.split("'")

    #  If there are two parts, either the first part contains a single character and the
    #  second part contains more than two characters, or the last part contains the character ‘s’.
    if len(items) == 2 and ((len(items[0]) == 1 and len(items[1]) > 2) or items[1].lower() == 's'):
        pass
    elif len(items) == 2 and len(items[1]) <= 2:
        t.value = " '".join(items)
    # If there are three parts, the first part should contain only one character and the last part
    # contains the character ‘s’
    elif len(items) == 3 and len(items[0]) == 1 and items[2].lower() == 's':
        pass
    else:
        t.value = " ' ".join(items)

    # deal contains -
    items = t.value.split("-")
    if len(items) > 3 or (len(items) == 3 and len(items[1]) > 2):
        t.value = ' - '.join(items)
        t.value = " ' ".join(t.value.split("'"))
    return t


def t_HYPHENATED(t):
    # For a HYPHENATED token, it should only contain two or three parts when separated by the
    # hyphens, and if there are three parts, the middle part can only have one or two characters.
    # Otherwise, the token should be split into a sequence of tokens. Such a rule will keep strings like
    # “data-base” and “father-in-law” as single tokens, but split strings like “this-is-just-atest”
    # into sequences of tokens such as “this - is - just - a - models”.
    r"""\w+(-\w+)+"""
    items = t.value.split("-")
    if len(items) > 3 or (len(items) == 3 and len(items[1]) > 2):
        t.value = ' - '.join(items)
    return t


def t_NUMBER(t): r'(\+|\-)?(\d+)?(\.)?\d+((e|E)(\+|\-)?\d)*'; return t


def t_WORD(t):
    r'\w+'
    return t


def t_DELIMITERS(t): r'\s+'; return t


def t_LABEL(t): r'\$(DOC|TITLE|TEXT)'; return t


def t_PUNCTUATION(t):
    r"""[^\d\s\w]"""
    # t.value = f" {t.value}"
    return t


# Error handling, always required
def t_error(t):
    print("\n===\n", t.value, "\n===")


# Build the lexer
lexer = lex.lex()


def getTokens(data) -> [lex.LexToken]:
    res = []
    lexer.input(data)
    while True:
        tok = lexer.token()
        if not tok:
            break
        res.append(tok)
    return res