"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import re
from pylatexenc.latexencode import UnicodeToLatexEncoder
from pylatexenc.latex2text import LatexNodes2Text
from unidecode import unidecode

syn = [
    ("\\rbrack ", "] "),
    ("\\lbrack ", "[ "),
    ("\\lbrace ", "\\} "),
    ("\\rbrace ", "\\{ "),
    ("\\lnot ", "\\neg "),
    ("\\land ", "\\wedge "),
    ("\\vee ", "\\lor "),
    ("\\doublecup ", "\\Cup "),
    ("\\doublecap ", "\\Cap "),
    ("\\llless ", "\\lll "),
    ("\\gggtr ", "\\ggg "),
    ("\\doteqdot ", "\\Doteq "),
    ("\\ne ", "\\neq "),
    ("\\le ", "\\leq "),
    ("\\ge ", "\\geq "),
    ("\\leftarrow ", "\\gets "),
    ("\\rightarrow ", "\\to "),
    ("\\restriction ", "\\upharpoonright "),
    ("\\owns ", "\\ni "),
    ("\\textlnot ", "\\neg "),
    ("\\textellipsis ", "\\ldots "),
    ("\\textbullet ", "\\bullet "),
    ("\\plusmn ", "\\pm "),
    ("\\texttimes", "\\times"),
    ("\\textmu", "\\mu"),
    ("\\textendash", "-"),
    ("\\textemdash", "---"),
    ("\\>", "\\:"),
    ("\\medspace", "\\:"),
    ("\\thinspace", "\\,"),
    ("\\negthinspace", "\\!"),
    ("\\thickspace", "\\;"),
]
umlaut_mapping = {
    "textasciicircum": "^",
    "ddot": '"',
    "textasciidieresis": '"',
    "textasciicaron": "v ",
}
umlaut_keys = "|".join(umlaut_mapping.keys())
umlaut_regex = re.compile(rf"\s?\\({umlaut_keys})\s(\w)")
latex_comments = re.compile(r"(?<!\\)%.*\n")
toascii = UnicodeToLatexEncoder(
    non_ascii_only=True, unknown_char_policy="ignore", unknown_char_warning=False
)


def remove_style(string: str) -> str:
    return (
        string.replace("\\displaystyle", "")
        .replace("\\scriptstyle", "")
        .replace("\\scriptscriptstyle", "")
        .replace("\\textstyle", "")
    )


def replace_duplicate_definitions(string: str) -> str:
    """In Latex there are many commands that are interchangeable. Use just one of them"""
    for pair in syn:
        string = string.replace(pair[0], pair[1])
    return string


def unicode_to_latex(s: str) -> str:
    s = re.sub(
        r"\s{2,}",
        " ",
        re.sub(
            r"\\ensuremath\s?\{\s?(.+?)\s?\}\s?",
            r" \1 ",
            toascii.unicode_to_latex(s.strip()),
        )
        .replace("}", " ")
        .replace("{", " "),
    )
    s = (
        s.strip()
        .replace(
            "\\textperiodcentered \\textperiodcentered \\textperiodcentered", "\\cdots"
        )
        .replace("\\textperiodcentered", "\\cdot")
        .replace("\\textquoteright", "'")
        .replace("\\textquoteleft", "'")
        .replace("\\textquotedblleft", '"')
        .replace("\\textquotedblright", '"')
    )
    s = umlaut_regex.sub(lambda x: "\\" + umlaut_mapping[x.group(1)] + x.group(2), s)
    s = replace_duplicate_definitions(s)
    s = unidecode(s)
    return s.replace("\u2009", " ")


latex_to_unicode = LatexNodes2Text()


def remove_line_breaks(string: str) -> str:
    string = latex_comments.sub("\n", string)
    return string.replace("\n", " ")


def normalize_tex(math: str, inline: bool) -> str:
    """
    Normalize TeX math expressions.

    This function takes a TeX math expression and performs various normalization steps to ensure
    consistency and proper formatting.

    Args:
        math (str): The input TeX math expression.
        inline (bool): Indicates whether the expression should be inline (True) or displayed (False).

    Returns:
        str: The normalized TeX math expression.
    """
    math = math.strip()
    if not math:
        return ""
    if math.startswith(r"\(") or math.startswith(r"\[") or math.startswith("$$"):
        math = math[2:]
    elif math.startswith("$"):
        math = math[1:]
    if math.endswith(r"\)") or math.endswith(r"\]") or math.endswith("$$"):
        math = math[:-2]
    elif math.endswith("$"):
        math = math[:-1]
    math = math.strip()
    if not math:
        return ""
    math = remove_line_breaks(math.strip())
    math = replace_duplicate_definitions(math)
    math = remove_style(math)
    if inline:
        return rf"\({math}\)"
    return rf"\[{math}\]"
