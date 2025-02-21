"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Tuple, Union
import re
import numpy as np
from rapidfuzz.fuzz import ratio as ratio_perc
from fuzzysearch import find_near_matches

math_start_regex = re.compile(r"(?<!\\)\\[\[\(]", re.M)
math_end_regex = re.compile(r"(?<!\\)\\[\]\)]", re.M)


def ratio(*args, **kwargs):
    return ratio_perc(*args, **kwargs) / 100


def reverse(lst: List[str]) -> List[str]:
    """Reverses a list and the strings inside

    Args:
        lst (List[str]): List to process

    Returns:
        List[str]: Reversed list
    """
    out = lst[::-1]
    for i in range(len(out)):
        out[i] = out[i][::-1]
    return out


def get_first_last(
    s: str,
    num_words: int = 8,
    delim: str = " ",
    first_only: bool = False,
    last_only: bool = False,
) -> Union[Tuple[str, str], str]:
    """
    Get the first and last `num_words` from a string `s`.

    Args:
        s (str): The string.
        num_words (int): The number of words.
        delim (str): The delimiter between words.
        first_only (bool): Whether to only get the first `num_words`.
        last_only (bool): Whether to only get the last `num_words`.

    Returns:
        Union[Tuple[str, str], str]: The first and last `num_words` from `s`, or `s` if `num_words` is 0.
    """
    s = s.split(delim)
    if not first_only and not last_only:
        return delim.join(s[:num_words]), delim.join(s[-num_words:])
    elif first_only:
        return delim.join(s[:num_words])
    elif last_only:
        return delim.join(s[-num_words:])


def get_glob_index(
    lengths: List[int], ind: int, return_breakpoints: bool = False
) -> int:
    """returns the index where ind is closest and greater than the lengths"""
    breakpoints = np.cumsum(lengths)
    overlap = breakpoints - ind
    overlap[overlap > 0] = -int(1e5)
    indices = overlap.argmax(0)
    if return_breakpoints:
        return indices, breakpoints
    else:
        return indices


# table-header-figure regex
# thf_regex = re.compile(r"(\[(FOOTNOTE|FIGURE|TABLE).*?END\2\])")


class Splitter:
    _split_locs: List[Tuple[int, int]] = None

    def __init__(self, paragraphs: List[str]) -> None:
        self.paragraphs = paragraphs
        self.paragraphs_no_space = [self.remove_special_chars(h) for h in paragraphs]
        self._split_locs = [(0, 0)]
        self.paragraphs_rev = reverse(self.paragraphs)
        self.paragraphs_rev_no_space = reverse(self.paragraphs_no_space)

    @staticmethod
    def remove_special_chars(string: str) -> str:
        # string = thf_regex.sub(r"", string)
        return (
            string.replace("\\ ", "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("*", "")
            .replace("_", "")
            .replace("^", "")
            .replace("\\[", "")
            .replace("\\]", "")
            .replace("\\(", "")
            .replace("\\)", "")
            .replace("\\right", "")
            .replace("\\left", "")
            .replace("\\sum", "X")  # old latex unicode encoding issue
            .replace("{", "")
            .replace("}", "")
            .replace("#", "")
            .replace("[REF]", "")
            .replace("[ENDREF]", "")
            .replace("\\varphi", "\\phi")  # https://meta.stackexchange.com/a/349360
            .replace("\\quad", "")
            .replace("\\qquad", "")
            .replace("\\hskip", "")
            .replace("\\vskip", "")
            .replace("\\frac", "")
            .replace("\\rm", "")
            .replace("\\,", "")
            .replace("-", "")
            .lower()
        )

    @staticmethod
    def count_special_chars(string: str, char_ind: int) -> int:
        if len(string) == 0:
            return 0
        add_space_ind = 0
        while True:
            string_ = string[: char_ind + add_space_ind]
            # last_first = string[: char_ind + add_space_ind+]
            add = (
                string_.count(" ")
                + string_.count("\\ ") * 2
                + string_.count("\n")
                + string_.count("*")
                + string_.count("_")
                + string_.count("^")
                + string_.count("\\[") * 2
                + string_.count("\\]") * 2
                + string_.count("\\(") * 2
                + string_.count("\\)") * 2
                + string_.count("\\right") * 6
                + string_.count("\\left") * 5
                + string_.count("\\sum") * 3  # replaced to X that's why not 4
                + string_.count("{")
                + string_.count("}")
                + string_.count("#")
                + string_.count("[REF]") * 5
                + string_.count("[ENDREF]") * 8
                + string_.count("\\varphi") * 3
                + string_.count("\\quad") * 5
                + string_.count("\\qquad") * 6
                + string_.count("\\hskip") * 6
                + string_.count("\\vskip") * 6
                + string_.count("\\frac") * 5
                + string_.count("\\rm") * 3
                + string_.count("\\,") * 2
                + string_.count("-")
            )
            if add == add_space_ind:
                break
            add_space_ind = add
        if len(string) <= char_ind + add_space_ind:
            add_space_ind = max(0, len(string) - 1 - char_ind)

        # check first chars of rest if they match closing expressions
        while True:
            rest = string[char_ind + add_space_ind :]
            string_ = string[: char_ind + add_space_ind]
            section_title = re.match(r"#+\s?\d*\s*$", string_)
            if rest.startswith("\\]") or rest.startswith("\\)"):
                add_space_ind += 2
            elif (rest.startswith(")") or rest.startswith("]")) and string_.endswith(
                "\\"
            ):
                add_space_ind += 1
            elif (rest.startswith("(") or rest.startswith("[")) and string_.endswith(
                "\\"
            ):
                add_space_ind -= 1
            elif rest.startswith(" "):
                add_space_ind += 1
            elif section_title:
                add_space_ind -= section_title.end() - section_title.start()
            elif (
                re.match(r"^[^\w\s]*_\s", rest)
                or re.match(r"^[^\w\s]*\*\*?\s", rest)
                or re.match(r"^.\n", rest)
            ):
                add_space_ind += 1
            else:
                break
        # check if it starts in a math env and include everything before
        end = math_end_regex.search(rest)
        if end is not None:
            start = math_start_regex.search(rest)
            if start is None or start.start() > end.start():
                inds = [
                    m.start()
                    for m in math_start_regex.finditer(string_)
                    if m.start() < end.start() + len(string_)
                ]
                if len(inds) > 0:
                    add_space_ind = inds[-1] - char_ind
                    # assert string_[char_ind+add_space_ind]=='\\'
        return add_space_ind

    def split_first_last(
        self, index: int, first: str, last: str, delta: int = 5
    ) -> Tuple[int, int, float]:
        """Refines a split by looking at both the first words from a new page and the last words from the previous page.

        Args:
            index (int): paragraph index
            first (str): first words
            last (str): last words
            delta (int, optional): paragraph search radius. Defaults to 5.

        Returns:
            Tuple[int, int, float]: split prediction
        """
        if first:
            first_split = glob_f, char_f, score_f = self.split(
                index, first, delta=delta
            )
        if last:
            last_split = glob_l, char_l, score_l = self.split(
                index, last, delta=delta, reverse=True
            )
        if first and not last:
            return first_split
        elif not first and last:
            return last_split
        elif not first and not last:
            return index, 0, 0.0
        if char_f == char_l and glob_f == glob_l and (score_f > 0.5 or score_l > 0.5):
            return glob_l, char_l, 1.0

        # score calculation
        first, last = self.remove_special_chars(first), self.remove_special_chars(last)
        matching = []
        for split in (first_split, last_split):
            first_source = []
            num_chars_first = len(first)
            num_chars_last = len(last)
            last_source = []
            for i, p in enumerate(self.paragraphs[split[0] :]):
                if i == 0:
                    p = p[split[1] :]
                first_source.append(self.remove_special_chars(p))
                if sum([len(s) for s in first_source]) >= num_chars_first:
                    break
            first_source = "".join(first_source)[:num_chars_first]
            for i, p in enumerate(self.paragraphs[split[0] :: -1]):
                if i == 0:
                    p = p[: split[1]]
                last_source.insert(0, self.remove_special_chars(p))
                if sum([len(s) for s in last_source]) >= num_chars_last:
                    last_source = last_source
                    break
            last_source = "".join(last_source)[-num_chars_last:]
            matching.append(
                [
                    ratio(first, first_source) * ratio(first[:10], first_source[:10]),
                    ratio(last, last_source) * ratio(last[-10:], last_source[-10:]),
                ]
            )
        scores = np.asarray(matching).max(0)
        return (
            (glob_l, char_l, scores[1])
            if scores.argmax()
            else (glob_f, char_f, scores[0])
        )

    def split(
        self, index: int, string: str, delta: int = 5, reverse: bool = False
    ) -> Tuple[int, int, float]:
        """
        refine split prediction. `string` are the first words from new page.
        delta can be used as uncertainty measure.
        returns new index and split index
        """
        if reverse:
            index = len(self.paragraphs) - 1 - index
            string = string[::-1]
            paragraphs = self.paragraphs_rev
            paragraphs_no_space = self.paragraphs_rev_no_space
        else:
            paragraphs = self.paragraphs
            paragraphs_no_space = self.paragraphs_no_space

        string_ = self.remove_special_chars(string)
        start_ind = max(0, index - delta)
        search_corpus = paragraphs_no_space[start_ind : index + delta + 1]
        lengths = np.asarray([0] + [len(p) for p in search_corpus])
        corp = "".join(search_corpus)
        if len(corp) == 0:
            self._split_locs.append((index, 0))
            return index, 0, 1
        ind, score = self._find_match(corp, string_)
        indices, breakpoints = get_glob_index(lengths, ind, True)
        global_ind, char_ind = int(start_ind + indices), int(ind - breakpoints[indices])
        self._split_locs.append((global_ind, char_ind))
        if reverse:
            char_ind = len(paragraphs_no_space[global_ind]) - char_ind
            global_ind = len(paragraphs) - global_ind - 1
        add_space_ind = self.count_special_chars(self.paragraphs[global_ind], char_ind)
        return global_ind, char_ind + add_space_ind, score

    def _find_match(
        self, corp: str, key: str, get_start: bool = True
    ) -> Tuple[int, float]:
        block, score = self._fuzzy(corp, key)
        index = max(0, block[0])
        if not get_start:
            index += block[2]
        return index, score

    @staticmethod
    def _fuzzy(
        corpus: str, string: str, max_error_rate: float = 0.025
    ) -> Tuple[Tuple[int, int, int], float]:
        max_dist = min(len(string) - 1, int(len(string) * min(0.9, max_error_rate)) + 5)
        matches = find_near_matches(string, corpus, max_l_dist=max_dist)
        if len(matches) > 0 and max_dist > 0:
            match = min(matches, key=lambda x: x.dist)
            block = (match.start, 0, match.end - match.start)
            score = 1 - match.dist / max_dist
            return block, score
        return (0, 0, 0), 0

    @staticmethod
    def fuzzysearch(
        corpus: str, string: str, max_error_rate: float = 0.025
    ) -> Tuple[Tuple[int, int, int], float]:
        corpus_ = Splitter.remove_special_chars(corpus)
        string_ = Splitter.remove_special_chars(string)
        (start, _, dist), score = Splitter._fuzzy(
            corpus_, string_, max_error_rate=max_error_rate
        )
        end = Splitter.count_special_chars(corpus, start + dist) + start + dist
        start = start + Splitter.count_special_chars(corpus, start)
        return (start, _, end - start), score

    def evaluate_split(self, page_num: int, page_content: str) -> float:
        if page_num > len(self._split_locs) or page_num < 1:
            return 0
        page_content = self.remove_special_chars(page_content)
        if page_num == len(self._split_locs):
            start, end = self._split_locs[-1], (-1, -1)
        else:
            start, end = self._split_locs[page_num - 1], self._split_locs[page_num]
        if (end[0] + 1) - start[0] < 0:
            return 0
        doc_content = self.paragraphs_no_space[start[0] : (end[0] + 1) or None]
        if (
            len(doc_content) < 1
            or len(doc_content[0]) < start[1]
            or len(doc_content[-1]) < end[1]
        ):
            return 0
        doc_content[0] = doc_content[0][start[1] :]
        doc_content[-1] = doc_content[-1][: end[1]]
        doc_content = "".join(doc_content)
        match = ratio(page_content, doc_content)
        return match
