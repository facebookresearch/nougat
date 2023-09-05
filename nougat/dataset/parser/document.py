"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import defaultdict
from copy import copy
import itertools
import re
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    List,
    Dict,
    Optional,
    TypeVar,
    Type,
    Generic,
)
import numpy as np

import logging

logger = logging.getLogger()

from dataclasses import dataclass, field, asdict
from typing import List, Dict, TypeVar, Type, Generic

T = TypeVar("T")
EL = TypeVar("EL")


@dataclass
class Element(Generic[EL]):
    """
    Generic class representing an element with children in a tree-like structure.

    Attributes:
        parent (Element): The parent element.
        children (List[Element]): List of child elements.
    """

    parent: "Element" = None
    children: List["Element"] = field(default_factory=list)

    @property
    def plaintext(self):
        return "".join([child.plaintext for child in self.children])

    def append(self, child: EL) -> EL:
        self.children.append(child)
        child.parent = self
        return child

    def find_parent(self, class_or_tuple: Type[T]) -> T:
        elem = self
        while elem:
            if isinstance(elem, class_or_tuple):
                return elem
            elem = elem.parent
        return None


@dataclass
class UnknownElement(Element):
    pass


@dataclass
class TextElement(Element):
    content: str = ""

    @property
    def plaintext(self):
        return self.content

    def append(self, child: "Element"):
        raise Exception(f"Cannot append elements to {self.__class__.__name__}")


@dataclass
class Math(Element):
    pass


@dataclass
class PlaintextMath(Math):
    pass


@dataclass
class LatexMath(Math):
    inline: bool = True
    code: str = ""

    @property
    def plaintext(self):
        return self.code


@dataclass
class Author:
    fullname: str = None
    lastname: str = None
    affiliation: str = None


@dataclass
class Link(Element):
    target: str = None


@dataclass
class InlineRef(Element):
    target: str = None

    def as_dict(self):
        return {
            "target": self.target,
        }


@dataclass
class Reference:
    """
    Data class representing a reference with various attributes.

    Attributes:
        title (Element): The title of the reference.
        authors (List[Author]): List of authors of the reference.
        ids (Dict[str, str]): Dictionary of identification information.
        date (str): The publication date of the reference.
        url (str): The URL link to the reference.
        journal (str): The journal where the reference is published.
        full_text (str): The full text content of the reference.

    Methods:
        as_dict(): Convert the reference object to a dictionary.
    """

    title: Element = None
    authors: List[Author] = field(default_factory=list)
    ids: Dict[str, str] = field(default_factory=dict)
    date: str = None
    url: str = None
    journal: str = None
    full_text: str = None

    def as_dict(self):
        return {
            "title": self.title.plaintext,
            "authors": [asdict(auth) for auth in self.authors],
            "ids": self.ids,
            "date": self.date,
            "url": self.url,
            "journal": self.journal,
            "full_text": self.full_text,
        }


@dataclass
class SpanElement(Element):
    pass


@dataclass
class Italic(SpanElement):
    pass


@dataclass
class Bold(SpanElement):
    pass


@dataclass
class Superscript(SpanElement):
    pass


@dataclass
class Subscript(SpanElement):
    pass


@dataclass
class Paragraph(Element):
    pass


@dataclass
class TableRow(Element):
    cells: List[Element] = field(default_factory=list)

    def add_cell(self, cell: Element):
        self.cells.append(cell)
        cell.parent = self
        return cell

    @property
    def plaintext(self):
        return "\t".join([cell.plaintext for cell in self.cells])


@dataclass
class TableHead(TableRow):
    pass


@dataclass
class Table(Element):
    id: str = None
    header: Element = None
    caption: Element = None
    rows: List[TableRow] = field(default_factory=list)
    keep_table: bool = False

    def add_row(self, row: TableRow) -> TableRow:
        self.rows.append(row)
        row.parent = self
        return row

    @property
    def plaintext(self):
        return "\n".join([row.plaintext for row in self.rows])


@dataclass
class Equation(Element):
    pass


@dataclass
class EquationList(Element):
    equations: List[Equation] = field(default_factory=list)

    def add_equation(self, eqn: Equation) -> Equation:
        self.equations.append(eqn)
        eqn.parent = self
        return eqn

    @property
    def plaintext(self):
        return "\n".join([eqn.plaintext for eqn in self.equations])


@dataclass
class Algorithm(Element):
    caption: Element = None
    lines: List[Element] = field(default_factory=list)
    inline: bool = False

    def add_line(self, line: Element) -> Element:
        self.lines.append(line)
        line.parent = self
        return line

    @property
    def plaintext(self):
        return "\n".join([line.plaintext for line in self.lines])


@dataclass
class Definition(Element):
    term: Element = None
    definition: Element = None

    @property
    def plaintext(self):
        parts = []
        if self.term:
            parts.append(f"{self.term.plaintext}:")
        if self.definition:
            parts.append(self.definition.plaintext)
        return " ".join(parts)


@dataclass
class DefinitionList(Element):
    """
    Data class representing a list of definitions with an optional header.

    Attributes:
        header (Element): The header element for the definition list.
        items (List[Definition]): List of Definition elements.

    Methods:
        add_item(item: Definition) -> Definition: Add a definition item to the list.
    """

    header: Element = None
    items: List[Element] = field(default_factory=list)

    def add_item(self, item: Definition) -> Definition:
        self.items.append(item)
        item.parent = self
        return item

    @property
    def plaintext(self):
        parts = []
        if self.header:
            parts.append(self.header.plaintext)
        parts.extend([df.plaintext for df in self.items])
        return "\n".join(parts)


@dataclass
class Figure(Element):
    id: str = None
    header: Element = None
    caption: Element = None


@dataclass
class Section(Element):
    id: str = None
    header: Element = None
    level: int = 0
    hnum: int = 1


@dataclass
class SectionHeader(Element):
    id: str = None
    header: Element = None
    level: int = 0


@dataclass
class ListItem(Element):
    label: str = ""


@dataclass
class ListContainer(Element):
    level: int = 0
    ordered: bool = False
    items: List[Element] = field(default_factory=list)

    def add_item(self, item: ListItem) -> ListItem:
        self.items.append(item)
        item.parent = self
        return item

    @property
    def plaintext(self):
        return "\n".join([item.plaintext for item in self.items])


@dataclass
class Footnote(Element):
    id: str = None


@dataclass
class Document(Element, Reference):
    abstract: Element = None
    language: str = None
    keywords: List[Element] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    inline_refs: List[InlineRef] = field(default_factory=list)
    bib: Reference = None

    def add_reference(self, reference):
        self.references.append(reference)

    def add_inline_ref(self, in_ref):
        self.inline_refs.append(in_ref)

    def set_bib(self, reference):
        self.bib = reference


@dataclass
class Spec:
    """
    Data class representing specifications for table cells.

    Attributes:
        t (int): The top border size.
        b (int): The bottom border size.
        l (int): The left border size.
        r (int): The right border size.
        align (str): The alignment of the cell content ('c' for center, 'l' for left, 'r' for right,
                     or 'p{width}' for justified with a specified width).

    Methods:
        __hash__() -> int: Compute the hash of the specification.
        __eq__(__o: object) -> bool: Check if two specifications are equal.
        set_align(classes: List[str], style: Optional[str] = None) -> None:
            Extract alignment information from HTML classes.
        set_border(classes: List[str]) -> None: Automatically set border specifications.
        set_attrs(attrs: Dict[str, Any]) -> None: Automatically set all attributes from HTML class attributes.
        __str__() -> str: Get the string representation of the specification.
    """

    t: int = field(default=0, repr=False)
    b: int = field(default=0, repr=False)
    l: int = field(default=0)
    r: int = field(default=0)
    align: str = field(default="")

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, __o: object) -> bool:
        return repr(self) == repr(__o)

    def set_align(self, classes: List[str], style: Optional[str] = None) -> None:
        """extract alignment information from available classes (html)"""
        aligns = [s for s in classes if "align" in s]
        if len(aligns) == 0:
            return
        elif len(aligns) > 1:
            logger.warn("Found multiple aligns in classes: %s", ", ".join(classes))
        align = aligns[0]
        if "center" in align or align == "c":
            self.align = "c"
        elif "left" in align or align == "l":
            self.align = "l"
        elif "right" in align or align == "r":
            self.align = "r"
        elif "justify" in align or align == "p":
            # assert style is not None, "justify without style information"
            if style is None:
                self.align = "c"
            else:
                width = style.partition("width:")[2].partition(";")[0]
                self.align = "p{%s}" % width
        else:
            logger.warn(
                "only center, left, right, justify supported at the moment. Found %s",
                align,
            )
            self.align = "c"

    def set_border(self, classes: List[str]) -> None:
        """automatically set spec with border classes e.g 'ltx_border_t'"""
        for border in classes:
            orientation = border.partition("border_")[2]
            if len(orientation) > 0 and orientation[0] in "tbrl":
                setattr(self, orientation[0], len(orientation))

    def set_attrs(self, attrs: Dict[str, Any]) -> None:
        """automatically set all attr from html class attributes"""
        classes = attrs["class"]
        style = attrs["style"] if "style" in attrs else None

        self.set_align(classes, style=style)
        self.set_border(classes)

    def __str__(self) -> str:
        if self.align:
            return "|" * self.l + self.align + "|" * self.r
        else:
            # default center
            return "|" * self.l + "c" + "|" * self.r


@dataclass
class TableCell(Element):
    """
    Represents a cell in an HTML table.

    Attributes:
        multicolumn (Optional[int]): The number of columns spanned by the cell.
        multirow (Optional[int]): The number of rows spanned by the cell.
        spec (Spec): The specification for the cell's formatting.
        content (Element): The content of the cell.

    Methods:
        __post_init__(*args, **kwargs) -> None: Initialize the cell, ensuring that the spec property is not None.
        __hash__() -> int: Compute the hash of the cell.
        __eq__(__o: object) -> bool: Check if two cells are equal.
        set_attrs(attrs: Dict[str, Any]) -> None: Set attributes for the cell from HTML attributes.
        plaintext() -> str: Get the plaintext content of the cell.
    """

    multicolumn: Optional[int] = None
    multirow: Optional[int] = None
    spec: Spec = None
    content: Element = None

    def __post_init__(self, *args, **kwargs) -> None:
        # spec property cannot be None
        if self.spec is None:
            self.spec = Spec()

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, __o: object) -> bool:
        return repr(self) == repr(__o)

    def set_attrs(self, attrs: Dict[str, Any]) -> None:
        if "colspan" in attrs:
            self.multicolumn = int(attrs["colspan"])
        if "rowspan" in attrs:
            self.multirow = int(attrs["rowspan"])
        self.spec.set_attrs(attrs)

    @property
    def plaintext(self):
        if self.content is None:
            return ""
        return self.content.plaintext


@dataclass
class TableRow(Element):
    """
    Represents a row in an HTML table.

    Attributes:
        cells (List[TableCell]): The list of cells in the row.

    Methods:
        add_cell(cell: TableCell) -> TableCell: Add a cell to the row.
        __iter__() -> Iterator: Iterate through the cells in the row.
        __len__() -> int: Get the number of cells in the row.
        __bool__() -> bool: Check if the row is not empty.
        cum_cell_widths() -> List[int]: Get the cumulative cell widths.
        cell_widths() -> List[int]: Get the widths of individual cells.
        width() -> int: Get the total width of the row.
        _hline(orientation: str) -> str: Determine horizontal lines to be inserted.
        hline_above() -> str: Get the horizontal line description for the top of the row.
        hline_below() -> str: Get the horizontal line description for the bottom of the row.
        plaintext() -> str: Get the plaintext content of the row.
    """

    cells: List[TableCell] = field(default_factory=list)

    def add_cell(self, cell: TableCell):
        self.cells.append(cell)
        cell.parent = self
        return cell

    def __iter__(self):
        return iter(self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    def __bool__(self) -> bool:
        return True

    @property
    def cum_cell_widths(self) -> List[int]:
        return np.cumsum(self.cell_widths)

    @property
    def cell_widths(self) -> List[int]:
        return [(cell.multicolumn or 1) for cell in self.cells]

    @property
    def width(self) -> int:
        return sum(self.cell_widths)

    def _hline(self, orientation: str) -> str:
        """Figure out if and where horizontal lines need to be inserted.

        Args:
            orientation (str): Either 't' (top) or 'b' (bottom)

        Returns:
            str: Correct vertical line description for latex tables.
        """
        assert orientation == "t" or orientation == "b"
        lines = []
        for cell in self.cells:
            lines.extend([getattr(cell.spec, orientation)] * (cell.multicolumn or 1))
        lines.append(0)
        indices = []
        start = None
        for i, v in enumerate(lines):
            if v and start is None:
                start = i
            elif start is not None and not v:
                indices.append((start, i - 1))
                start = None
        s = ""
        for a, b in indices:
            if b - a + 1 == self.width:
                s += "\\hline " * lines[0]
            else:
                s += "\\cline{%i-%i} " % (a + 1, b + 1)
        return s.strip()

    @property
    def hline_above(self) -> str:
        return self._hline("t")

    @property
    def hline_below(self) -> str:
        return self._hline("b")

    @property
    def plaintext(self) -> str:
        return "\t".join([cell.plaintext for cell in self.cells])


@dataclass
class Tabular(Element):
    rows: List[TableRow] = field(default_factory=list)
    """
    Represents a tabular structure, such as an HTML table.

    Attributes:
        rows (List[TableRow]): The list of rows in the tabular structure.

    Methods:
        add_row(row: TableRow) -> TableRow: Add a row to the tabular structure.
        width() -> int: Get the maximum width of the tabular structure.
        cols() -> List[List[TableCell]]: Get a list of columns in the tabular structure.
        _square_table() -> None: Ensure the table has an equal number of columns in each row.
        get_table_spec() -> str: Generate a LaTeX table specification based on cell alignments.
        plaintext() -> str: Get the plaintext content of the tabular structure.
    """

    def add_row(self, row: TableRow) -> TableRow:
        self.rows.append(row)
        row.parent = self
        return row

    @property
    def width(self) -> int:
        if len(self.rows) > 0:
            return max([r.width for r in self.rows])
        else:
            return 0

    @property
    def cols(self) -> List[List[TableCell]]:
        return list(
            map(
                list,
                itertools.zip_longest(*[r.cells for r in self.rows], fillvalue=None),
            )
        )

    def _square_table(self) -> None:
        """check if number of columns is equal for every row. Add placeholders for `\multirow` instances"""
        for i, row in enumerate(self.rows):
            for j, cell in enumerate(row.cells):
                if cell.multirow is not None and cell.multirow > 1:
                    spec = copy(cell.spec)
                    # assume no hlines in multi cells: disable bottom lines for top and top lines for lower cells.
                    spec.t = 0
                    cell.spec.b = 0
                    for k in range(i + 1, i + cell.multirow):
                        if k < len(self.rows):
                            for _ in range(row.cell_widths[j]):
                                # add empty cell
                                self.rows[k].cells.insert(
                                    j, TableCell(parent=self.rows[k], spec=spec)
                                )

    def get_table_spec(self) -> str:
        """Generates a LaTeX table spec."""
        # First make table square
        self._square_table()
        # Find the most used spec in regular cells (no multi-col/row)
        specs = [Spec() for _ in range(self.width)]
        for i, col in enumerate(self.cols):
            counts = defaultdict(int)
            for cell in col:
                if cell is None or cell.spec.align == "":
                    continue
                if cell.multicolumn is None and cell.multirow is None:
                    counts[cell.spec] += 1
            if len(counts) > 0:
                specs[i] = max(counts, key=counts.get)
        # convert all cells that don't match the column style into a multicol{1}{custom_spec}
        for i, col in enumerate(self.cols):
            for cell in col:
                if cell is not None and cell.spec != specs[i]:
                    # check if there is text in the cell. If not alignment doesn't matter
                    if (
                        len(cell.children) == 0
                        and cell.spec.l == specs[i].l
                        and cell.spec.r == specs[i].r
                    ):
                        continue
                    # convert any standard cell into a multicol cell of width 1
                    if cell.multicolumn is None:
                        cell.multicolumn = 1
        # generate final latex table spec
        out = " ".join([str(spec) for spec in specs])
        out = re.sub(r"(\|) +(\w)", r"\1\2", out)
        out = re.sub(r"(\w) +(\|)", r"\1\2", out)
        return out

    @property
    def plaintext(self):
        return "\n".join([row.plaintext for row in self.rows])


@dataclass
class Table(Element):
    id: str = None
    caption: Element = None
