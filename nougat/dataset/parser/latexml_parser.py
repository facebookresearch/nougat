"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import re
import sys
import requests
from typing import Optional, Set
from bs4 import BeautifulSoup, NavigableString
import soupsieve as sv

from nougat.dataset.parser.document import *


def printerr(*args, **kwargs):
    # uncomment for debugging
    # print(*args, **kwargs)
    pass


latexml_wrapper_selector = sv.compile(
    ", ".join(
        [
            ".ltx_engrafo_equation_container",
            "tbody",
            ".ltx_note_content",
            ".ltx_role_footnote",
            ".ltx_note_type",
            ".ltx_theorem",
            ".ltx_proof",
            ".ltx_quote",
            "blockquote",
            ".ltx_inline-para",
            ".ltx_inline-block",
        ]
    )
)
latexml_ignore_selector = sv.compile(".ltx_rule, .ltx_pagination.ltx_role_newpage")


def is_wrapper_element(element: BeautifulSoup) -> bool:
    return latexml_wrapper_selector.match(element)


def ignore_element(element: BeautifulSoup) -> bool:
    return latexml_ignore_selector.match(element)


def _get_classes(el: BeautifulSoup) -> Set[str]:
    if not hasattr(el, "attrs"):
        return set()
    classes = el.attrs.get("class")
    if classes is None:
        return set()
    return set(classes)


def _detach_selected(element: BeautifulSoup, selector: str) -> None:
    for elem in element.select(selector):
        elem.extract()


def parse_latexml_authors(ltx_authors: BeautifulSoup) -> List[Author]:
    authors = Paragraph()
    parse_latexml_children(ltx_authors, authors)
    return authors


def parse_latexml_citations(cite: BeautifulSoup, parent: Element) -> None:
    """
    Parses LaTeXML citations and appends them as children to the given parent element.

    Args:
        cite (BeautifulSoup): The BeautifulSoup object containing the citation data.
        parent (Element): The parent element to which the citations will be added as children.
    """
    parse_latexml_children(cite, parent)
    if ("[" in parent.plaintext and "]" in parent.plaintext) or re.search(
        r"[A-Za-z]", parent.plaintext
    ):
        return

    parent.children.insert(0, TextElement(content="["))
    parent.children.append(TextElement(content="]"))


def _clean_html_whitespace(text: str) -> str:
    if text.strip():
        text = re.sub(r"(^\n+|\n+$)", "\n", text)
    else:
        text = text.strip("\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def parse_latexml_children(html: BeautifulSoup, parent: Element) -> None:
    """
    Parses LaTeXML children and appends them as appropriate elements to the given parent element.

    Args:
        html (BeautifulSoup): The BeautifulSoup object containing the HTML data.
        parent (Element): The parent element to which the parsed children will be added.
    """
    if html is None:
        return
    for child in html.children:
        classes = _get_classes(child)
        if isinstance(child, NavigableString):
            parent.append(TextElement(content=_clean_html_whitespace(str(child))))
        elif sv.match(
            "p, .ltx_p, div.ltx_para, span.ltx_para, section.ltx_paragraph", child
        ):
            paragraph = parent.append(Paragraph())
            parse_latexml_children(child, paragraph)
        elif sv.match(".ltx_tag", child):
            if "ltx_tag_note" not in classes:
                if sv.match(".ltx_tag_section", child):
                    child.string = child.string.upper()
                elif sv.match(".ltx_tag_subsection", child):
                    child.string = ""
                parse_latexml_children(child, parent)
            elif "ltx_tag_bibitem" in classes:
                parse_latexml_children(child, parent.append(SpanElement()))
        elif sv.match(".ltx_note_outer", child):
            # try to place the footnote outside the current paragraph
            paragraph = parent.find_parent(Paragraph)
            if paragraph is not None and paragraph.parent is not None:
                footnote = paragraph.parent.append(Footnote())
            else:
                footnote = parent.append(Footnote())
            parse_latexml_children(child, footnote)
        elif sv.match(".ltx_note_content > .ltx_note_mark", child):
            footnote = parent.find_parent(Footnote)
            if footnote is not None:
                footnote.id = child.get_text(strip=True)
            else:
                printerr("Unable to find footnote to set its id", file=sys.stderr)
                parse_latexml_children(child, parent)
        elif sv.match("sup", child):
            sup = parent.append(Superscript())
            parse_latexml_children(child, sup)
        elif sv.match("sub", child):
            sub = parent.append(Subscript())
            parse_latexml_children(child, sub)
        elif sv.match("span.ltx_Math, span.ltx_DisplayMath", child):
            inline = "ltx_DisplayMath" not in classes
            math_elem = child.select_one(".mjx-math")
            if math_elem:
                tex = math_elem.attrs["aria-label"]
                if inline:
                    tex = rf"\({tex}\)"
                else:
                    tex = rf"\[{tex}\]"
                parent.append(LatexMath(code=tex, inline=inline))
        elif sv.match("math.ltx_Math", child):
            # not sure if the math tag LaTeXML version specific, but that seems to work
            inline = True
            if "display" in child.attrs:
                inline = child.attrs["display"] == "inline"
            tex = child.attrs["alttext"]
            if inline:
                tex = rf"\({tex}\)"
            else:
                tex = rf"\[{tex}\]"
            parent.append(LatexMath(code=tex, inline=inline))
        elif sv.match("a.ref", child):
            link = parent.append(Link())
            link.target = child.attrs.get("href")
            parse_latexml_children(child, link)
        elif sv.match(
            ".ltx_ref.ltx_missing_citation, .ltx_ref.ltx_missing_label", child
        ):
            placeholder = child.get_text().strip()
            resolved = False
            if placeholder.isnumeric():
                parent.append(TextElement(content=placeholder))
                resolved = True
            else:
                target = child.attrs.get("href")
                if target is not None:
                    potential_num = target.partition(".bib")[2]
                    if potential_num.isnumeric():
                        parent.append(TextElement(content=potential_num))
                        resolved = True
            if not resolved:
                raise ValueError("missing reference detected")
        elif sv.match(
            ".ltx_bibblock, .ltx_role_author, .ltx_contact, .ltx_role_email, .ltx_role_affiliation",
            child,
        ):
            parse_latexml_children(child, parent.append(SpanElement()))
            parent.append(TextElement(content="\n"))
        elif sv.match(
            ".ltx_authors, .ltx_personname, .ltx_role_creation.ltx_date, .ltx_engrafo_author_notes, .ltx_author_notes, .ltx_date.ltx_role_creation",
            child,
        ):
            parse_latexml_children(child, parent.append(Paragraph()))
            parent.append(TextElement(content="\n"))
        elif sv.match(
            ".ltx_author_before, .ltx_role_pubyear, .ltx_role_pagerange", child
        ):
            pass
        elif sv.match("h1.ltx_title_document", child):
            doc = parent.find_parent(Document)
            if doc is not None:
                if doc.title is None:
                    doc.title = SectionHeader(parent=doc)
                    doc.title.hnum = int(child.name[1])
                    parse_latexml_children(child, doc.title)
                else:
                    printerr("Document title is already set", file=sys.stderr)
            else:
                printerr("Unable to find document to set title", file=sys.stderr)
        elif sv.match("section", child):
            if ".ltx_bibliography" not in classes:
                section = parent.append(Section())
                parse_latexml_children(child, section)
        elif sv.match("h1, h2, h3, h4, h5, h6", child) and "ltx_title" in classes:
            if {"ltx_title_theorem", "ltx_title_proof"} & classes:
                parse_latexml_children(child, parent)
                parent.append(TextElement(content=": "))
            elif isinstance(parent, Section):
                parent.hnum = int(child.name[1])
                if parent.header is None:
                    parent.header = SpanElement()
                parse_latexml_children(child, parent.header)
            else:
                printerr("Dangling title element", file=sys.stderr)
                parse_latexml_children(child, parent)
        elif sv.match(".ltx_TOC.ltx_toc_toc", child):
            s = parent.append(Section(hnum=6, header=TextElement(content="Contents")))
            parse_latexml_children(child, s.append(Paragraph()))
        elif sv.match(
            "ul.ltx_itemize, ul.ltx_toclist, ul.ltx_biblist, ol.ltx_enumerate", child
        ):
            lst = parent.append(ListContainer())
            lst.ordered = child.name == "ol"
            parent_list = parent.find_parent(ListContainer)
            lst.level = parent_list.level + 1 if parent_list is not None else 1
            parse_latexml_children(child, lst)
        elif sv.match("li.ltx_item, li.ltx_tocentry, li.ltx_bibitem", child):
            lst = parent.find_parent(ListContainer)
            if lst is not None:
                item = lst.add_item(ListItem())
                parse_latexml_children(child, item)
            else:
                printerr("List item outside list", file=sys.stderr)
        elif sv.match("cite", child):
            span = parent.append(SpanElement())
            parse_latexml_citations(child, span)
        elif sv.match("a.ltx_ref", child):
            target = child.attrs.get("href")
            if target.startswith("#bib"):  # citation link
                in_ref = parent.append(InlineRef())
                in_ref.target = target
                text = child.get_text()
                in_ref.target = target
                if text.strip().isnumeric():
                    in_ref.append(TextElement(content=text))
                elif re.search(r"[A-Za-z][:;.,_]?\d", text):
                    # probably a broken citation, go with link number instead
                    in_ref.append(
                        TextElement(
                            content=re.sub(r"\D", "", target.partition(".bib")[2])
                        )
                    )
                else:
                    raise ValueError('unusable reference "%s"' % text)
                doc = parent.find_parent(Document)
                if doc:
                    doc.add_inline_ref(in_ref)
            else:
                link = parent.append(Link())
                link.target = target
                parse_latexml_children(child, link)
        elif sv.match("a", child) and len(classes) == 0:
            target = child.attrs.get("href")
            parse_latexml_children(child, parent.append(Link(target=target)))
        elif sv.match(".ltx_eqn_table", child):
            eqn_list = parent.append(EquationList())
            parse_latexml_children(child, eqn_list)
        elif sv.match(".ltx_eqn_row", child):
            eqn_list = parent.find_parent(EquationList)
            if eqn_list is not None:
                eqn = eqn_list.add_equation(Equation())
                parse_latexml_children(child, eqn)
            else:
                printerr("Dangling equation row", file=sys.stderr)
                parse_latexml_children(child, parent)
        elif sv.match(".ltx_eqn_cell", child):
            parse_latexml_children(child, parent)
        elif sv.match("table, span.ltx_tabular, div.ltx_tabular", child):
            tabular = parent.append(Tabular())
            parse_latexml_children(child, tabular)
        elif sv.match("thead.ltx_thead", child):
            table = parent.find_parent(Tabular)
            if table is not None:
                parse_latexml_children(child, table)
            else:
                printerr("Table header element outside table", file=sys.stderr)
        elif sv.match("tbody.ltx_tbody", child):
            parse_latexml_children(child, parent)
        elif sv.match("tr.ltx_tr", child):
            table = parent.find_parent(Tabular)
            if table is not None:
                row = table.add_row(TableRow())
                parse_latexml_children(child, row)
            else:
                printerr("TableRow element outside table", file=sys.stderr)
        elif sv.match("td.ltx_td, th.ltx_th", child):
            row = parent.find_parent(TableRow)
            if row is not None:
                cell = TableCell()
                cell.set_attrs(child.attrs)
                row.add_cell(cell)
                parse_latexml_children(child, cell)
            else:
                printerr("TableData element outside table row", file=sys.stderr)
        elif sv.match("span.ltx_text, em.ltx_emph", child):
            if (
                child.find_parent(ListItem) is None
                or child.get_text() != "[label=0)]"
                or child.get_text() != "[leftmargin=*] "
            ):
                if "ltx_font_italic" in classes:
                    elem = Italic()
                elif "ltx_font_bold" in classes:
                    elem = Bold()
                else:
                    elem = SpanElement()
                parent.append(elem)
                parse_latexml_children(child, elem)
            else:
                parent.find_parent(ListContainer).items.pop()
        elif sv.match("figure.ltx_table", child):
            figure = parent.append(Table())
            if "id" in child.attrs:
                figure.id = child.attrs["id"]
            parse_latexml_children(child, figure)
        elif sv.match("figure.ltx_figure", child):
            figure = parent.append(Figure())
            if "id" in child.attrs:
                figure.id = child.attrs["id"]
            parse_latexml_children(child, figure)
        elif sv.match("figure.ltx_float", child):
            parse_latexml_children(child, parent)
        elif sv.match(".ltx_listing", child):
            alg = parent.append(Algorithm())
            parse_latexml_children(child, alg)
        elif sv.match(".ltx_listingline", child):
            alg = parent.find_parent(Algorithm)
            if alg is not None:
                line = alg.add_line(Element())
                parse_latexml_children(child, line)
            else:
                printerr("Listing line outside algorithm environment", file=sys.stderr)
        elif sv.match("dl.ltx_description", child):
            def_list = parent.append(DefinitionList())
            parse_latexml_children(child, def_list)
        elif sv.match("dt.ltx_item", child):
            def_list = parent.find_parent(DefinitionList)
            if def_list is not None:
                item = def_list.add_item(Definition())
                item.term = SpanElement(parent=item)
                parse_latexml_children(child, item.term)
            else:
                printerr("Found dangling definition term", file=sys.stderr)
        elif sv.match("dd.ltx_item", child):
            def_list = parent.find_parent(DefinitionList)
            if def_list is not None:
                if def_list.items and def_list.items[-1].definition is None:
                    item = def_list.items[-1]
                else:
                    printerr("Found definition without term", file=sys.stderr)
                    item = def_list.add_item(Definition())
                item.definition = SpanElement(parent=item)
                parse_latexml_children(child, item.definition)
            else:
                printerr("Found dangling definition", file=sys.stderr)
                parse_latexml_children(child, parent)
        elif sv.match("figcaption", child):
            fig = parent.find_parent((Figure, Table))
            if fig is not None:
                if fig.caption is None:
                    fig.caption = Paragraph(parent=fig)
                parse_latexml_children(child, fig.caption)
                fig.caption.append(TextElement(content="\n"))
            else:
                printerr("Figure caption outside figure element", file=sys.stderr)
                para = parent.append(Paragraph())
                parse_latexml_children(child, para)
        elif sv.match(".ltx_break", child):
            parent.append(TextElement(content="\n\n"))
        elif sv.match(".ltx_abstract, .ltx_acknowledgements", child):
            abstract = parent.append(Section())
            parse_latexml_children(child, abstract)
        elif sv.match(".ltx_ERROR", child):
            printerr(
                f"LaTeX error element: {child.get_text(strip=True)}", file=sys.stderr
            )
        elif is_wrapper_element(child):
            parse_latexml_children(child, parent)
        elif ignore_element(child):
            continue
        else:
            printerr(
                f"Unknown LaTeXML element <{child.name}> with classes {', '.join(classes)}",
                file=sys.stderr,
            )
            elem = parent.append(UnknownElement())
            parse_latexml_children(child, elem)


# TODO: move this somewhere else, so I can use it with plaintext too
sess = requests.Session()


def parse_latexml_references(html: BeautifulSoup, doc: Document) -> None:
    for child in html.select("li.ltx_bibitem"):
        child.attrs.get("id")
        ref_text = child.get_text(strip=False).replace("\n", " ")
        reference = Reference()
        reference.title = TextElement(content=child.get_text(strip=True))
        doc.add_reference(reference)


def parse_latexml(
    html: BeautifulSoup,
) -> Optional[Document]:
    if html.article is None:
        printerr("Missing article element", file=sys.stderr)
        return None
    doc = Document()
    parse_latexml_children(html.article, doc)
    parse_latexml_references(
        html.article,
        doc,
    )
    return doc
