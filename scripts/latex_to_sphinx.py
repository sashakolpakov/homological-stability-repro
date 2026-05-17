#!/usr/bin/env python3
"""Generate a Sphinx/reStructuredText manuscript page from dire_short.tex."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


UNDERLINES = {
    1: "=",
    2: "-",
    3: "~",
    4: "^",
}


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        escaped = False
        cut = len(line)
        for index, char in enumerate(line):
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == "%" and not escaped:
                cut = index
                break
            escaped = False
        lines.append(line[:cut].rstrip())
    return "\n".join(lines)


def find_matching_brace(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unmatched brace")


def command_arg(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\s*\{{", text)
    if not match:
        return ""
    start = match.end() - 1
    end = find_matching_brace(text, start)
    return text[start + 1 : end]


def command_args(text: str, command: str) -> list[str]:
    values = []
    pattern = re.compile(rf"\\{command}(?:\[[^\]]+\])?\s*\{{")
    index = 0
    while True:
        match = pattern.search(text, index)
        if not match:
            break
        start = match.end() - 1
        end = find_matching_brace(text, start)
        values.append(text[start + 1 : end])
        index = end + 1
    return values


def replace_one_arg_commands(text: str, commands: dict[str, str]) -> str:
    changed = True
    while changed:
        changed = False
        for command, template in commands.items():
            needle = f"\\{command}"
            index = text.find(needle)
            while index >= 0:
                brace = index + len(needle)
                while brace < len(text) and text[brace].isspace():
                    brace += 1
                if brace >= len(text) or text[brace] != "{":
                    index = text.find(needle, index + 1)
                    continue
                end = find_matching_brace(text, brace)
                value = replace_one_arg_commands(text[brace + 1 : end], commands)
                text = text[:index] + template.format(value) + text[end + 1 :]
                changed = True
                break
            if changed:
                break
    return text


def replace_texorpdfstring(text: str) -> str:
    needle = r"\texorpdfstring"
    index = text.find(needle)
    while index >= 0:
        first_start = index + len(needle)
        while first_start < len(text) and text[first_start].isspace():
            first_start += 1
        if first_start >= len(text) or text[first_start] != "{":
            index = text.find(needle, index + 1)
            continue
        first_end = find_matching_brace(text, first_start)
        second_start = first_end + 1
        while second_start < len(text) and text[second_start].isspace():
            second_start += 1
        if second_start >= len(text) or text[second_start] != "{":
            index = text.find(needle, index + 1)
            continue
        second_end = find_matching_brace(text, second_start)
        replacement = text[first_start + 1 : first_end]
        text = text[:index] + replacement + text[second_end + 1 :]
        index = text.find(needle, index + len(replacement))
    return text


def latex_to_text(text: str, figure_refs: dict[str, str], section_refs: dict[str, str]) -> str:
    text = replace_texorpdfstring(text)
    text = re.sub(r"``([^']*?)''", r'"\1"', text)
    text = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(m.group(1).split(",")) + "]", text)
    text = re.sub(
        r"\\ref\{([^}]+)\}",
        lambda m: figure_refs.get(m.group(1), section_refs.get(m.group(1), m.group(1))),
        text,
    )
    text = replace_one_arg_commands(
        text,
        {
            "textbf": "**{}**",
            "textit": "*{}*",
            "emph": "*{}*",
            "texttt": "``{}``",
        },
    )
    math_tokens = {}

    def protect_math(content: str) -> str:
        token = f"@@INLINE_MATH_{len(math_tokens)}@@"
        math_tokens[token] = f":math:`{content}`"
        return token

    text = re.sub(r"\\\((.*?)\\\)", lambda m: protect_math(m.group(1)), text)
    text = re.sub(r"\$(.+?)\$", lambda m: protect_math(m.group(1)), text)
    text = re.sub(r"\\url\{([^}]+)\}", r"`\1 <\1>`_", text)
    text = re.sub(r"\\operatorname\{([^}]+)\}", r"\\operatorname{\1}", text)
    replacements = {
        r"\\": " ",
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\{": "{",
        r"\}": "}",
        "~": " ",
        "--": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    for token, value in math_tokens.items():
        text = text.replace(token, value)
    text = re.sub(r"(:math:`[^`]+`)(?=[A-Za-z0-9])", r"\1\\ ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def heading(title: str, level: int) -> list[str]:
    underline = UNDERLINES.get(level, '"')
    return [title, underline * len(title), ""]


def extract_body(text: str) -> str:
    match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, flags=re.S)
    if not match:
        raise ValueError("document body not found")
    return match.group(1)


def extract_environment(text: str, env: str) -> tuple[str, str]:
    pattern = rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return "", text
    return match.group(1).strip(), text[: match.start()] + text[match.end() :]


def build_reference_maps(body: str) -> tuple[dict[str, str], dict[str, str]]:
    figure_refs = {}
    for index, match in enumerate(re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", body, flags=re.S), start=1):
        for label in re.findall(r"\\label\{([^}]+)\}", match.group(0)):
            figure_refs[label] = str(index)

    section_refs = {}
    for match in re.finditer(r"\\(section|subsection|subsubsection)\{", body):
        brace = match.end() - 1
        end = find_matching_brace(body, brace)
        title = body[brace + 1 : end]
        tail = body[end : end + 120]
        for label in re.findall(r"\\label\{([^}]+)\}", tail):
            section_refs[label] = latex_to_text(title, figure_refs, {})
    return figure_refs, section_refs


def protect_blocks(body: str) -> tuple[str, dict[str, tuple[str, str]]]:
    blocks: dict[str, tuple[str, str]] = {}

    def repl(kind: str):
        def inner(match: re.Match[str]) -> str:
            token = f"@@{kind.upper()}{len(blocks)}@@"
            blocks[token] = (kind, match.group(0))
            return f"\n\n{token}\n\n"

        return inner

    for kind, pattern in (
        ("figure", r"\\begin\{figure\}.*?\\end\{figure\}"),
        ("enumerate", r"\\begin\{enumerate\}.*?\\end\{enumerate\}"),
        ("itemize", r"\\begin\{itemize\}.*?\\end\{itemize\}"),
        ("math", r"\\\[(.*?)\\\]"),
    ):
        body = re.sub(pattern, repl(kind), body, flags=re.S)
    body = body.replace(r"\printbibliography", "\n\n@@BIBLIOGRAPHY@@\n\n")
    return body, blocks


def render_math(block: str) -> list[str]:
    content = re.sub(r"^\\\[|\\\]$", "", block.strip(), flags=re.S).strip()
    lines = [".. math::", ""]
    lines.extend(f"   {line}" if line.strip() else "" for line in content.splitlines())
    lines.append("")
    return lines


def render_list(block: str, ordered: bool, figure_refs: dict[str, str], section_refs: dict[str, str]) -> list[str]:
    content = re.sub(r"^\\begin\{(?:enumerate|itemize)\}|\\end\{(?:enumerate|itemize)\}$", "", block.strip(), flags=re.S)
    items = re.split(r"\\item\s*", content)
    marker = "#." if ordered else "*"
    lines = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        lines.append(f"{marker} {latex_to_text(item, figure_refs, section_refs)}")
    lines.append("")
    return lines


def captions_from_block(block: str) -> list[str]:
    captions = []
    index = 0
    while True:
        match = re.search(r"\\caption\{", block[index:])
        if not match:
            break
        start = index + match.end() - 1
        end = find_matching_brace(block, start)
        captions.append(block[start + 1 : end])
        index = end + 1
    return captions


def render_figure(block: str, number: str, figure_refs: dict[str, str], section_refs: dict[str, str]) -> list[str]:
    images = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
    captions = captions_from_block(block)
    overall_caption = captions[-1] if captions else ""
    subcaptions = captions[: len(images)]
    lines = [f".. container:: manuscript-figure", ""]
    for image, caption in zip(images, subcaptions):
        lines.extend(
            [
                f"   .. figure:: _static/paper/{image}",
                "      :width: 48%",
                "",
                f"      {latex_to_text(caption, figure_refs, section_refs)}",
                "",
            ]
        )
    if overall_caption:
        lines.extend(
            [
                f"   **Figure {number}.** {latex_to_text(overall_caption, figure_refs, section_refs)}",
                "",
            ]
        )
    return lines


def parse_bib_entries(path: Path) -> list[tuple[str, dict[str, str]]]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    entries = []
    index = 0
    while True:
        match = re.search(r"@\w+\s*\{\s*([^,]+),", text[index:], flags=re.S)
        if not match:
            break
        entry_start = index + match.start()
        body_start = index + match.end()
        brace = text.find("{", entry_start)
        entry_end = find_matching_brace(text, brace)
        key = match.group(1).strip()
        body = text[body_start:entry_end]
        fields = {}
        pos = 0
        while pos < len(body):
            field = re.search(r"([A-Za-z]+)\s*=", body[pos:])
            if not field:
                break
            name = field.group(1).lower()
            value_start = pos + field.end()
            while value_start < len(body) and body[value_start].isspace():
                value_start += 1
            if value_start >= len(body):
                break
            if body[value_start] == "{":
                value_end = find_matching_brace(body, value_start)
                value = body[value_start + 1 : value_end]
                pos = value_end + 1
            elif body[value_start] == '"':
                value_end = body.find('"', value_start + 1)
                value = body[value_start + 1 : value_end]
                pos = value_end + 1
            else:
                value_end = body.find(",", value_start)
                if value_end < 0:
                    value_end = len(body)
                value = body[value_start:value_end]
                pos = value_end + 1
            fields[name] = value.strip()
        entries.append((key, fields))
        index = entry_end + 1
    return entries


def render_bibliography(bib_path: Path) -> list[str]:
    lines = heading("References", 1)
    for key, fields in parse_bib_entries(bib_path):
        author = latex_to_text(fields.get("author", ""), {}, {})
        title = latex_to_text(fields.get("title", ""), {}, {})
        journal = latex_to_text(fields.get("journal") or fields.get("publisher") or fields.get("howpublished", ""), {}, {})
        year = fields.get("year", "").strip()
        doi = fields.get("doi", "").strip()
        url = fields.get("url", "").strip()
        parts = [part for part in (author, f"*{title}*" if title else "", journal, year) if part]
        line = f"* [{key}] " + ". ".join(parts)
        if doi:
            line += f". DOI: {doi}"
        if url:
            line += f". URL: {url}"
        lines.append(line)
    lines.append("")
    return lines


def render_body(body: str, blocks: dict[str, tuple[str, str]], bib_path: Path, figure_refs: dict[str, str], section_refs: dict[str, str]) -> list[str]:
    lines: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            lines.append(latex_to_text(" ".join(paragraph), figure_refs, section_refs))
            lines.append("")
            paragraph.clear()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line in blocks:
            flush()
            kind, block = blocks[line]
            if kind == "math":
                lines.extend(render_math(block))
            elif kind == "figure":
                label = next(iter(re.findall(r"\\label\{([^}]+)\}", block)), "")
                number = figure_refs.get(label, "")
                lines.extend(render_figure(block, number, figure_refs, section_refs))
            elif kind == "enumerate":
                lines.extend(render_list(block, True, figure_refs, section_refs))
            elif kind == "itemize":
                lines.extend(render_list(block, False, figure_refs, section_refs))
            continue
        if line == "@@BIBLIOGRAPHY@@":
            flush()
            lines.extend(render_bibliography(bib_path))
            continue
        section = re.match(r"\\(section|subsection|subsubsection)\{", line)
        if section:
            flush()
            brace = section.end() - 1
            end = find_matching_brace(line, brace)
            title = latex_to_text(line[brace + 1 : end], figure_refs, section_refs)
            level = {"section": 1, "subsection": 2, "subsubsection": 3}[section.group(1)]
            lines.extend(heading(title, level))
            rest = re.sub(r"\\label\{[^}]+\}", "", line[end + 1 :]).strip()
            if rest:
                paragraph.append(rest)
            continue
        if line in {r"\maketitle", r"\newpage", r"\medskip"}:
            flush()
            continue
        line = re.sub(r"\\label\{[^}]+\}", "", line)
        paragraph.append(line)

    flush()
    return lines


def convert(latex_path: Path, bib_path: Path, output_path: Path) -> None:
    source = strip_comments(latex_path.read_text(encoding="utf-8"))
    body = extract_body(source)
    figure_refs, section_refs = build_reference_maps(body)
    abstract, body = extract_environment(body, "abstract")
    body = body.replace(r"\maketitle", "")
    body, blocks = protect_blocks(body)

    title = latex_to_text(command_arg(source, "title"), figure_refs, section_refs)
    authors = [latex_to_text(author, figure_refs, section_refs) for author in command_args(source, "author")]
    affiliations = [latex_to_text(affil, figure_refs, section_refs) for affil in command_args(source, "affil")]

    lines = heading(title, 1)
    if authors:
        lines.extend(["; ".join(authors), ""])
    for affiliation in affiliations:
        lines.extend([affiliation, ""])
    if abstract:
        lines.extend(heading("Abstract", 2))
        lines.extend([latex_to_text(abstract, figure_refs, section_refs), ""])
    lines.extend(render_body(body, blocks, bib_path, figure_refs, section_refs))

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", type=Path, default=Path("dire_short.tex"))
    parser.add_argument("--bib", type=Path, default=Path("dire_short.bib"))
    parser.add_argument("--output", type=Path, default=Path("docs/paper.rst"))
    args = parser.parse_args()
    convert(args.latex, args.bib, args.output)


if __name__ == "__main__":
    main()
