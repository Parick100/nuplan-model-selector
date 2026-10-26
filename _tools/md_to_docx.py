"""把 REPORT2 的 Markdown 子集转成 .docx（只用标准库，不装 python-docx）。

为什么要自己写：这个环境里 python-docx 和 pandoc 都没有装，而往课程环境里装包
会污染别人的 conda 环境。一个 .docx 本质上就是 zip + 几个固定 XML 文件，
所以用 zipfile 直接拼即可。

支持的 Markdown 子集（正好是报告用到的）：
    # / ## 标题、**粗体**、`行内代码`、表格、有序/无序列表、普通段落。
不支持嵌套、图片、链接渲染（链接按纯文本保留）。

    python _tools/md_to_docx.py REPORT2_second_project_report.md REPORT2.docx
"""
import re
import sys
import zipfile
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ----------------------------------------------------------------- 行内格式
def runs(text, bold=False, size=None, mono=False):
    """把 **粗体** 与 `等宽` 拆成若干 <w:r>。"""
    out = []
    # 先按 **粗体** 切分
    for i, seg in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        is_b = bold or (i % 2 == 1)
        if not seg:
            continue
        # 再按 `行内代码` 切分
        for j, sub in enumerate(re.split(r"`(.+?)`", seg)):
            if not sub:
                continue
            is_c = (j % 2 == 1)
            out.append(_run(sub, bold=is_b, mono=is_c or mono, size=size))
    return "".join(out)


def _run(text, bold=False, mono=False, size=None):
    props = []
    if bold:
        props.append("<w:b/>")
    if mono:
        props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
    if size:
        props.append(f'<w:sz w:val="{int(size*2)}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return (f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>')


# ----------------------------------------------------------------- 块级
def para(text, style=None, size=10.0, space_after=60):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if space_after is not None:
        ppr.append(f'<w:spacing w:after="{space_after}" w:line="240" w:lineRule="auto"/>')
    p = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return f"<w:p>{p}{runs(text, size=size)}</w:p>"


def table(rows, size=9.0):
    ncol = len(rows[0])
    w = 9000 // ncol                     # 总宽约 9000 twips
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
           '<w:tblW w:w="0" w:type="auto"/>'
           '<w:tblBorders>'
           + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
                     for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
           + '</w:tblBorders></w:tblPr>']
    for r, row in enumerate(rows):
        out.append("<w:tr>")
        for cell in row:
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>'
                       + para(cell, size=size, space_after=20)
                       + "</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def convert(md):
    body = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        # 表格
        if s.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            rows = []
            hdr = [c.strip() for c in s.strip("|").split("|")]
            rows.append(hdr)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body.append(table(rows))
            body.append(para("", space_after=40))
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            if lvl == 1:
                body.append(para(txt, style="Title", size=15.0, space_after=80))
            elif lvl == 2:
                body.append(para(txt, style="Heading1", size=11.5, space_after=50))
            else:
                body.append(para(txt, style="Heading2", size=10.5, space_after=40))
            i += 1
            continue

        # 有序列表
        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            body.append(para(f"{m.group(1)}. {m.group(2)}", style="ListParagraph",
                             space_after=40))
            i += 1
            continue

        # 无序列表
        m = re.match(r"^[-*]\s+(.*)$", s)
        if m:
            body.append(para(f"•  {m.group(1)}", style="ListParagraph", space_after=40))
            i += 1
            continue

        # 普通段落（连续的正文行合并）
        buf = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and \
                not re.match(r"^(#{1,4}\s|\||\d+\.\s|[-*]\s)", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        body.append(para(" ".join(buf)))
    return "".join(body)


# ----------------------------------------------------------------- docx 打包
CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

# 紧凑的页面设置：A4 + 0.7 英寸边距，保证一页装得下
SECT = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1000" w:right="1080" w:bottom="1000" w:left="1080"'
        ' w:header="0" w:footer="0" w:gutter="0"/></w:sectPr>')

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/>
<w:sz w:val="20"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="60" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>
<w:pPr><w:jc w:val="left"/><w:spacing w:after="80"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="60" w:after="50"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="40" w:after="40"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>
<w:pPr><w:ind w:left="284" w:hanging="284"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>
<w:tblPr><w:tblCellMar><w:top w:w="20" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>
<w:bottom w:w="20" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr>
</w:style>
</w:styles>"""


def main():
    src, dst = sys.argv[1], sys.argv[2]
    md = open(src, encoding="utf-8").read()
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<w:document xmlns:w="{W}"><w:body>{convert(md)}{SECT}</w:body></w:document>')
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", doc)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
