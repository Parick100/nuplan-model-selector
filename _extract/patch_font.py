"""The PPTD compiler hardcodes typeface="Arial" and ignores the page fontFace field.
Rewrite the slide XML in place so the delivered deck uses Times New Roman while every
run of text stays a native, editable PowerPoint text box.

Re-run this after any pptd_render of this project."""
import io
import os
import zipfile

DECK = os.path.join(
    "Which-Model-When-Generative-Vehicle-Trajectory-Prediction",
    "pre1_which_model_when.pptx",
)
FONT = "Times New Roman"
OLD = ['typeface="Arial Black"', 'typeface="Arial"']

zin = zipfile.ZipFile(DECK, "r")
items = zin.infolist()
blobs = {it.filename: zin.read(it.filename) for it in items}
zin.close()

patched = 0
hits = 0
for name in list(blobs):
    if not name.endswith(".xml"):
        continue
    try:
        txt = blobs[name].decode("utf-8")
    except UnicodeDecodeError:
        continue
    new = txt
    for o in OLD:
        hits += new.count(o)
        new = new.replace(o, 'typeface="%s"' % FONT)
    if new != txt:
        blobs[name] = new.encode("utf-8")
        patched += 1

zout = zipfile.ZipFile(DECK, "w", zipfile.ZIP_DEFLATED)
for it in items:
    zout.writestr(it, blobs[it.filename])
zout.close()

print("patched parts:", patched, "| typeface hits replaced:", hits)

# verify
z = zipfile.ZipFile(DECK)
s3 = z.read("ppt/slides/slide3.xml").decode("utf-8")
import re
print("remaining Arial refs in slide3:", s3.count("Arial"))
print("Times New Roman refs in slide3:", s3.count("Times New Roman"))
z.close()
