#!python
# append multiple images, tables, pdfs and plain text into a single document
# in case of using pdf files as input, this module is required:
# pip install pymupdf
'''
usage: $0 items#item*csv,xlsx,png,jpg,txt,pdf,md output*html,pdf,md
'''

import sys, os.path, re
import numpy as np
import pandas as pd
# import modules from a pyz (zip) file with same name as scripts
sys.path.append(os.path.splitext(sys.argv[0])[0] + '.pyz')

from io import BytesIO
import base64
import struct

from _gui import usage_gui, commalist

def rl_append_document(items, output = None):
  from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, Preformatted
  from reportlab.lib import colors
  from reportlab.lib.styles import getSampleStyleSheet
  stylesheet=getSampleStyleSheet()
  od = []
  for k in range(len(items)):
    v = items[k]
    if isinstance(v, tuple):
      k,v = v
    d = None
    if isinstance(v, pd.DataFrame):
      d = Table([v.columns.tolist()] + v.values.tolist(), style=[('TEXTCOLOR', (0,0), (-1,0), colors.gray)])
    elif isinstance(v, BytesIO):
      d = Image(v)
      print(d)
    elif isinstance(v, bytes):
      d = Image(BytesIO(v))
    else:
      d = Preformatted(v, stylesheet['Code'])

    if d is not None:
      od.append(d)

  doc = SimpleDocTemplate(output)
  doc.build(od)
  return doc

def md_append_document(items, output = None):
  import markdown
  md = markdown.Markdown(output_format='html5', extensions=['tables'])
  
  od = ''
  for k in range(len(items)):
    v = items[k]
    if isinstance(v, tuple):
      k,v = v
    d = None
    if isinstance(v, BytesIO):
      v = v.getvalue()
    if isinstance(v, pd.DataFrame):
      d = v.to_markdown(index=False)
    elif isinstance(v, bytes):
      if len(v) > 24:
        magic, width, height = struct.unpack('>x3s12xII', v[:24])
        if magic == b'PNG':
          b = base64.b64encode(v)
          d = f'![{k}](data:image/png;base64,{b.decode()})'

    else:
      d = v

    if d is not None:
      od += f'### {k}'
      od += chr(10)
      od += d
      od += chr(10)

  if not output:
    print(od)
  elif output.lower().endswith('md'):
    with open(output, 'w') as fd:
      fd.write(od)
  elif output.lower().endswith('html'):
    with open(output, 'w', encoding='utf-8', errors='xmlcharrefreplace') as fd:
      fd.write('<html><head><title>%s</title></head><body>\n' % os.path.basename(output))
      fd.write(md.convert(od))
      fd.write('</body></html>\n')

  return od


def parse_items(items):
  kv = []
  for item in items:
    bn = os.path.splitext(os.path.basename(item))[0]
    print(bn)
    if isinstance(item, pd.DataFrame):
      kv.append((bn,item))
    elif not isinstance(item, str):
      ...
    elif item.lower().endswith('csv'):
      kv.append((bn,pd.read_csv(item)))
    elif item.lower().endswith('xlsx'):
      kv.append((bn,pd.read_excel(item)))
    elif re.search('png|jpg', item, re.IGNORECASE):
      kv.append((bn, '![%s](%s)' % (os.path.basename(item), item)))
    elif item.lower().endswith('pdf'):
      try:
        import fitz
      except:
        print("pymupdf module required for PDF support: ! pip install pymupdf")

      for k,v in enumerate(fitz.open(item)):
        kv.append((f'{bn} page {k}', v.get_pixmap().tobytes()))
    else:
      with open(item) as fd:
        kv.append((bn, fd.read()))

  return kv

def db_append_document(items, output = None):
  if not output and output is not None:
    output = None
  if output is not None and output.lower().endswith('pdf'):
    return rl_append_document(items, output)
  else:
    return md_append_document(items, output)

def main(items, output):
  db_append_document(parse_items(items.split(';')), output)

if __name__=='__main__':
  usage_gui(__doc__)
