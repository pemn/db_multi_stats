#!python
# create a simple custom histogram chart
# input: a database in a known format (csv, xls, bmf, isis)
# condition: a expression to filter data where evaluated to false
# series: (optional) zero or more variables which will split generate charts
# variables: which variables to analise
# output: (optional) write result to a file
# display: show results in graphic windows
# v1.0 04/2024 paulo.ernesto
'''
usage: $0 input*bmf,csv,isis condition series:input variables#variable:input output*pdf,html,md display@
'''
import sys, os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, log, pd_load_dataframe, table_field, commalist, plt_getfig_bytes, save_images_as_pdf

def pd_histogram(df, series, variables):
  r = []
  rows = [df.index]
  if len(series):
    df = df.set_index(series)
    rows = df.index.unique()

  for row in rows:
    df_row = df.loc[row]
    log(row,len(df_row),'rows')
    if df_row.empty:
      continue
    #None, figsize=np.multiply(plt.rcParams["figure.figsize"], 2)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(row if isinstance(row, str) else '/'.join(row))
    for v in variables:
      dv = df_row[v]
      if np.ndim(dv) == 0:
        dv = np.array([dv])

      if not dv.any():
        continue
      ax.hist(dv, 16, [dv.min(),dv.max()], histtype='stepfilled', alpha=0.5, ec='k', label="%s x̄ %.2f" % (v, dv.mean()))
    ax.legend()
    r.append(plt_getfig_bytes(fig))

  return r


def db_histogram(input_path, condition, series, variables, output, display):
  if series:
    series = series.split(',')
  else:
    series = []
  variables = commalist(variables).split()

  od = pd_histogram(pd_load_dataframe(input_path, condition), series, variables)

  if output and len(od):
    if output.lower().endswith('pdf'):
      save_images_as_pdf(od, output)
    else:
      from db_append_document import md_append_document
      md_append_document(od, output)

  if int(display):
    plt.show()

main = db_histogram

if __name__=="__main__":
  usage_gui(__doc__)
