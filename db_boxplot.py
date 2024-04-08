#!python
# create a custom boxplot chart
# input: a database in a known format (csv, xls, bmf, isis)
# condition: a expression to filter data where evaluated to false
# series: (optional) zero or more variables which will split on its own chart
# variables: which variables to analise
# output: (optional) write result to a file
# display: show results in graphic windows
# v1.0 04/2024 paulo.ernesto
'''
usage: $0 input*bmf,csv,isis condition series:input variables#variable:input mode%lito,variable output*pdf,html,md display@
'''
import sys, os.path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, log, pd_load_dataframe, commalist, plt_getfig_bytes, save_images_as_pdf


def pd_boxplot_variable(df, series, variables):
  r = []
  rows = [df.index]
  if len(series):
    df = df.set_index(series)
    rows = df.index.unique()

  for row in rows:
    name = row
    if not isinstance(name, str):
      if isinstance(name, pd.Index):
        name = ''
      else:
        name = '/'.join(row)
    df_row = df.loc[row]
    log(name,len(df_row),'rows')
    if df_row.empty:
      continue
    fig = plt.figure()
    ax = fig.gca()

    ax.boxplot([df_row[_] for _ in variables], labels=variables)
    ax.set_title(name)
    # min and max points with different markers, sobreposing flyers
    ax.scatter(np.arange(1, len(variables)+1), [df_row[_].min() for _ in variables], marker='o')
    ax.scatter(np.arange(1, len(variables)+1), [df_row[_].max() for _ in variables], marker='o')

    r.append(plt_getfig_bytes(fig))

  return r

def pd_boxplot_lito(df, series, variables):
  r = []
  rows = [df.index]
  if len(series):
    df = df.set_index(series)
    rows = df.index.unique()

  for v in variables:
    if v not in df:
      continue
    fig = plt.figure()
    ax = fig.gca()

    ax.boxplot([df.loc[_, v] for _ in rows], labels=[''] if len(rows) == 1 else rows)
    ax.scatter(np.arange(1, len(rows)+1), [df.loc[_, v].min() for _ in rows], marker='o')
    ax.scatter(np.arange(1, len(rows)+1), [df.loc[_, v].max() for _ in rows], marker='o')

    ax.set_title(v)

    r.append(plt_getfig_bytes(fig))

  return r

def db_boxplot(input_path, condition, series, variables, mode, output, display):
  if series:
    series = series.split(',')
  else:
    series = []
  variables = commalist(variables).split()

  if mode == 'lito':
    od = pd_boxplot_lito(pd_load_dataframe(input_path, condition), series, variables)
  else:
    od = pd_boxplot_variable(pd_load_dataframe(input_path, condition), series, variables)

  if output and len(od):
    if output.lower().endswith('pdf'):
      save_images_as_pdf(od, output)
    else:
      from db_append_document import md_append_document
      md_append_document(od, output)

  if int(display):
    plt.show()

main = db_boxplot

if __name__=="__main__":
  usage_gui(__doc__)
