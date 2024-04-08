#!python
# create a simple custom scatterplot chart
# input: a database in a known format (csv, xls, bmf, isis)
# condition: a expression to filter data where evaluated to false
# series: (optional) zero or more variables which will split on its own chart
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
import math
import matplotlib.pyplot as plt

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, log, pd_load_dataframe, commalist, plt_getfig_bytes, save_images_as_pdf


def pd_scatterplot(df, series, variables):
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
    #None, figsize=np.multiply(plt.rcParams["figure.figsize"], 2)
    fig = plt.figure()
    ax = fig.gca()
    for i in range(1, len(variables), 2):
      vx = variables[i-1]
      vy = variables[i-0]      
      dx = df_row[vx]
      dy = df_row[vy]
      if np.ndim(dx) == 0:
        dx = np.array([dx])
      if np.ndim(dy) == 0:
        dy = np.array([dy])

      m, b = np.polyfit(dx, dy, 1)
      ax.plot(dx, m * dx + b, color='black')
      coef = np.corrcoef(dx, dy)[0][1]
      ax.set_title(r'%s %s ✕ %s $\rho$ %.2f slope %.2f inter %.2f' % (name, vx, vy, coef, m, b))
      ax.scatter(dx, dy, marker='s')
    r.append(plt_getfig_bytes(fig))

  return r


def db_scatterplot(input_path, condition, series, variables, output, display):
  if series:
    series = series.split(',')
  else:
    series = []
  variables = commalist(variables).split()

  od = pd_scatterplot(pd_load_dataframe(input_path, condition), series, variables)

  if output and len(od):
    if output.lower().endswith('pdf'):
      save_images_as_pdf(od, output)
    else:
      from db_append_document import md_append_document
      md_append_document(od, output)

  if int(display):
    plt.show()

main = db_scatterplot

if __name__=="__main__":
  usage_gui(__doc__)
