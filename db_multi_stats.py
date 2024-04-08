#!python
# run multiple statistic routines and generate a single report (html or pdf)
"""
usage: $0 input_path*bmf,csv,xlsx,isis condition series:input_path values#value:input_path fivenum@1 weight:input_path boxplot_lito@ boxplot_variable@ histogram@ scatter@ output*pdf,html
"""

import sys, os.path
import pandas as pd

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, log, pd_load_dataframe, pd_save_dataframe
from db_append_document import db_append_document

def db_multi_stats(input_path, condition, series, values, fivenum, weight, boxplot_lito, boxplot_variable, histogram, scatter, output):
  log("# db_multi_stats start")
  df = pd_load_dataframe(input_path, condition)
  if values:
    values = values.split(';')

  items = []

  if int(fivenum):
    from bm_fivenum_weight import pd_fivenum_weight
    weight = weight.split(',') if weight else []
    items.append(('fivenum', pd_fivenum_weight(df, series, values, weight)))
  
  if int(boxplot_lito):
    from db_boxplot import pd_boxplot_lito
    items.extend(pd_boxplot_lito(df, series, values))
    
  if int(boxplot_variable):
    from db_boxplot import pd_boxplot_variable
    items.extend(pd_boxplot_variable(df, series, values))
  
  if int(histogram):
    from db_histogram import pd_histogram
    items.extend(pd_histogram(df, series, values))

  if int(scatter):
    from db_scatterplot import pd_scatterplot
    items.extend(pd_scatterplot(df, series, values))

  db_append_document(items, output)

  log("# db_multi_stats finished")

main = db_multi_stats

if __name__=="__main__": 
  usage_gui(__doc__)
