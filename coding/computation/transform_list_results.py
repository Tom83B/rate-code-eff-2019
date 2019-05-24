import sys
sys.path.append('/Users/tomasbarta/Documents/skola/doktorat/ms-thesis/program')

from tqdm import tqdm
import pandas as pd
from analysis.information import CIDOChannel
import numpy as np

print('loading pkl file')
res_df = pd.read_pickle('transition_scan.pkl')
print('pkl file loaded')

rates = {}

for ix, sub_df in tqdm(res_df.groupby(level=[0, 1, 2])):
    sub_df = sub_df.loc[ix]
    
    for from_stim, col in sub_df.iteritems():
        rates[(*ix, from_stim)] = CIDOChannel(col, max_rate=500).rates

pd.Series(rates).to_pickle('transition_rates.pkl')