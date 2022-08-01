import pandas as pd
import math

from pathlib import Path

if __name__ == '__main__':
    basepath = Path('results', 'reports')
    csv_path = basepath / 'ari_scores.csv'
    dadc_path = Path('results', 'reports', 'ari_DADC.csv')
    
    
    df = pd.read_csv(csv_path) \
            .set_index('dataset')
    
    dadc_rw_df = pd.read_csv(dadc_rw_path) \
                    .set_index('dataset')
    dadc_syn_df = pd.read_csv(dadc_syn_path) \
                    .set_index('dataset')
    
    dadc_df = pd.concat([dadc_rw_df, dadc_syn_df], axis=0)
    dadc_df = dadc_df.rename({'ari': 'DADC'}, axis=1)
    dadc_df.index = dadc_df.index + '.arff'
    
    df = df.merge(dadc_df, left_index=True, right_index=True, how='outer')
    
    mask = df.index.isin(to_keep)
    df = df.loc[mask]
    
    def add_type(row):
        t = 'R' if row.name in to_keep_rw else 'S'
        row['Type'] = t
        return row
    df = df.apply(add_type, axis=1) \
            .sort_values(['Type', 'dataset'], ascending=[False, True])
    
    col_order = ['BridgeClustering', 'DBSCAN', 'HDBSCAN', 'BorderPeelingWrapper', 'DenMuneWrapper', 'AUTOCLUST', 'OPTICS', 'DADC']
    renamer = {
        'BridgeClustering': 'BAC',
        'BorderPeelingWrapper': 'BP',
        'DenMuneWrapper': 'DenMune'
    }
    df = df[col_order].rename(renamer, axis=1)
    
    def formatter(row):
        maxval = None
        for r in row.index:
            if r == 'Type':
                continue
            maxval = max(maxval, row[r]) if not maxval is None else row[r]
        
        for r in row.index:
            if r == 'Type':
                continue
            val = row[r]
            if val == maxval:
                row[r] = f'\\textbf{{{val:.3f}}}'
            elif math.isnan(val):
                row[r] = 'N/A'
            else:
                row[r] = f'{val:.3f}'
        return row
    
    mean_all, median_all = df.mean(), df.median()
    mask = ~df.isna().any(axis=1)
    mean_nan, median_nan = df.loc[mask].mean(), df.loc[mask].median()
    
    mean_all.name = 'Mean (All)'
    median_all.name = 'Median (All)'
    mean_nan.name = f'Mean ({mask.sum()} datasets)'
    median_nan.name = f'Median ({mask.sum()} datasets)'
    
    
    df.index = df.index.str[:-5]
    df.loc[f'Mean ({mask.sum()} datasets)'] = mean_nan
    df.loc[f'Median ({mask.sum()} datasets)'] = median_nan
    df.loc['Mean (All)'] = mean_all
    df.loc['Median (All)'] = median_all
    
    df = df.apply(formatter, axis=1)
    
    df.to_latex(basepath / 'table.tex', escape=False)
        