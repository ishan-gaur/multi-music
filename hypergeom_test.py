import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import argparse
import statsmodels
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom, mannwhitneyu, ks_2samp
import networkx as nx

## calculate jaccard score
def jaccard(setA, setB):
    return len(setA.intersection(setB)) / len(setA.union(setB))

def run_hypergeo_enrichr(ont_ts, hierarchygenes, ref_file, fdr_thre=0.01, ji_thre = 0.4, minCompSize=4):  
    '''
    ont_ts: the result hierarchy from the community detection 
    hierarchy genes: total number of genes in the root of the hierarchy 
    ref_file: the reference cellular component/protein complexes table 
    fdr_thre: the fdr cutoff to be collected in the enriched terms (default = 0.01)
    ji_thre: the threshold for Jaccard Idex (default = 0, will set the threshold in the organization step)
    minCompSize: the smallest component size to be considered for the enrichment (default = 4)
    '''
    
    M = len(hierarchygenes)

    track = 0
    ref_df = pd.DataFrame(index=ont_ts.index, columns=ref_file.index, dtype=float)
    ji_df = pd.DataFrame(index=ont_ts.index, columns=ref_file.index, dtype=float)
    for ont_comp, ont_row in tqdm(ont_ts.iterrows(), total=ont_ts.shape[0]):
        track += 1
        ont_comp_genes = ont_row['genes'].split(' ')
        n = ont_row['tsize']
        for comp, row in ref_file.iterrows():
            comp_genes = row['genes'].split(' ')
            N = row['tsize']
            overlap_genes = list(set(ont_comp_genes).intersection(set(comp_genes)))
            x = len(overlap_genes)
            ref_df.at[ont_comp, comp] = hypergeom.sf(x - 1, M, n, N) # calculare the hypergeometric distribution 
            ji_df.at[ont_comp, comp] = jaccard(set(ont_comp_genes), set(comp_genes)) # calculate the jaccard score
            if ji_df.at[ont_comp, comp] == 0 and ref_df.at[ont_comp, comp] < fdr_thre:
                print('=== no overlap, but pval < fdr_thre ===')
                print(ont_comp, comp, ji_df.at[ont_comp, comp], ref_df.at[ont_comp, comp])
                print(ont_comp_genes)
                print(comp_genes)
                print('=== no overlap, but pval < fdr_thre ===')
                sys.exit()
    fdr = multipletests(ref_df.values.flatten(), method='fdr_bh')[1]
    fdr_df = pd.DataFrame(fdr.reshape(ref_df.shape), index=ont_ts.index, columns=ref_file.index, dtype=float)
    
    # 1 if greater than threshold     
    fdr_df[fdr_df > fdr_thre] = 1
    # if perfect overlap, counts regardless of significance
    fdr_df[ji_df == 1] = 0
    # if no overlap, doesn't count regardless of significance
    fdr_df[ji_df == 0] = 1
    # filter out those that don't meet ji threshold
    fdr_df[ji_df < ji_thre] = 1
    return fdr_df



parser = argparse.ArgumentParser(description='Analyze each system in the given hierarchy.')
parser.add_argument('--infname', help='Process number of the test node attributes file')
parser.add_argument('--refname', help='Process number of the reference node attributes file')
parser.add_argument('--w_root', action='store_true', help='Do analysis for root term (largest term).')
parser.add_argument('--minTermSize', default=4, type=int)
parser.add_argument('--FDRthre', default=1.0, type=float)
# parser.add_argument('--FDRthre', default=0.001, type=float)
parser.add_argument('--JIthre', default=0.4, type=float)
args = parser.parse_args()

f = f"sc_node_attributes_{args.infname}.csv"
ref_f = f"sc_node_attributes_{args.refname}.csv"
minTermSize = args.minTermSize
fdrthre = args.FDRthre
jithre = args.JIthre

def load_node_attributes_file(f):
    if not os.path.exists(f):
        print(f)
        raise ValueError('Input node attributes file does not exist!')

    if os.path.getsize(f) == 0:
        print('=== No term in hierarchy! ===')
        sys.exit()

    df = pd.read_csv(f)
    df = df[["CD_MemberList_Size", "CD_MemberList", "HiDeF_persistence"]]
    df.columns = ['tsize', 'genes', 'stability']
    return df

df = load_node_attributes_file(f)
df_ref = load_node_attributes_file(ref_f)

root_size = df_ref['tsize'].max()
hierarchygenes = set(df_ref[df_ref['tsize'] == df_ref['tsize'].max()]['genes'].str.split(' ').sum())

if args.w_root:
    df = df[df['tsize'] >= minTermSize]
    df_ref = df_ref[df_ref['tsize'] >= minTermSize]
else:
    df_ref = df_ref[(df_ref['tsize'] >= minTermSize) & (df_ref['tsize'] < root_size)]
    df = df[(df['tsize'] >= minTermSize) & (df['tsize'] < root_size)]
    
if (df_ref.shape[0] == 0) | (df.shape[0] == 0):
    print('=== No system left after size filter ===')
    sys.exit()
    
    
#run the analysis 
hypergeom_enrich_result = run_hypergeo_enrichr(df.copy(), hierarchygenes, df_ref, fdr_thre=fdrthre, minCompSize=minTermSize,  ji_thre =jithre)
output_file = f'{args.infname}_{args.refname}_hypergeom.csv'
print(f'output_file = {output_file}')
hypergeom_enrich_result.to_csv(output_file)
print('=== finished analyze_hidef_output ===')