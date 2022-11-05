import sys
import pandas as pd

sys.path.insert(0, "../")
import toyplot.svg as ttsvg
import os
from helpers.draw_tree_utils import draw_tree



data_name = "cirrhosis"
agg_level = "Species"
data_path = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/data_processed/"
output_path = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/experiments/tree_visualization/results"
file_cfi = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/experiments/tree_visualization/results" \
           "/cirrhosis_weights/CFIandLabels.csv"


# --------------------------------------------------------------------------------------------------------------
# Load cirrhosis data and weights
# --------------------------------------------------------------------------------------------------------------
# get label for data
node_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Final Parameter"]
# read in cfi values
df_cfi = pd.read_csv(file_cfi, index_col=0)

node_df = df_cfi[node_levels]

# --------------------------------------------------------------------------------------------------------------
# Draw tree
# --------------------------------------------------------------------------------------------------------------
cfi_models = [mod for mod in df_cfi.columns if mod not in node_levels]

for mod in cfi_models:
    cfi_vals = df_cfi[mod].values
    canvas = draw_tree(node_df, cfi_vals, agg_level="Species", title="CFI per microbe", show_only_important=True)
    # save tree
    ttsvg.render(canvas, os.path.join(output_path, f"{data_name}_weights", str(data_name)+"_"+str(mod)+"_tree.svg"))



