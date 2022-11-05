import os
import numpy as np
import pandas as pd
import toyplot.svg
import toyplot.svg as ttsvg
import toyplot.pdf as ttpdf
from helpers.draw_tree_utils import draw_tree
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.express as px
from plotly.express.colors import sample_colorscale
from matplotlib import rc

from helpers.draw_tree_utils import edge_color_dict

width = 6.34
fontsize = 6
rc('font', **{'size': fontsize})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# Load results
#output_path = "experiments/tree_visualization/results"

# --------------------------------------------------------------------------------------------------------------
# Set path and parameters
# --------------------------------------------------------------------------------------------------------------
#data_path = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/data_processed/"
output_path = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/experiments/tree_visualization/results"

data_name = "centralparksoil"
agg_level = "Species"
file_cfi = os.path.join(output_path, f'{data_name}_weights/CFIandLabels.csv')

# --------------------------------------------------------------------------------------------------------------
# Load data and weights
# --------------------------------------------------------------------------------------------------------------
# get label for data
node_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Final Parameter"]
# read in cfi values
df_cfi = pd.read_csv(file_cfi, index_col=0)
node_df = df_cfi[node_levels]

# additional label cleaning
node_df["Phylum"] = node_df["Phylum"].str.replace(",", "")
node_df["Species"] = node_df["Species"].str.replace("(", "_")
node_df["Species"] = node_df["Species"].str.replace(")", "_")




if data_name == "cirrhosis":
    node_df_long = node_df.copy(deep=True)
    # shorten labels
    df_labels = node_df["Species"].str.split("_", expand=True)
    df_labels[0] = df_labels[0].str.replace(r"[", "", regex=True)
    df_labels[0] = df_labels[0].str.replace(r"]", "", regex=True)
    node_df["Species"] = df_labels[0].str[:4]+"*_*" + df_labels[1].str[-4:]

# --------------------------------------------------------------------------------------------------------------
# Create Treeplots
# --------------------------------------------------------------------------------------------------------------
cfi_models = [mod for mod in df_cfi.columns if mod not in node_levels]

# shortened labels
for mod in cfi_models:
    cfi_vals = df_cfi[mod].values

    # short labels
    canvas = draw_tree(node_df, cfi_vals, agg_level="Species", show_only_important=True)
    # save tree
    ttsvg.render(canvas, os.path.join(output_path, f"{data_name}_weights", str(data_name)+"_"+str(mod)+"_tree.svg"))

    if data_name == "cirrhosis":
        # long labels
        canvas = draw_tree(node_df_long, cfi_vals, agg_level="Species", show_only_important=True)
        # save tree
        ttsvg.render(canvas,
                     os.path.join(output_path, f"{data_name}_weights", str(data_name) + "_" + str(mod) + "_tree_long.svg"))


# --------------------------------------------------------------------------------------------------------------
# Create Weight Plots
# --------------------------------------------------------------------------------------------------------------
# Load weights
w_dummy = np.load(os.path.join(output_path, f'{data_name}_weights', f'{data_name}_w_dummy.npy'))
w_unifrac_MA = np.load(os.path.join(output_path, f'{data_name}_weights', f'{data_name}_w_unifrac_MA.npy'))
w_unifrac_MB = np.load(os.path.join(output_path, f'{data_name}_weights', f'{data_name}_w_unifrac_MB.npy'))

fig, ax = plt.subplots(1, 3)
# Plot the heatmaps
im0 = ax[0].imshow(w_dummy, cmap="coolwarm")
cbar0 = plt.colorbar(im0,
                     fraction=0.045,
                     location='right',
                     orientation="vertical",
                     drawedges=False, ax=ax[0])
cbar0.ax.tick_params(labelsize=fontsize*0.65, right=False, pad=0)
cbar0.outline.set_visible(False)
im1 = ax[1].imshow(w_unifrac_MA, cmap="coolwarm")
cbar1 = plt.colorbar(im1,
                     fraction=0.045,
                     location='right',
                     orientation="vertical",
                     drawedges=False, ax=ax[1])
cbar1.ax.tick_params(labelsize=fontsize*0.65, right=False, pad=0)
cbar1.outline.set_visible(False)
im2 = ax[2].imshow(w_unifrac_MB, cmap="coolwarm")
cbar2 = plt.colorbar(im2,
                     fraction=0.045,
                     location='right',
                     orientation="vertical",
                     drawedges=False, ax=ax[2])
cbar2.ax.tick_params(labelsize=fontsize*0.65, right=False, pad=0)
cbar2.outline.set_visible(False)
# Add title and adjust formating
ax[0].set_title("Phylum-weights", fontsize=fontsize)
ax[1].set_title("UniFrac-weights ($W^A$)", fontsize=fontsize)
ax[2].set_title("UniFrac-weights ($W^B$)", fontsize=fontsize)
for k in range(2):
    for spine in ax[k].spines.values():
        spine.set(color='gray', linewidth=1, alpha=0.2)
ax[0].tick_params(left=False, bottom=False, labelleft=0, labelbottom=0, pad=0)
ax[1].tick_params(left=False, bottom=False, labelleft=0, labelbottom=0, pad=0)
ax[2].tick_params(left=False, bottom=False, labelleft=0, labelbottom=0, pad=0)
# Finalize plot
plt.tight_layout(pad=2)
fig.set_size_inches(width, width/3)
plt.savefig(os.path.join(output_path, f'{data_name}_weights', f'weight_matrices_{data_name}.pdf'),
            bbox_inches='tight')



# --------------------------------------------------------------------------------------------------------------
# Create Legend
# --------------------------------------------------------------------------------------------------------------

colors_neg = np.sort(cfi_vals[(cfi_vals < 0)] / cfi_vals.min())
colors_pos = np.sort(cfi_vals[(cfi_vals >= 0)] / cfi_vals.max())
discrete_colors_1 = sample_colorscale('Oranges', colors_neg[::-1])
discrete_colors_2 = sample_colorscale("Blues", colors_pos)
discrete_colors = np.hstack([discrete_colors_1, discrete_colors_2])

sorted_cfi_vals_index = np.argsort(cfi_vals)
cfi_dict = dict(zip(node_df[agg_level].values[sorted_cfi_vals_index], np.arange(node_df.shape[0])))

label_style={"font-size":"50px"}#, "font-weight":"bold"}
canvas = toyplot.Canvas(width=500, height=200)
canvas.color_scale(toyplot.color.brewer.map(
    name="Oranges",
    domain_min=-1.0,
    domain_max=0.0,
),bounds=(200, 450, 0, 100), label="Positive CFI")
canvas.color_scale(toyplot.color.brewer.map(
    name="Blues",
    domain_min=0.0,
    domain_max=1.0,
    reverse=True,
),bounds=(200, 450, 80, 150),label="Negative CFI")
phyla_list = list(node_df["Phylum"].value_counts().index)
markers = [(name, toyplot.marker.create(shape="o", size=20, mstyle={"fill": edge_color_dict[name]})) #edge_color_dict[name]
            for name in phyla_list]
canvas.legend(markers,
              label="Phylum",
              bounds=(0, 50, 20, 200)
              )
ttpdf.render(canvas, os.path.join(output_path, f'{data_name}_weights', f'legend_{data_name}.pdf'))
###
# Weight plots for centralparksoil
###

# Load weights
#w_dummy = np.load(os.path.join(output_path,
#                               'centralparksoil_w_dummy.npy'))
#w_unifrac_MA = np.load(os.path.join(output_path,
#                                    'centralparksoil_w_unifrac_MA.npy'))
#w_unifrac_MB = np.load(os.path.join(output_path,
#                                    'centralparksoil_w_unifrac_MB.npy'))

#fig, ax = plt.subplots(1, 3)
# Plot the heatmaps
#im0 = ax[0].imshow(w_dummy, cmap="coolwarm")
#cbar0 = plt.colorbar(im0,
#                     fraction=0.045,
#                     location='right',
#                     orientation="vertical",
#                     drawedges=False, ax=ax[0])
#cbar0.ax.tick_params(labelsize=fontsize*0.65, right=False, pad=0)
#cbar0.outline.set_visible(False)
#im1 = ax[1].imshow(w_unifrac_MA, cmap="coolwarm")
#cbar1 = plt.colorbar(im1,
#                     fraction=0.045,
#                     location='right',
#                     orientation="vertical",
 #                    drawedges=False, ax=ax[1])
#cbar1.ax.tick_params(labelsize=fontsize*0.65, right=False, pad=0)
#cbar1.outline.set_visible(False)
#im2 = ax[2].imshow(w_unifrac_MB, cmap="coolwarm")
#cbar2 = plt.colorbar(im2,
#                     fraction=0.045,
#                     location='right',
#                     orientation="vertical",
#                     drawedges=False, ax=ax[2])
#cbar2.ax.tick_params(labelsize=fontsize*0.65, right=False, pad=0)
#cbar2.outline.set_visible(False)
# Add title and adjust formating
#ax[0].set_title("Phylum-weights", fontsize=fontsize)
#ax[1].set_title("UniFrac-weights ($W^A$)", fontsize=fontsize)
#ax[2].set_title("UniFrac-weights ($W^B$)", fontsize=fontsize)
#for k in range(2):
#    for spine in ax[k].spines.values():
#        spine.set(color='gray', linewidth=1, alpha=0.2)
#ax[0].tick_params(left=False, bottom=False, labelleft=0, labelbottom=0, pad=0)
#ax[1].tick_params(left=False, bottom=False, labelleft=0, labelbottom=0, pad=0)
#ax[2].tick_params(left=False, bottom=False, labelleft=0, labelbottom=0, pad=0)
## Finalize plot
#plt.tight_layout(pad=2)
#fig.set_size_inches(width, width/3)
#plt.savefig(os.path.join(output_path, "weight_matrices_centralparksoil.pdf"),
#            bbox_inches='tight')
