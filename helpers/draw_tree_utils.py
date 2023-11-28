import pandas as pd
import numpy as np
import toytree as tt
import toyplot

from plotly.express.colors import sample_colorscale
import plotly.express as px


tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]


palette = np.hstack([px.colors.colorbrewer.Pastel1,
                    px.colors.colorbrewer.Pastel2, px.colors.colorbrewer.Set3])
edge_color_dict = {
    "Proteobacteria": palette[11],
    "Acidobacteria": palette[14],
    "Bacteroidetes": palette[7],
    "Actinobacteria": palette[3],
    "Verrucomicrobia": palette[10],
    "Firmicutes": palette[2],
    "Crenarchaeota": palette[6],
    "Planctomycetes": palette[7],
    "Nitrospirae": palette[8],
    "Fusobacteria": palette[13],
    'dsDNA_viruses_no_RNA_stage': palette[1],
    "Spirochaetes": palette[0],
    "85": palette[5],
    "25": palette[10],
    "8": palette[15],
    "22": palette[16],
    "20": palette[17],
    "24": palette[9],
    "56": palette[9],
    "32": palette[9],
    "53": palette[9],
    "64": palette[9],
    "55": palette[9],
    "29": palette[9],
    "17": palette[9],
    "62": palette[9],
    "49": palette[9],
    "9": palette[9],
    "16": palette[9],
    "42": palette[9],
    "58": palette[9],
}


def create_node_df(label,
                   sep=";",
                   agg_level="Species",
                   tax_levels=['Kingdom', 'Phylum', 'Class',
                               'Order', 'Family', 'Genus', 'Species']):
    """

    """
    df = pd.DataFrame(label)
    node_df = df[0].str.split(sep, expand=True)
    node_df.columns = tax_levels
    node_df.index = node_df[agg_level].values

    node_df["Final Parameter"] = "lightgray"

    node_df[agg_level].replace('', np.nan, inplace=True)
    node_df.dropna(subset=[agg_level], inplace=True)

    # Phylum level is needed for colouring
    for level in tax_levels:
        node_df[level] = node_df[level].str.replace(r".__", "", regex=True)

    return node_df


def draw_tree(node_df,
              cfi_vals,
              agg_level="Species",
              eff_max=5,
              tip_labels=True,
              title="CFI per microbe",
              show_only_important=False,
              fontsize="32px"):
    node_df = node_df.reset_index()

    tree_dat = {}

    tree_dat[agg_level] = node_df

    # Make toytree object, initialize toyplot canvas
    tree, markers = build_fancy_tree(tree_dat, agg_level)

    # create colouring
    colors_neg = np.sort(cfi_vals[(cfi_vals < 0)] / cfi_vals.min())
    colors_pos = np.sort(cfi_vals[(cfi_vals >= 0)] / cfi_vals.max())
    # TODO : cnames = list(mcolors.TABLEAU_COLORS.keys())
    # ccodes = mcolors.TABLEAU_COLORS
    discrete_colors_1 = sample_colorscale('Oranges', colors_neg[::-1])
    discrete_colors_2 = sample_colorscale("Blues", colors_pos)
    discrete_colors = np.hstack([discrete_colors_1, discrete_colors_2])

    sorted_cfi_vals_index = np.argsort(cfi_vals)
    cfi_dict = dict(zip(
        node_df[agg_level].values[sorted_cfi_vals_index], np.arange(node_df.shape[0])))
    # cfi_dict = dict(zip(node_df[agg_level].values, sorted_cfi_vals_index))

    if show_only_important:
        list_highest_influence = []
        font_type = {"font-size": fontsize}
        show_title = False
        show_legend = False
        for n in tree.treenode.traverse():
            n.add_feature(pr_name="color", pr_value="lightgray")
            n.add_feature(pr_name="effect", pr_value=0.0)
            if n.name in cfi_dict.keys():
                n.add_feature(pr_name="color",
                              pr_value=discrete_colors[cfi_dict[n.name]])
                n.add_feature(pr_name="effect", pr_value=.5)
                if cfi_dict[n.name] > (len(cfi_vals) - 5):
                    n.add_feature(pr_name="effect",
                                  pr_value=cfi_dict[n.name]/(8*len(cfi_vals)))
                    n.add_feature(pr_name="color",
                                  pr_value=discrete_colors[cfi_dict[n.name]])
                    list_highest_influence.append(n.name)
                if cfi_dict[n.name] < 5:
                    n.add_feature(pr_name="effect",
                                  pr_value=(len(cfi_vals)
                                            - cfi_dict[n.name])/(8*len(cfi_vals)))
                    n.add_feature(pr_name="color",
                                  pr_value=discrete_colors[cfi_dict[n.name]])
                    list_highest_influence.append(n.name)
        # add attributes to the tree
        label_colors = [discrete_colors[cfi_dict[n]]
                        if n in cfi_dict.keys() and n in list_highest_influence else "rgb(255, 255, 255)" for n in
                        tree.get_tip_labels()]
        label_colors = [discrete_colors[cfi_dict[n]]
                        if n in cfi_dict.keys() else "rgb(0, 0, 0)" for n in
                        tree.get_tip_labels()]

    else:
        for n in tree.treenode.traverse():
            n.add_feature(pr_name="color", pr_value="lightgray")
            n.add_feature(pr_name="effect", pr_value=0.0)
            if n.name in cfi_dict.keys():
                if cfi_dict[n.name] > (len(cfi_vals) - 4):
                    n.add_feature(pr_name="effect",
                                  pr_value=cfi_dict[n.name]/(8*len(cfi_vals)))
                    n.add_feature(pr_name="color",
                                  pr_value=discrete_colors[cfi_dict[n.name]])
                if cfi_dict[n.name] < 4:
                    n.add_feature(pr_name="effect",
                                  pr_value=(len(cfi_vals)
                                            - cfi_dict[n.name])/(8*len(cfi_vals)))
                    n.add_feature(pr_name="color",
                                  pr_value=discrete_colors[cfi_dict[n.name]])
        font_type = {"font-size": "20px"}
        # add colouring
        label_colors = [discrete_colors[cfi_dict[n]]
                        if n in cfi_dict.keys() else "rgb(0, 0, 0)" for n in
                        tree.get_tip_labels()]
        show_legend = True
        show_title = True

    canvas = toyplot.Canvas(width=1000, height=1000)

    # Area for tree plot
    # ax0 = canvas.cartesian(bounds=(50, 800, 50, 800), padding=0)
    ax0 = canvas.cartesian(bounds=(0, 1000, 0, 1000), padding=0)
    ax0.x.show = False
    ax0.y.show = False

    # Determine max effect size, if necessary
    if eff_max is None:
        eff_max = np.max([np.abs(n.effect) for n in tree.treenode.traverse()])

    tree.draw(
        axes=ax0,
        layout='c',  # circular layout
        edge_type='p',  # rectangular edges
        node_sizes=[(np.abs(n.effect) * 100 / eff_max) + 10
                    if n.effect != 0 else 0.0 for n in
                    tree.treenode.traverse()],
        # node color from node feature "color"
        node_colors=[n.color
                     for n in tree.treenode.traverse()],
        node_style={
            "stroke": "lightgray",
            "stroke-width": "1",
        },
        node_markers="s",
        width=800,
        height=800,
        tip_labels=tip_labels,  # Print tip labels or not
        tip_labels_align=True,
        tip_labels_style=font_type,
        tip_labels_colors=label_colors,
        edge_colors=[tree.idx_dict[x[1]].edge_color for x in tree.get_edges()],
        edge_widths=3  # width of tree edges
    )

    # add area for plot title
    if show_title:
        ax1 = canvas.cartesian(bounds=(50, 800, 50, 100),
                               padding=0, label=title)
        ax1.x.show = False
        ax1.y.show = False

    if show_legend:
        # add legend for phylum colors
        canvas.legend(markers,
                      bounds=(0, 100, 50, 200),
                      label="Phylum"
                      )

        markers2 = [
            ("Negative CFI",
             toyplot.marker.create(shape="o", size=20, mstyle={"fill": discrete_colors_2[20]})),
            ("Positive CFI",
             toyplot.marker.create(shape="o", size=20, mstyle={"fill": discrete_colors_1[20]})),
        ]
        canvas.legend(markers2,
                      bounds=(800, 1000, 50, 200),
                      label="Directional Influence CFI"
                      )
    # save plot if desired
    return canvas


def get_phylo_levels(results, level, col="Cell Type"):
    """Get a taxonomy table (columns are "Kingdom", "Phylum", ...) from a
    DataFrame where the one column contains full taxon names.

    :param results: pandas DataFrame

        One column must be strings of the form
        "<Kingdom>*<Phylum>*...",
        e.g. "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*
              Rikenellaceae*Alistipes"

    :param level: string

        Lowest taxonomic level (from ["Kingdom", "Phylum", "Class",
        "Order", "Family", "Genus"]) that should be included

    :param col: string

        Name of the column with full taxon names

    :return:

    DataFrame with columns ["Kingdom", "Phylum", "Class", "Order",
    "Family", "Genus"]

    """

    max_level_id = tax_levels.index(level)+1
    cols = tax_levels

    tax_table = pd.DataFrame(columns=cols, index=np.arange(len(results)))
    for i in range(len(results)):
        char = results.loc[i, col]
        split = char.split(sep="*")
        for j in range(max_level_id):
            try:
                tax_table.iloc[i, j] = split[j]
            except IndexError:
                tax_table.iloc[i, j] = None

    return tax_table


def traverse(df_, a, i, innerl):
    """
    Helper function for df2newick
    :param df_:
    :param a:
    :param i:
    :param innerl:
    :return:
    """
    if i+1 < df_.shape[1]:
        a_inner = pd.unique(
            df_.loc[np.where(df_.iloc[:, i] == a)].iloc[:, i+1])

        desc = []
        for b in a_inner:
            desc.append(traverse(df_, b, i+1, innerl))
        if innerl:
            il = a
        else:
            il = ""
        out = f"({','.join(desc)}){il}"
    else:
        out = a

    return out


def df2newick(df, agg_level, inner_label=True):
    """Converts a taxonomy DataFrame into a Newick string

    :param df: DataFrame

        Must have columns ["Kingdom", "Phylum", "Class", "Order",
        "Family", "Genus"]

    :param inner_label: Boolean

        If True, internal nodes in the tree will keep their respective
        names

    :return: Newick string

    """
    if agg_level == "Phylum":
        df = df.drop(columns=["Class", "Order", "Family", "Genus"])
    if agg_level == "Class":
        df = df.drop(columns=["Order", "Family", "Genus"])
    if agg_level == "Order":
        df = df.drop(columns=["Family", "Genus"])
    if agg_level == "Family":
        df = df.drop(columns=["Genus"])

    tax_levels = [col for col in df.columns
                  if col in ["Kingdom", "Phylum", "Class",
                             "Order", "Family", "Genus", "Species"]]
    df_tax = df.loc[:, tax_levels]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick


def build_fancy_tree(data, agg_level):
    """Make a toytree object with all kinds of extras, like edge colors,
    effect sizes, ...

    :param data: Dictionary of DataFrames.

        Contains effect sizes, taxonomy info, etc. for all taxonomic
        ranks Dictionary keys should be ["Phylum", "Class", "Order",
        "Family", "Genus"] Each DataFrame should have the columns
        "Final Parameter" (= effect size), as well as columns for all
        taxonomic ranks

    :return:
        toytree object

    """

    # Get genus-level data for building a complete tree
    data_gen = data[agg_level]
    # Combine results from all levels into one df
    data_all = pd.concat(data.values())

    # Get newick string and build toytree with no extras
    newick = df2newick(data_gen, agg_level)
    tree = tt.tree(newick, tree_format=1)

    # Edge colors.  The color dictionary is hard-coded, keys must be
    # the names of all phyla in the data
    # TODO : move that out

    # marker objects for plotting a legend
    markers = []

    # Height of the colored level (here: 2nd highest)
    max_height = np.max([n.height - 2 for n in tree.treenode.traverse()])

    # Iterate over all tree nodes and assign edge colors
    c = 0
    for n in tree.treenode.traverse():
        # If node corresponds to the colored level (here: Phylum),
        # determine color and assign it as feature "edge_color" to all
        # descendants
        if n.height == max_height:
            col = edge_color_dict[n.name]
            n.add_feature("edge_color", col)
            for n_ in n.get_descendants():
                n_.add_feature("edge_color", col)

            # Also add a marker for the legend
            # col2 = '#%02x%02x%02x' % tuple([int(255*x)
            #                                for x in col.tolist()[:-1]])
            col2 = col
            m = toyplot.marker.create(shape="o", size=8, mstyle={"fill": col2})
            markers.append((n.name, m))

            c += 1
        # For all levels above the colored level, assign edge color black
        elif n.height > max_height:
            n.add_feature("edge_color", "black")

    # assign taxonomic levels to nodes (e.g. "Genus" for a node on the
    # lowest level)
    for n in tree.treenode.traverse():
        if n.height == "":
            ll = tax_levels[-1]
        elif n.height >= len(tax_levels):
            ll = ""
        else:
            ll = tax_levels[-(int(n.height) + 1)]
        n.add_feature("tax_level", ll)

    # add effects to each node as feature "effect": For all results,
    # add the taxonomic rank (forgot to do that when combining
    # initially)
    data_all["level"] = "Kingdom"
    for ll in tax_levels[1:]:
        data_all.loc[pd.notna(data_all[ll]), "level"] = ll

    return tree, markers
