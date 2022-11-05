import unifrac
import skbio
import numpy as np
import pandas as pd
from newick import loads, write
from biom.table import Table
from biom.util import biom_open


def compute_unifrac(comp_lbl, biom_file, tree_file,
                    set_branch_length=True,
                    branch_length=1.0,
                    levels=["Kingdom", "Phylum", "Class",
                            "Order", "Family", "Genus", "Species"],
                    delim=";",
                    table_id="Dataset"
                    ):
    """
    Computation of unweighted unifrac distance

    Parameters
    ----------
    biom_file
    tree_file

    Returns
    -------

    """
    # generate tree and tree file
    tree, df = generate_tree_file(comp_lbl,
                                  filename=tree_file,
                                  set_branch_length=set_branch_length,
                                  branch_length=branch_length,
                                  levels=levels,
                                  delim=delim)

    # generate biom dataset and biom file
    table = generate_biom_file(tree, df, biom_file, table_id=table_id)

    # compute unifrac distance
    table_unifrac = unifrac.unweighted(biom_file, tree_file)

    res = {"BiomTable": table,
           "Tree": tree,
           "TreeData": df,
           "UnifracTable": table_unifrac}

    return res


def generate_tree_file(comp_lbl,
                       filename,
                       set_branch_length=True,
                       branch_length=1.0,
                       levels=["Kingdom", "Phylum", "Class",
                               "Order", "Family", "Genus", "Species"],
                       delim=";"):
    """
    Generate tree file for computation of unifrac distance

    Parameters
    ----------
    comp_lbl
    set_branch_length
    branch_length
    filename
    levels

    Returns
    -------
    tree
    df
    """

    comp_lbl_labels = comp_lbl

    ar = pd.DataFrame([x.split(delim)
                      for x in list(comp_lbl_labels)], columns=levels)
    # ar.iloc[:, -1].fillna(levels[-1], inplace=True)

    ar.fillna(method="ffill", axis=1, inplace=True)

    if ar.iloc[:, -1].duplicated().sum() != 0:
        print("Create dummy otu")
        # generate labels to be sure that we have unique node names
        labels = np.array(['%d' % i for i in range(len(comp_lbl))])
        comp_lbl_labels = np.array(
            [comp_lbl[i] + "_" + labels[i] for i in range(len(labels))])

    # remove potential white spaces
    comp_lbl_labels = [item.replace(" ", "") for item in comp_lbl_labels]
    comp_lbl_labels = [item.replace("(", "") for item in comp_lbl_labels]
    comp_lbl_labels = [item.replace(")", "") for item in comp_lbl_labels]
    comp_lbl_labels = [item.replace("[", "") for item in comp_lbl_labels]
    comp_lbl_labels = [item.replace("]", "") for item in comp_lbl_labels]

    # convert labels to dataframe for conversion to newick file
    ar.iloc[:, -1] = [x.split(delim)[-1] for x in list(comp_lbl_labels)]
    df = ar

    tree_str = df2newick(df, levels)
    tree_str = tree_str.replace(" ", "")
    # generate newick tree from string
    trees = loads(tree_str)

    # save generated tree to file
    write(trees, filename)

    # reload tree
    tree = skbio.TreeNode.read(str(filename), convert_underscores=False)

    # set weight to individual branches
    if set_branch_length:
        for n in tree.traverse():
            n.length = branch_length

    # save final tree
    tree.write(filename)

    return tree, df


def generate_biom_file(tree, df, filename, table_id="Dataset"):
    """
    Generate Biom File which contains only one single microbiota per sample

    Parameters
    ----------
    tree
    df
    filename
    table_id

    Returns
    -------
    table
    """

    # generate tip labels
    tree_tips = list(df.iloc[:, -1].values)

    # generate Sample Dataset
    data1sample = np.identity(len(tree_tips))
    sample_ids = ['S%d' % i for i in range(len(tree_tips))]

    observ_ids = tree_tips
    observ_metadata = [
        {'taxonomy': list(df.iloc[i, :].values)}
        for i in range(len(observ_ids))]

    # generate Biom Table
    table = Table(data1sample, observ_ids, sample_ids, observ_metadata)
    table.table_id = table_id

    # save Biom Table
    with biom_open(filename, "w") as f:
        table.to_hdf5(f, generated_by="1sample")

    return table


def df2newick(df, levels, inner_label=True):
    df_tax = df.loc[:, [x for x in levels if x in df.columns]]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick


def traverse(df_, a, i, innerl):
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
