'''
sc2enrich.py - enrichment analysis from single cell data
====================================================

:Author: Mike Morgan
:Release: $Id$
:Date: |today|
:Tags: Python

Purpose
-------

.. Enrichment of genes within functional categories of interest is a
common task in genomics analysis.  Single cell transcriptomics suffers
from many zero observations and potentially unequal sample sizes.

This script allows the calculation of empirical enrichment/depletion
p-values using permutation approaches.  It does this by downsampling
to the smallest sample size, and generates a null distribution, matching
genes on expression from a background set.

The output is a table with the enrichment category size, and two empirical
 p-values, one for depletion and one for enrichment.  The p-values are
limited by the number of permutations run, i.e. 100 permutations will give
the smallest possible p-value as 0.01, 1000 permutations would be 0.001, etc.

There is also an option to generate histograms for all categories and
null distributions to check for artifacts.

Inputs
-----------

Expression table - gene expression table, with genes as rows and cells
  as columns::

            cell1    cell2    cell3 ...
    gene1     .        .        .
    gene2     .        .        .
    gene3     .        .        .
      .
      .
      .

Category table - table mapping genes onto categories for testing.  Assumes
  two columns in the table labelled "Category" and "Gene"
  e.g.::

    Gene    Category
    gene1   cat1 
    gene2   cat2
    gene1   cat3
    gene3   cat4
      .      .
      .      .
      .      .


Options
-------
  `--permutations` - number of permutations to run to generate the
  null distribution for each category

  `--sample-size` - number of cells to downsample, or the proportion
  of the total dataset to down sample to. i.e. if 0.5 = 50% of cells

  `--gene-list` - a file containing genes of interest to test for
  enrichment of categories.  One gene per line

  `--gene-categories` - file containing the category table, mapping
  genes to categories

  `--min-category-size` - minimum number of genes in a category
  to test for enrichment against.  This is useful for filtering
  out categories where there is very little power to detect an
  enrichment/depletion, but also where the distribution may
  essentially just be 0's and 1's.
  


Usage
-----

.. Example use case

Example::

   python sc2enrich.py

Type::

   python sc2enrich.py --help

for command line help.

Command line options
--------------------

'''

# TODO
# repeated dictionary lookups/get_item calls
# are expensive.
# is there an alternative to permutation, i.e.
# a distribution that adequately describes the data?

import sys
import CGAT.Experiment as E
import scEnrichment as scEn
import pandas as pd
import numpy as np
import scipy.stats as stats
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
import random
import collections
import cProfile
import functools


def matchGenes(foreground_expression, background_expression):
    '''
    Create a matched set of background genes, based on the
    expression of the foreground genes

    Arguments
    ---------
    foreground_expression: pandas.DataFrame
      dataframe containing gene expression values of
      the foreground genes,
      with genes as rows (index) and cells as columns

    background_expression: pandas.DataFrame
      dataframe containing gene expression values of
      the background genes, with genes as rows (index)
      and cells as columns

    Outputs
    -------
    background_match: set
      a set of gene IDs that are matched to the
      foreground
    '''

    fore_av = foreground_expression.mean(axis=1)
    back_av = background_expression.mean(axis=1)

    # set up a dictionary with genes binned by expression
    # as a potential matched background genes
    max_exprs = np.max(background_expression.iloc[:, :-1].max(axis=1))
    all_bins = np.linspace(0, np.ceil(max_exprs), 18)
    exprs_bin = collections.defaultdict(list)

    # 18 is the magic number to keep values within 1 log2 value
    # of eachother for the observed range of values, i.e. 0 - ~16
    for ibin in all_bins:
        exprs_bin[ibin] = []

    exprs_keys = exprs_bin.keys()

    # bin all of the background genes based on expression
    for bx in back_av.index:
        gene_min = back_av[bx] - 0.5

        # find the bin(s) within that range
        vbin = min([xi for xi in exprs_keys if xi >= gene_min])
        exprs_bin[vbin].append(bx)

    # select a set of background genes with similar expression
    # to the test genes
    background_match = set()

    for fg in fore_av.index:
        fgene_min = fore_av[fg] - 0.5

        # alway want the minimum bin value which the gene
        # falls into, don't need max value
        fbin = min([fi for fi in exprs_keys if fi >= fgene_min])

        # randomly subsample a gene from that bin
        bin_size = len(exprs_bin[fbin])
        mi = random.randint(0, bin_size - 1)
        match_gene = exprs_bin[fbin][mi]
        match_bin = exprs_bin[fbin]

        # if the gene is already in the background set then
        # find another one at random <- this might falsely
        # generate multiple of the same gene anyway
        # what is the probability of picking a gene already
        # in the background set?  Will be depdendent on how
        # big it is.  Prob will increase as background set
        # gets bigger, need to find a way around this
        # pop each matched gene from the background, so can't
        # match again?

        if match_gene in background_match:
            mi = random.randint(0, bin_size - 1)
            match_gene = match_bin[mi]

        background_match.add(match_gene)

    # E.info("{} background genes matched for "
    #        "{} foreground genes".format(len(background_match),
    #                                     len(fore_av.index)))
    return background_match


def generateNulls(express_table, cat_table, nperm, gene_list,
                  sample_size, categories, cat_index):
    '''
    Generate a null distribution for each functional category
    of interest, downsampling and matching genes based on
    expression level.

    Arguments
    ---------
    express_table: pandas.DataFrame
      a table of gene expression values, with genes as rows
      and cells as columns

    cat_table: pandas.DataFrame
      a mapping of genes to categories for testing. Assumes
      column headers are `Gene` and `Category`.

    nperm: int
      number of permutations to run for each category to
      generate the null distribution for testing

    gene_list: list
      list of genes of interest

    sample_size: float
      number of cells, or proportion of cells,
      to downsample to when generating null and matching on
      gene expression

    categories: list
      list of sorted unique categories agaisnt which to test enrichment

    cat_index: list
      list of indexes of sorted categories

    Returns
    -------
    null_dist: dict
      dictionary of categories with frequency arrays
      describing the null distribution for each
    '''

    if sample_size < 1:
        sample_size = np.floor(express_table.shape[1] * sample_size)
    else:
        pass

    E.info("Downsampling expression table to {} cells".format(sample_size))
    total_cells = express_table.shape[1] - 1

    array_size = (len(cat_index), nperm)
    # use an array of arrays instead?
    null_dist = np.ndarray(shape=array_size,
                           dtype=np.uint64)
    # null_dist = dict(zip(categories,
    #                      [np.zeros(nperm) for px in categories]))

    # select background genes, anything not in the gene list
    # but in the expression table
    background_genes = set(express_table.index).difference(gene_list)

    E.info("Matching background genes on expression")
    is_exprs = lambda x: x.mean > 1.0

    E.info("Running {} permutations".format(nperm))
    # make categories once, rather than calling null_dist.keys()
    # on each permutation, might reduce the number of get_item calls
    for p in xrange(nperm):
        if not p % 10:
            E.info("{} permutations run".format(p))
        # randomly subsample cells 
        cell_indx = set()
        while len(cell_indx) < sample_size:
            cell_indx.add(random.randint(0, total_cells - 1))

        # generate expression tables, remove genes that 
        # are not expressed - av log2(exprs) > 1?
        df = express_table.iloc[:, [cx for cx in cell_indx]]
        geneset_exprs = df.loc[gene_list]
        fore_is_exprs = geneset_exprs.apply(is_exprs, axis=1)
        geneset_exprs = geneset_exprs.loc[fore_is_exprs]

        back_exprs = df.loc[background_genes]
        back_is_exprs = back_exprs.apply(is_exprs, axis=1)
        back_exprs = back_exprs.loc[back_is_exprs]

        matched = matchGenes(geneset_exprs,
                             back_exprs)

        # test each category
        # use functools.partial and map to perform
        # faster iteration over categories <- becomes
        # a loop in C, not much faster
        cat_wrap = functools.partial(calculateOverlaps, 
                                     categories=categories,
                                     cat_table=cat_table,
                                     matched=matched,
                                     null_dist=null_dist,
                                     p=p)
        map(cat_wrap, cat_index)

    return null_dist


def calculateOverlaps(cats, categories, cat_table, matched,
                      null_dist, p):
    '''
    Calculate the overlap between the matched genes and
    the genes for each category.

    Overlap is assigned to the null_dist array on each
    iteration call.

    Arguments
    ---------
    cats: int
      index of category to test for overlap

    categories: list
      categories to test for overlap against

    cat_table: pandas.DataFrame
      table containing category: gene mapping

    matched: set
      background geneset matched for expression
      with foreground genes

    null_dist: numpy.ndarray
      array containing overlap sizes for each
      permutation and category

    p: int
      permutation number, used to index null_dist

    Returns
    -------
    null_dist: numpy.ndarray
      updated array with overlap between background
      and tested category
    '''

    cat = categories[cats]
    cat_genes = cat_table[cat_table["Category"] == cat].index
    overlap = len(matched.intersection(cat_genes))
    null_dist[cats, p] = overlap

    return null_dist


def main(argv=None):
    """script main.
    parses command line options in sys.argv, unless *argv* is given.
    """

    if argv is None:
        argv = sys.argv

    # setup command line parser
    parser = E.OptionParser(version="%prog version: $Id$",
                            usage=globals()["__doc__"])

    parser.add_option("--permutations", dest="perms", type="int",
                      help="number of permutations to use to generate"
                      " the null distribution for each category")

    parser.add_option("--sample-size", dest="sample_size", type="float",
                      help="number of cells to select for downsampling")

    parser.add_option("--gene-list", dest="gene_list", type="string",
                      help="list of genes of interest to test for enrichment")

    parser.add_option("--gene-categories", dest="gene_cats", type="string",
                      help="table with each gene and a functional category "
                      "to which it is assigned.  A gene may be assigned to "
                      "multiple categories")

    parser.add_option("--min-category-size", dest="min_cat_size", type="int",
                      help="minimum size for a category to test for enrichment/"
                      "depletion of gene list against")    

    # add common options (-h/--help, ...) and parse command line
    (options, args) = E.Start(parser, argv=argv)

    parser.set_defaults(perms=1000,
                        min_cat_size=1,
                        seed=86136)

    # read in expression table first
    infile = argv[-1]

    expression_table = pd.read_table(infile, sep="\t", header=0,
                                     index_col=0)

    # assume data are expression rates, and not on a log scale
    E.info("Transforming expression data onto log2 scale")
    log_expression = np.log2(expression_table + 1)

    E.info("Reading gene list")
    with open(options.gene_list, "r") as gfile:
        gene_list = [x.rstrip("\n") for x in gfile.readlines()]

    # category table needs genes as the index
    # column order should be genes then cats
    # if there is a 1:1 mapping, else it should be
    # a dictionary of categories: genes
    E.info("Parsing gene: category mapping file")
    cat_table = pd.read_table(options.gene_cats,
                              sep="\t", header=0,
                              index_col=None)
    cat_table.index = cat_table["Gene"]

    # generate a dictionary of frequency arrays
    # containing the null distributions
    E.info("Generating null distributions")

    # only run on categories with a minimum size?
    cat_sizes = cat_table["Category"].value_counts()
    keep_cats =  cat_sizes[cat_sizes >= options.min_cat_size].index

    E.info("{} categories contain more "
           "than {} genes".format(len(keep_cats),
                                  options.min_cat_size))
    categories = [cx for cx in set(cat_table["Category"]) if cx in keep_cats]
    cat_indx = [ic for ic, iy in enumerate(categories)]

    null_dist = generateNulls(express_table=log_expression,
                              cat_table=cat_table,
                              nperm=options.perms,
                              gene_list=gene_list,
                              sample_size=options.sample_size,
                              categories=categories,
                              cat_index=cat_indx)

    test_vals = dict(zip(categories,
                         np.zeros(len(categories))))

    for cat in test_vals.keys():
        cat_genes = cat_table[cat_table["Category"] == cat].index
        inter = len(set(gene_list).intersection(cat_genes))
        test_vals[cat] = inter

    # calculate p-values for each tail of the distribution,
    # i.e. enrichment and depletion
    pval_dict = dict(zip(categories,
                         np.zeros(len(categories))))

    E.info("Testing statistical significance of overlap between "
           "null and gene list")

    # calculate p-values, median of null and median absolute deviation of the null
    for ecat in cat_indx:
        cat = categories[ecat]
        right_p = 1 - sum([1 for vi in sorted(null_dist[ecat, :]) if vi < test_vals[cat]])/float(options.perms)
        left_p = 1 - sum([1 for li in sorted(null_dist[ecat, :]) if li > test_vals[cat]])/float(options.perms)
        overlap_cat = int(test_vals[cat])
        median_null = np.median(null_dist[ecat, :])
        # 0.67448978501 is a normalization constant from a standard
        # normal distribution.  Assumes median are drawn from a standard normal
        # how robust is this?
        mad = np.median(abs(null_dist[ecat, :] - median_null))/0.6744897501
        pval_dict[cat] = (left_p, right_p, overlap_cat, median_null, mad)

    pval_df = pd.DataFrame(pval_dict).T
    pval_df.columns = ["P_deplete", "P_enrich", "GenesInCat",
                       "MedianNullInCat", "MedianAbsolutDeviation"]

    # add category size information
    pval_df.loc[:, "CategorySize"] = cat_sizes

    # set p=0 to minimum based on number of permutations,
    # i.e. if n=100, then min p=0.01, n=1000, p=0.001, etc
    min_p = 1.0/options.perms
    E.info("Minimum p-value set to {}".format(min_p))
    pval_df.loc[pval_df["P_enrich"] < min_p, "P_enrich"] = min_p
    pval_df.loc[pval_df["P_deplete"] < min_p, "P_deplete"] = min_p

    pval_df.to_csv(options.stdout, sep="\t", index_label="Category")

    # write footer and output benchmark information.
    E.Stop()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
