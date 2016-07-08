'''
scEnrichment.py is a module for testing enrichment of categories for a geneset
derived from single cell experiments.

Currently it tests for enrichment of a gene set over categories by
downsampling and generating a null distribution of expectation by
permutation.  Probabilities of rejecting the null hypothesis are empirical.

'''

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
