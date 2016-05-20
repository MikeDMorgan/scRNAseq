'''
SCRNA.py is a module file for single cell RNA seq analysis functions.

Functions
---------


Classes
-------

SingleCell: object
This is the principle object in this module.  It holds single cell
expression data in multiple forms, transform data, performs analysis
and plots data and results.

TODO:
* wrapper for BASiCs
* Brennecke et al technical noise estimation
* normalization in the absence of spike-in method
* ASE accounting for technical noise?
* plotting and visualisation - gene and cell level
* handle R classes?
* annotation enrichment analysis
* plot generation
'''

import sys
import rpy2.robjects as ro
from rpy2.robjects import r as R
import rpy2.robjects.pandas2ri as py2ri
import re
import pandas as pd
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

class SingleCell(object):
    '''
    A container class for a single cell data set.
    It allows access to multiple data tables, perform
    processing operations and analyses on it.  There are
    methods for plotting and visualisation of data.

    Its primary design is for programmatic, rather than
    interactive, analysis.  I intend to develop it
    further for more interactive use inspired by
    the bioconductor package scater

    Atributes
    ---------
    `__name__`: name of the single cell object.  Required to
    save and retrieve data stored on disk. `name` must
    be set on instantiation.

    `design_table`: table of cells and metadata attached
    to each.

    `counts_table`: read counts associated with each
    gene/genomic feature.  Must be integer counts.

    `transcript_expression`: transcript-level expression
    estimates, e.g. FPKM, TPM

    `parameter_estimates`: parameters associated with the
    data, namely from Bayesian hierarchical model, BASiCs

    `spikein_table`: read counts for spike in transcripts

    `spikein_dilution`: dilution factor used for spike in
    transcript RNAs during sample preparation

    `annotation_table`: gene/feature-level annotations used
    for enrichment analysis, e.g. GO annotations    

    Methods
    -------
    `summary_statistics`: compute summary statistics over
    input data values

    `estimate_technical_noise`: Use the method of Brennecke
    et al to estimate technical noise and highly
    variable genes

    `normalize`: normalize expression/read count data when
    spike in transcripts are not present

    `compute_parameters`: apply BASiCs to estimate gene and
    cell-level parameters using MCMC.

    `differential_testing`: use BASiCs to calculate
    differentially expressed and overdispersed genes

    `dimensionality_reduction`: apply dimensionality
    reduction method(s) to data table(s)

    `latent_variables`: compute latent variables using
    method of Buettner et al to estimate contribution
    of latent factors to variance components. Requires
    technical noise estimation.

    `annotation_enrichment`: perform annotation enrichment
    analysis, accounting for cell and gene sample sizes.
    Uses permutation by default.

    `allele_specific_expression`: estimate allele specific
    expression for genes, accounting for technical noise.

    `plot_dimensions`: plot results of dimensionality
    reduction method(s).

    `plot_heatmap`: generate gene-level heatmap(s) for
    defined gene set(s).

    Accessors
    ---------
    These accessors are used in prefernce to directly accessing
    the attributes and variables themselves.

    `get_expression`: getter method for transcript expression data

    `get_counts`: getter method for counts data

    `get_metadata`: retrieve metadata

    `get_parameters`: retrieve model parameters

    `get_spikeins`: `getter method for spike in transcript expression

    `get_annotations`: getter method for gene-level annotations
    '''

    def __init__(self, name=None, **kwargs):

        if name:
            self.__name__ = name
        else:
            self.__name__ = id(self)
        # dynamically set attributes upon instantiation
        for attr in kwargs.keys():
            setattr(self, attr, kwargs[attr])

    # ---------------------------------------- #
    # attribute getters and setters            #
    # ---------------------------------------- #

    def set_attribute(self, attribute, value):
        '''
        set an attribute with a value
        '''

        setattr(self, attribute, value)

    def add_counts_table(self, counts_matrix=None,
                         gene_ids=None, cell_ids=None,
                         sep="\t", compression=None,
                         cell_qc=None, remove_zeros=True,
                         exprs_threshold=5,
                         rename_genes=True, rename_cells=True):
        '''
        Assign a counts matrix to the class instance,
        or read in a file of counts, with genes as rows
        and cells as columns.  Optional QC can be performed
        at this stage, assuming a list of input cells
        to remove.

        Poor quality cells should be removed before
        0-expression genes/features.  This function will
        generate a warning if zero-expressed genes are
        removed without first poor quality cells being
        removed.

        Arguments
        ---------
        counts_matrix: numpy.ndarray/pandas.core.frame.DataFrame
        table of gene/feature counts as rows, cells as
        columns.  If a numpy array is provided the
        gene name and cell IDs must also be specified.

        gene_ids: list/tuple
        list or tuple of gene IDs of the same length
        as the array dimension 0 (rows).  Only relevant
        if the `counts_matrix` is an numpy array.  It
        can be used to rename the `counts_matrix` rows
        if it is type <pandas.core.frame.DataFrame> and
        `rename_genes` is True.

        cell_ids: list/tuple
        list or tuple of cell IDs of the same length
        as the array dimension 1 (columns).  This option
        is only really relevant if the `counts_matrix` is
        an numpy array.  It can be used to rename the columns
        of the `counts_matrix` if it is a type 
        <pandas.core.frame.DataFrame> and `rename_cells` is True.

        sep: string
        separator used to parse the counts table file

        compression: string
        compression algorithm used to properly parse the
        counts table

        cell_qc: list
        list containing cell IDs to remove from the counts
        table.  If the list contents are all integers, these
        will be assumed indexes to remove
        
        remove_zeros: boolean
        remove zero-expression genes.  If this is specified
        without `cell_qc` a warning message will be generated.

        exprs_threshold: int
        threshold below which to mask out genes as not expressed

        rename_genes: boolean
        whether to replace the row identifiers with the input
        list of `gene_ids` when `counts_matrix` is of type
        <pandas.core.frame.DataFrame>.

        rename_cells: boolean
        whether to replace the column identifiers with the
        input list of `cell_ids` when `counts_matrix` is
        of type <pandas.core.frame.DataFrame>.        

        Returns
        -------
        None - no value is returned. The `counts_matrix`
        is assigned to the attribute `counts_table`, which
        can be retrieved with the class method `get_counts()`.
        '''

        # first check if it's a file path, matrix or
        # pandas dataframe

        # assume index and headers are set
        if type(counts_matrix) == str:
            if os.path.isabs(counts_matrix):
                pass
            else:
                counts_matrix = os.path.abspath(counts_matrix)

            counts = pd.read_table(counts_matrix,
                                   sep=sep,
                                   index_col=0,
                                   header=0,
                                   compression=compression)

            logging.info("Counts table read. Genes detected {} "
                         "Cells detected {}.".format(counts.shape[0],
                                                     counts.shape[1]))

        elif type(counts_matrix) == pd.core.frame.DataFrame:
            counts = counts_matrix
            if rename_genes and gene_ids:
                # check length of genes and rows
                if len(gene_ids) == len(counts.index):
                    counts.index = gene_ids
                    logging.info("Renaming gene IDs")
                else:
                    logging.warn("New Gene IDs (length {})and `counts_matrix` "
                                 "rows (length {}) are not "
                                 "the same length.".format(len(gene_ids),
                                                           len(counts.index)))
            else:
                pass

            if rename_cells and cell_ids:
                if len(cell_ids) == len(counts.columns):
                    counts.columns = cell_ids
                    logging.info("Renaming cell IDs")
                else:
                    logging.warn("New cell IDs (length {}) and `counts_matrix` "
                                 "columns (length {}) are not the "
                                 "same length ".format(len(cell_ids),
                                                       len(counts.columns)))
            else:
                pass

            logging.info("Counts table read. Genes detected {} "
                         "Cells detected {}.".format(counts.shape[0],
                                                     counts.shape[1]))


        elif type(counts_matrix) == 'numpy.ndarray':
            # convert the numpy array to a pandas dataframe
            # to make QC easier
            counts = pd.DataFrame(counts_matrix)
            try:
                assert cell_ids
                counts.columns = cell_ids
            except AssertionError:
                logging.warn("No cell IDs have been provided "
                             "with an input `counts_matrix` "
                             "of type {}".format(type(counts_matrix)))
            try:
                assert gene_ids
                counts.index = gene_ids
            except AssertionError:
                logging.warn("No gene IDs have been provided "
                             "with an input `counts_matrix` "
                             "of type {}".format(type(counts_matrix)))

            logging.info("Counts table read. Genes detected {} "
                         "Cells detected {}.".format(counts.shape[0],
                                                     counts.shape[1]))


        elif type(counts_matrix) == 'file':
            counts = pd.read_table(counts_matrix,
                                   sep=sep,
                                   index_col=0,
                                   header=0,
                                   compression=compression)

            logging.info("Counts table read. Genes detected {} "
                         "Cells detected {}.".format(counts.shape[0],
                                                     counts.shape[1]))

        else:
            raise AttributeError("`counts_matrix` type not recognized. "
                                 "Must be either a pandas dataframe, "
                                 "numpy array, open file object or "
                                 "a path to a file containing "
                                 "the counts table.")

        if cell_qc:
            # test if the cell IDs look like indices
            # if str can they be converted to ints
            # without raising an error?
            try:
                int_indx = [int(qx) for qx in cell_qc]
            except ValueError:
                # contains strings that don't look like indices!
                int_indx = None
            logging.info("removing cell IDs: {}".format(",".join(cell_qc)))
                
            if int_indx:
                indx_in = [qi for qi, qy in enumerate(counts.columns) if qi not in int_indx]
                counts_qc = counts.iloc[:, indx_in]
            else:
                cell_in = [ci for ci in counts.columns if ci not in cell_qc]
                counts_qc = counts.loc[:, cell_in]

        if remove_zeros:
            # base it on av counts >= 5, default value
            try:
                assert counts_qc.shape
                notzero = counts_qc.apply(lambda x: x.mean() >= exprs_threshold,
                                          axis=1)
                counts_qc = counts_qc.loc[notzero]

            except UnboundLocalError:
                logging.warn("Poor quality cells may not have been removed. "
                             "Make sure bad quality cells are removed before "
                             "filtering out non-expressed genes.")

                notzero = counts.apply(lambda x: x.mean() >= exprs_threshold,
                                       axis=1)
                counts_qc = counts.loc[notzero]

        try:
            # you can't test the presence of a dataframe,
            # but you can test for the existence of it's
            # attributes
            assert counts_qc.shape
            self.counts_qc = counts_qc
            logging.info("Attaching QC'd counts table")
        except UnboundLocalError:
            pass

        self.counts_table = counts

    def counts_table_qc(self, cell_qc=None,
                        exprs_threshold=5):
        '''
        Perform QC on the raw counts table:
        remove poor quality cells and non-expressed
        genes

        Arguments
        ---------
        cell_qc: list
        list containing cell IDs to remove from the counts
        table.  If the list contents are all integers, these
        will be assumed indexes to remove

        exprs_threshold: int
        threshold below which to mask out genes as not expressed

        Returns
        -------
        None - assigns the QC counts table to the attribute
        `counts_qc`.
        '''

        # test counts table has been set first
        try:
            assert self.counts_table.shape 
        except:
            raise AttributeError("Counts table is missing. "
                                 "Please load a counts table "
                                 "before proceeding")

        counts = self.counts_table

        if cell_qc:
            # test if the cell IDs look like indices
            # if str can they be converted to ints
            # without raising an error?
            try:
                int_indx = [int(qx) for qx in cell_qc]
            except ValueError:
                # contains strings that don't look like indices!
                int_indx = None
            logging.info("removing cell IDs: {}".format(",".join(cell_qc)))
                
            if int_indx:
                indx_in = [qi for qi, qy in enumerate(counts.columns) if qi not in int_indx]
                counts_qc = counts.iloc[:, indx_in]
            else:
                cell_in = [ci for ci in counts.columns if ci not in cell_qc]
                counts_qc = counts.loc[:, cell_in]

        # base it on av counts >= 5, default value
        try:
            assert counts_qc.shape
            notzero = counts_qc.apply(lambda x: x.mean() >= exprs_threshold,
                                      axis=1)
            counts_qc = counts_qc.loc[notzero]

        except UnboundLocalError:
            logging.warn("Poor quality cells may not have been removed. "
                         "Make sure bad quality cells are removed before "
                         "filtering out non-expressed genes.")

            notzero = counts.apply(lambda x: x.mean() >= exprs_threshold,
                                   axis=1)
            counts_qc = counts.loc[notzero]

        try:
            assert counts_qc.shape
            self.counts_qc = counts_qc
            logging.info("Attaching QC'd counts table")
        except UnboundLocalError:
            pass

        self.counts_qc = counts_qc

    def get_counts(self):
        '''
        Retrieve the expression table,
        if it has been set
        '''

        try:
            return getattr(self, "counts_table")
        except:
            raise AttributeError("No counts table found")

    def get_counts_qc(self):
        '''
        Retrieve the QC'd counts
        table if set/generated
        '''

        try:
            return getattr(self, "counts_qc")
        except:
            raise AttributeError("No QC'd counts table found")

    def get_expression(self):
        '''
        Retrieve expression table if it
        has been set
        '''

        try:
            return getattr(self, "transcript_expression")
        except:
            raise AttributeError("No expression table found")
            
    # ---------------------------------------- #
    # analysis methods                         #
    # ---------------------------------------- #

    def compute_parameters(self, niter=1000, thin=2, burn=500,
                           store_chains=True, chains_dir=None):
        '''
        Calculate gene and cell parameters using BASiCs Bayesian
        hierarchical model.  Uses MCMC and can be comutationally
        expensive for large datasets and long run times.

        It assumes all QC has been carried out prior to parameter
        estimation.  It also assumes that the counts table has
        been subset appropriately if the parameter estimation is
        to be used for differential expression and overdispersion
        testing.

        Arguments
        ---------
        `niter`: int
        number of iterations to run the MCMC sampler for

        `thin`: int
        thinning rate for the MCMC sampler.  This equates to
        selecting every nth sample before burning, where n=thin

        `burn`: int
        the number of samples to discard as the burn-in phase
        of the sampler.  It is recommended to discard at least
        50% of samples for each chain.

        `store_chains`: boolean
        store chains in a text file, output into `chains_dir`

        `chains_dir`: string
        directory in which to store chains containing MCMC
        parameter estimates.  A file will be created for each
        chain and parameter.

        Returns
        -------
        parameter_estimates: pandas.core.frame.DataFrame
        a pandas dataframe containing parameter estimates from the
        (assumed) stationary distribution of each cell and gene.
        This dataframe is automatically set as the attribute 
        `parameter_estimates`, which can be accessed using the
        `get_parameters(**kwargs)` bound method.
        '''

        # steps:
        # push counts table into R
        # push spike in info into R
        # set up BASiCs data object
        # push MCMC parameters into R
        # start MCMC running <- capture output
        # store chains and parameter estimates as attributes
        py2ri.activate()
        try:
            # make sure QC performed first
            R.assign("counts", self.counts_qc)
        except:
            raise AttributeError("QC'd counts table not "
                                 "detected.  Run QC steps " 
                                 "to filter poor cells and "
                                 "zero-expression genes.")

        try:
            getattr(self, "spikein_table")
            try:
                self.spikein_table["copies_per_well"]
            except KeyError:
                # how do you guess what the dilution is??
                logging.warn("Spike in copies per well is not present. "
                             "Using standard formulation: "
                             "conc * 10^-18 * (Avogadro's number) * "
                             "(10*10^-3) * dilution factor")
                try:
                    assert self.spikein_dilution
                except AttributeError:
                    raise AttributeError("Dilution factor for spike in "
                                         "transcripts has not been set. "
                                         "This is required to calculate the "
                                         "# copies per cell of each spike in.")

                # should this be calculated on a log scale
                # for numerical stability?
                conc = self.spikein_table["conc"]
                avogadro = 6.022 * (10**23)
                # assume reaction volume is in microlitres, hence 10^-3 scaling
                copies = conc * (10**-18) * avogadro *\
                         (10 * (10**-3)) * (1/float(self.spikein_dilution))

                self.spikein_table["copies_per_well"] = copies
                
        except:
            raise AttributeError("Spike in transcript table "
                                 "not found. Cannot perform "
                                 "parameter estimation in the "
                                 "absence of spike ins.")

        R.assign("spike_in", self.spikein_table)

        R('''Counts <- as.matrix(counts)''')
        R('''TechQC <- grepl("ERCC", rownames(Counts))''')
        R('''SpikeInfo <- spike_in[spike_in$gene_id %in% rownames(Counts)][TechQC]''')

        # setup the BASiCS data object
        R('''suppressMessages(library(BASiCS))''')

        R('''Data <- newBASiCS_Data(Counts = Counts,'''
          '''Tech = TechQC, SpikeInfo = SpikeInfo)''')

        # define MCMC parameters in the R workspace
        R('''niter <- %(niter)i''' % locals())
        R('''thin <- %(thin)i''' % locals())
        R('''burn <- %(burn)i''' % locals())
        R('''loops <- (niter - burn)/thin''')

        # ladies and gentlemen, start your engines
        #if store_chains:          
        #    R('''MCMC_output <- BASiCS_MCMC(Data, N=niter,'''
        #      '''Thin=thin, Burn=burn, StoreChains=True,'''
        #      '''RunName="%s(self.__name__)s", StoreDir="%(chains_dir)s")''' % locals())
        #else:
        #    R('''MCMC_output <- BASiCS_MCMC(Data, N=niter,'''
        #      '''Thin=thin, Burn=burn, StoreChains=False)''')



if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)
    # use this for testing the module
    s1 = SingleCell()
    print "execute tests"

    # testing set a name for each instance
    print __name__

    print s1.__name__
    s1.set_attribute("counts_table", "mytable")
    print s1.get_counts()    
    s1.set_attribute("counts_table", "notatable")
    print s1.get_counts()
    print s1.__name__

    # test cell QC
    with open("/ifs/projects/proj056/pipeline_scqc_steveTec/qc_out.dir/mTEC_qc.tsv", "r") as qfile:
        qc_list = [q.rstrip("\n") for q in qfile.readlines()]

    ifile = "/ifs/projects/proj056/pipeline_scqc_steveTec/feature_counts.dir/001-1-feature_counts.tsv.gz"
    s1.add_counts_table(counts_matrix=ifile,
                        compression="gzip",
                        remove_zeros=True,
                        cell_qc=qc_list)
    print s1.get_counts_qc().shape
    s2 = SingleCell()
    #s2.counts_table_qc(cell_qc=qc_list, exprs_threshold=5)

    df = pd.read_table(ifile, compression="gzip",
                       index_col=0, header=0)
    s2.add_counts_table(counts_matrix=df,
                        remove_zeros=False)

    print s2.get_counts().shape
    s2.counts_table_qc(cell_qc=qc_list, exprs_threshold=5)
    print s2.get_counts_qc().shape
    # test MCMC
    #s2.compute_parameters()
