'''
SCRNA.py is a module file for single cell RNA seq analysis functions.

Functions
---------


Classes
-------


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

import CGAT.Experiment as E
import sys
import rpy2.robjects as ro
from rpy2.robjects import r as R
import rpy2.pandas2ri as py2ri
import re
import pandas as pd
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import seaborn as sns


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

    def __init__(self, **kwargs):

        # dynamically set attributes upon instantiation
        for attr in kwargs.keys():
            setattr(self, attr, kwargs[attr])

    def set_attribute(self, attribute, value):
        '''
        set an attribute with a value
        '''

        setattr(self, attribute, value)

    def get_expression(self):
        '''
        Retrieve expression table if it
        has been set
        '''

        try:
            getattr(self, "transcript_expression")
        except:
            raise AttributeError("No expression table found")
            
    def compute_parameters(self, niter=1000, thin=2, burn=500):
        '''
        Calculate gene and cell parameters using BASiCs Bayesian
        hierarchical model.  Uses MCMC and can be comutationally
        expensive for large datasets and long run times.

        It assumes all QC has been carried out prior to parameter
        estimation.  It also assumes that the counts table has
        been subset appropriately if the parameter estimation is
        to be used for differential expression and overdispersion
        testing.
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
                E.warn("Spike in copies per well is not present. "
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
        # setup the BASiCS data object
        R('''Data <- newBASiCS_Data(Counts = Counts,'''
          '''Tech = TechQC, SpikeInfo = SpikeInfo)''')

        # define MCMC parameters in the R workspace
        R('''niter <- %(niter)i''' % locals())
        R('''thin <- %(thin)i''' % locals())
        R('''burn <- %(burn)i''' % locals())
        R('''loops <- (niter - burn)/thin''')

        # ladies and gentlemen, start your engines
        R('''MCMC_output <- BASiCS_MCMC(Data, N=niter,'''
          '''Thin=thin, Burn=burn, StoreChains=True,'''
          '''RunName="%s()s", StoreDir="%()s")''')
