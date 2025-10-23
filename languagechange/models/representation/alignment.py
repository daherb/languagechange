import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from languagechange.usages import TargetUsage
from languagechange.corpora import LinebyLineCorpus
from LSCDetection.modules.utils_ import Space
from languagechange.models.representation.static import StaticModel
import os
from LSCDetection.modules import embeddings
from LSCDetection.modules.cupy_utils import *
from LSCDetection.alignment.map_embeddings import dropout, topk_mean
import re
import sys
import collections
import time
import languagechange.logging


class OrthogonalProcrustes():
    """
    A class to align word embeddings using the Orthogonal Procrustes method.

    This method aligns two embedding spaces by finding an optimal orthogonal transformation.
    """
    
    def __init__(self, savepath1:str, savepath2:str):
        """
        Initialize the class with paths to save the aligned embeddings.

        Args:
            savepath1 (str): Path to save the aligned version of the first model.
            savepath2 (str): Path to save the aligned version of the second model.
        """
        self.savepath1 = savepath1
        self.savepath2 = savepath2


    # This function is adapted from https://github.com/Garrafao/LSCDetection/blob/master/alignment/map_embeddings.py
    def align(self, model1:StaticModel, model2:StaticModel,
                   encoding = 'utf-8', # the character encoding for input/output (defaults to utf-8)
                   precision = 'fp32', # should be in {'fp16','fp32','fp64'}
                   cuda = False, # use cuda (requires cupy)
                   batch_size : int = 10000, # batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory
                   seed = 0, # the random seed (defaults to 0)

                   # recommended args
                   supervised = None, # recommended if you have a large training dictionary 
                   semi_supervised = None, # recommended if you have a small seed dictionary
                   identical = False, # recommended if you have no seed dictionary but can rely on identical words
                   unsupervised = False, # recommended if you have no seed dictionary and do not want to rely on identical words
                   acl2018 = False, # reproduce our ACL 2018 system
                   aaai2018 = None, # reproduce our AAAI 2018 system
                   acl2017 = False, # reproduce our ACL 2017 system with numeral initialization
                   acl2017_seed = None, # reproduce our ACL 2017 system with a seed dictionary
                   emnlp2016 = None, # reproduce our EMNLP 2016 system

                   # init args. Below four are mutually exclusive
                   init_dictionary = sys.stdin.fileno(), # the training dictionary file (defaults to stdin)
                   init_identical = True, # use identical words as the seed dictionary
                   init_numerals = False, # use latin numerals (i.e. words matching [0-9]+) as the seed dictionary
                   init_unsupervised = False, # recommended if you have no seed dictionary and do not want to rely on identical words

                   unsupervised_vocab : int = 0, # restrict the vocabulary to the top k entries for unsupervised initialization

                   # mapping args
                   normalize = ['unit'], # the normalization actions to perform in order. Should be a list of {'unit', 'center', 'unitdim', 'centeremb', 'none'}
                   whiten = False, # whiten the embeddings
                   src_reweight : float = 0, # re-weight the source language embeddings
                   trg_reweight : float = 0, # re-weight the target language embeddings
                   src_dewhiten = None, # de-whiten the source language embeddings
                   trg_dewhiten = None, # de-whiten the target language embeddings
                   dim_reduction : int = 0, # apply dimensionality reduction

                   # The two arguments below are mutually exclusive
                   orthogonal = True, # use orthogonal constrained mapping
                   unconstrained = False, # use unconstrained mapping

                   # self-learning args
                   self_learning = False, # enable self-learning 
                   vocabulary_cutoff : int = 0, # restrict the vocabulary to the top k entries
                   direction = 'union', # the direction for dictionary induction (defaults to union). Choices=['forward', 'backward', 'union']
                   csls_neighborhood : int = 0, # use CSLS for dictionary induction
                   threshold : float = 0.000001, # the convergence threshold (defaults to 0.000001)
                   validation = None, # a dictionary file for validation at each iteration
                   stochastic_initial : float = 0.1, # initial keep probability stochastic dictionary induction (defaults to 0.1)
                   stochastic_multiplier : float = 2.0, # stochastic dictionary induction multiplier (defaults to 2.0)
                   stochastic_interval : int = 50, # stochastic dictionary induction interval (defaults to 50)
                   log = None, # write to a log file in tsv format at each iteration
                   verbose = False # write log information to stderr at each iteration
                   ):
        """
        Perform orthogonal alignment between two embedding models using a subprocess.

        Args:
            model1 (StaticModel): The first static word embedding model to align.
            model2 (StaticModel): The second static word embedding model to align.
        """
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.alignment.map_embeddings", 
        #    "--normalize", "unit",
        #    "--init_identical",
        #    "--orthogonal",
        #    model1.matrix_path,
        #    model2.matrix_path,
        #    self.savepath1,
        #    self.savepath2])    

        logging.info(__file__.upper())
        start_time = time.time()

        # Asserting correct arguments
        assert src_dewhiten in {'src', 'trg', None}
        assert trg_dewhiten in {'src', 'trg', None}

        assert not (orthogonal and unconstrained)

        assert direction in {'forward', 'backward', 'union'}

        assert (sum(bool(arg) for arg in {supervised, semi_supervised, identical, unsupervised, acl2018, aaai2018, acl2017, acl2017_seed, emnlp2016}) <= 1)
        
        if init_identical or init_numerals or init_unsupervised:
            init_dictionary = None

        assert (sum(bool(arg) for arg in {init_dictionary, init_identical, init_numerals, init_unsupervised}) <= 1) 

        for e in normalize:
            assert e in {'unit', 'center', 'unitdim', 'centeremb', 'none'}

        if supervised is not None:
            init_dictionary = supervised
            normalize=['unit', 'center', 'unit']
            whiten=True
            src_reweight=0.5
            trg_reweight=0.5
            src_dewhiten='src'
            trg_dewhiten='trg'
            batch_size=1000

        if semi_supervised is not None:
            init_dictionary=semi_supervised
            normalize=['unit', 'center', 'unit']
            whiten=True
            src_reweight=0.5
            trg_reweight=0.5
            src_dewhiten='src'
            trg_dewhiten='trg'
            self_learning=True
            vocabulary_cutoff=20000
            csls_neighborhood=10

        if identical:
            init_identical=True
            normalize=['unit', 'center', 'unit']
            whiten=True
            src_reweight=0.5
            trg_reweight=0.5
            src_dewhiten='src'
            trg_dewhiten='trg'
            self_learning=True
            vocabulary_cutoff=20000
            csls_neighborhood=10

        if unsupervised or acl2018:
            init_unsupervised=True
            unsupervised_vocab=4000
            normalize=['unit', 'center', 'unit']
            whiten=True
            src_reweight=0.5
            trg_reweight=0.5
            src_dewhiten='src'
            trg_dewhiten='trg'
            self_learning=True
            vocabulary_cutoff=20000
            csls_neighborhood=10

        if aaai2018:
            init_dictionary=aaai2018
            normalize=['unit', 'center']
            whiten=True
            trg_reweight=1
            src_dewhiten='src'
            trg_dewhiten='trg'
            batch_size=1000

        if acl2017:
            init_numerals=True
            orthogonal=True
            normalize=['unit', 'center']
            self_learning=True
            direction='forward'
            stochastic_initial=1.0
            stochastic_interval=1
            batch_size=1000

        if acl2017_seed:
            init_dictionary=acl2017_seed
            orthogonal=True
            normalize=['unit', 'center']
            self_learning=True
            direction='forward'
            stochastic_initial=1.0
            stochastic_interval=1
            batch_size=1000

        if emnlp2016:
            init_dictionary=emnlp2016
            orthogonal=True
            normalize=['unit', 'center']
            batch_size=1000


        # Check arguments
        if (src_dewhiten is not None or trg_dewhiten is not None) and not whiten:
            logging.info('ERROR: De-whitening requires whitening first')
            sys.exit(-1)

        # Choose the right dtype for the desired precision
        if precision == 'fp16':
            dtype = 'float16'
        elif precision == 'fp32':
            dtype = 'float32'
        elif precision == 'fp64':
            dtype = 'float64'
        else:
            logging.info("ERROR: Precision needs to be one of ('fp16','fp32','fp64')")
            sys.exit(-1)

        # Read input embeddings
        src_input = model1.matrix_path #the input source embeddings
        trg_input = model2.matrix_path #the input target embeddings

        srcfile = open(src_input, encoding=encoding, errors='surrogateescape')
        trgfile = open(trg_input, encoding=encoding, errors='surrogateescape')

        src_words, x = embeddings.read(srcfile, dtype=dtype)
        trg_words, z = embeddings.read(trgfile, dtype=dtype)

        # NumPy/CuPy management
        if cuda:
            if not supports_cupy():
                print('ERROR: Install CuPy for CUDA support', file=sys.stderr) # Change to logging
                sys.exit(-1)
            xp = get_cupy()
            x = xp.asarray(x)
            z = xp.asarray(z)
        else:
            xp = np
        xp.random.seed(seed)

        # Build word to index map
        src_word2ind = {word: i for i, word in enumerate(src_words)}
        trg_word2ind = {word: i for i, word in enumerate(trg_words)}

        # STEP 0: Normalization
        embeddings.normalize(x, normalize)
        embeddings.normalize(z, normalize)

        # Build the seed dictionary
        src_indices = []
        trg_indices = []
        if init_unsupervised:
            sim_size = min(x.shape[0], z.shape[0]) if unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], unsupervised_vocab)
            u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
            xsim = (u*s).dot(u.T)
            u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
            zsim = (u*s).dot(u.T)
            del u, s, vt
            xsim.sort(axis=1)
            zsim.sort(axis=1)
            embeddings.normalize(xsim, normalize)
            embeddings.normalize(zsim, normalize)
            sim = xsim.dot(zsim.T)
            if csls_neighborhood > 0:
                knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
                knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
                sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2
            if direction == 'forward':    
                src_indices = xp.arange(sim_size)
                trg_indices = sim.argmax(axis=1)
            elif direction == 'backward':
                src_indices = sim.argmax(axis=0)
                trg_indices = xp.arange(sim_size)
            elif direction == 'union':
                src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
                trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
            del xsim, zsim, sim
        elif init_numerals:
            numeral_regex = re.compile('^[0-9]+$')
            src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
            trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
            numerals = src_numerals.intersection(trg_numerals)
            for word in numerals:
                src_indices.append(src_word2ind[word])
                trg_indices.append(trg_word2ind[word])
        elif init_identical:
            identical = set(src_words).intersection(set(trg_words))
            for word in identical:
                src_indices.append(src_word2ind[word])
                trg_indices.append(trg_word2ind[word])
        else:
            f = open(init_dictionary, encoding=encoding, errors='surrogateescape')
            for line in f:
                src, trg = line.split()
                try:
                    src_ind = src_word2ind[src]
                    trg_ind = trg_word2ind[trg]
                    src_indices.append(src_ind)
                    trg_indices.append(trg_ind)
                except KeyError:
                    print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

        # Read validation dictionary
        if validation is not None:
            f = open(validation, encoding=encoding, errors='surrogateescape')
            validation_dict = collections.defaultdict(set)
            oov = set()
            vocab = set()
            for line in f:
                src, trg = line.split()
                try:
                    src_ind = src_word2ind[src]
                    trg_ind = trg_word2ind[trg]
                    validation_dict[src_ind].add(trg_ind)
                    vocab.add(src)
                except KeyError:
                    oov.add(src)
            oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
            validation_coverage = len(validation_dict) / (len(validation_dict) + len(oov))

        # Create log file
        if log:
            log_file = open(log, mode='w', encoding=encoding, errors='surrogateescape')

        # Allocate memory
        xw = xp.empty_like(x)
        zw = xp.empty_like(z)
        src_size = x.shape[0] if vocabulary_cutoff <= 0 else min(x.shape[0], vocabulary_cutoff)
        trg_size = z.shape[0] if vocabulary_cutoff <= 0 else min(z.shape[0], vocabulary_cutoff)
        simfwd = xp.empty((batch_size, trg_size), dtype=dtype)
        simbwd = xp.empty((batch_size, src_size), dtype=dtype)
        if validation is not None:
            simval = xp.empty((len(validation_dict.keys()), z.shape[0]), dtype=dtype)

        best_sim_forward = xp.full(src_size, -100, dtype=dtype)
        src_indices_forward = xp.arange(src_size)
        trg_indices_forward = xp.zeros(src_size, dtype=int)
        best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
        src_indices_backward = xp.zeros(trg_size, dtype=int)
        trg_indices_backward = xp.arange(trg_size)
        knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
        knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

        # Training loop
        best_objective = objective = -100.
        it = 1
        last_improvement = 0
        keep_prob = stochastic_initial
        t = time.time()
        end = not self_learning
        while True:

            # Increase the keep probability if we have not improve in stochastic_interval iterations
            if it - last_improvement > stochastic_interval:
                if keep_prob >= 1.0:
                    end = True
                keep_prob = min(1.0, stochastic_multiplier*keep_prob)
                last_improvement = it

            # Update the embedding mapping
            if orthogonal or not end:  # orthogonal mapping
                u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
                w = vt.T.dot(u.T)
                x.dot(w, out=xw)
                zw[:] = z
            elif unconstrained: # unconstrained mapping
                x_pseudoinv = xp.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
                w = x_pseudoinv.dot(z[trg_indices])
                x.dot(w, out=xw)
                zw[:] = z
            else:  # advanced mapping

                # TODO xw.dot(wx2, out=xw) and alike not working
                xw[:] = x
                zw[:] = z

                # STEP 1: Whitening
                def whitening_transformation(m):
                    u, s, vt = xp.linalg.svd(m, full_matrices=False)
                    return vt.T.dot(xp.diag(1/s)).dot(vt)
                if whiten:
                    wx1 = whitening_transformation(xw[src_indices])
                    wz1 = whitening_transformation(zw[trg_indices])
                    xw = xw.dot(wx1)
                    zw = zw.dot(wz1)

                # STEP 2: Orthogonal mapping
                wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
                wz2 = wz2_t.T
                xw = xw.dot(wx2)
                zw = zw.dot(wz2)

                # STEP 3: Re-weighting
                xw *= s**src_reweight
                zw *= s**trg_reweight

                # STEP 4: De-whitening
                if src_dewhiten == 'src':
                    xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
                elif src_dewhiten == 'trg':
                    xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
                if trg_dewhiten == 'src':
                    zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
                elif trg_dewhiten == 'trg':
                    zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

                # STEP 5: Dimensionality reduction
                if dim_reduction > 0:
                    xw = xw[:, :dim_reduction]
                    zw = zw[:, :dim_reduction]

            # Self-learning
            if end:
                break
            else:
                # Update the training dictionary
                if direction in ('forward', 'union'):
                    if csls_neighborhood > 0:
                        for i in range(0, trg_size, simbwd.shape[0]):
                            j = min(i + simbwd.shape[0], trg_size)
                            zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                            knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=csls_neighborhood, inplace=True)
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                        simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                        simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                        dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
                if direction in ('backward', 'union'):
                    if csls_neighborhood > 0:
                        for i in range(0, src_size, simfwd.shape[0]):
                            j = min(i + simfwd.shape[0], src_size)
                            xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                            knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=csls_neighborhood, inplace=True)
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                        simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                        simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                        dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
                if direction == 'forward':
                    src_indices = src_indices_forward
                    trg_indices = trg_indices_forward
                elif direction == 'backward':
                    src_indices = src_indices_backward
                    trg_indices = trg_indices_backward
                elif direction == 'union':
                    src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                    trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

                # Objective function evaluation
                if direction == 'forward':
                    objective = xp.mean(best_sim_forward).tolist()
                elif direction == 'backward':
                    objective = xp.mean(best_sim_backward).tolist()
                elif direction == 'union':
                    objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
                if objective - best_objective >= threshold:
                    last_improvement = it
                    best_objective = objective

                # Accuracy and similarity evaluation in validation
                if validation is not None:
                    src = list(validation_dict.keys())
                    xw[src].dot(zw.T, out=simval)
                    nn = asnumpy(simval.argmax(axis=1))
                    accuracy = np.mean([1 if nn[i] in validation_dict[src[i]] else 0 for i in range(len(src))])
                    similarity = np.mean([max([simval[i, j].tolist() for j in validation_dict[src[i]]]) for i in range(len(src))])

                # Logging
                duration = time.time() - t
                if verbose:
                    print(file=sys.stderr)
                    print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                    print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                    print('\t- Drop probability: {0:9.4f}%'.format(100 - 100*keep_prob), file=sys.stderr)
                    if validation is not None:
                        print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
                        print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
                        print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
                    sys.stderr.flush()
                if log is not None:
                    val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                        100 * similarity, 100 * accuracy, 100 * validation_coverage) if validation is not None else ''
                    
                    print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log_file)
                    log_file.flush()

            t = time.time()
            it += 1

        # Write mapped embeddings
        srcfile = open(self.savepath1, mode='w', encoding=encoding, errors='surrogateescape')
        trgfile = open(self.savepath2, mode='w', encoding=encoding, errors='surrogateescape')
        embeddings.write(src_words, xw, srcfile) #the output source embeddings
        embeddings.write(trg_words, zw, trgfile) #the output target embeddings
        srcfile.close()
        trgfile.close()

        logging.info("--- %s seconds ---" % (time.time() - start_time))