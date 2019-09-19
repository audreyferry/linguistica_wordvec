# -*- encoding: utf8 -*-
 
from collections import defaultdict

#from sympy import *   # may just Matrix??
#xfrom sympy.matrices import Matrix


from scipy import (sparse, spatial)
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
from scipy.linalg import LinAlgError
#from scipy import scikitlearn as sklearn          # AUDREY  2017_04_05  SyntaxError: invalid syntax
#from scipy import sklearn                          # AUDREY  2017_04_05  ImportError: cannot import name 'sklearn'
import sklearn
from sklearn.preprocessing import normalize as skl_normalize
from sklearn.decomposition import DictionaryLearning, sparse_encode
from sklearn.decomposition import FastICA
from sklearn.cluster       import spectral_clustering, k_means   #lifted, so may not need spectral_clustering     
from sklearn.manifold      import spectral_embedding             #lifted, so may not need spectral_embedding
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from scipy.sparse.csgraph  import laplacian as csgraph_laplacian

import numpy as np
import networkx as nx

import plotly
import plotly.graph_objs as go

import jsonpickle   # used   conda install -c conda-forge jsonpickle   
import xlsxwriter

import os   # audrey  2017_01_29  Needed only for os.mkdir('HEATMAPS')
import warnings

import math
import cmath
import time                         # AUDREY       2017_08_14
import datetime                     # AUDREY
import matplotlib.patches as mpatches   # AUDREY   2017_06_20
import matplotlib.pyplot as plt     # AUDREY    2017_04_09
from mpl_toolkits.mplot3d import axes3d

from linguistica.util import double_sorted


def remove_outliers(wordlist):
	wordlist.remove('bombers')
	wordlist.remove('ballistic')
	return wordlist
	

def get_context_array(wordlist, bigram_to_freq, trigram_to_freq,
              min_context_count, verbose=False):
    worddict = {word: wordlist.index(word) for word in wordlist}
    
    
    # VERY TEMPORARY (omitting 'greville' and 'bargaining' specifically)  Feb. 26, 2018   10:59 pm

    # convert the bigram and trigram counter dicts into list and sort them
    # throw away bi/trigrams whose frequency is below min_context_count

    bigram_to_freq_sorted = [(bigram, freq) for bigram, freq in
                             double_sorted(bigram_to_freq.items(),
                                           key=lambda x: x[1],
                                           reverse=True) if
                             (freq >= min_context_count 
                             #and bigram[0] in worddict					# condition added Feb. 26, 2018 
                             #and bigram[1] in worddict					# (one cause of anomalous entries on eigenvectors)
                             )]

    trigram_to_freq_sorted = [(trigram, freq) for trigram, freq in
                              double_sorted(trigram_to_freq.items(),
                                            key=lambda x: x[1],
                                            reverse=True) if
                              (freq >= min_context_count 
                              #and trigram[0] in worddict				# condition added Feb. 26, 2018 
                              #and trigram[1] in worddict 
                              #and trigram[2] in worddict
                              )]

    # This is necessary so we can reference variables from inner functions
    class Namespace:
        pass

    ns = Namespace()
    ns.n_contexts = 0

    # We use "n_contexts" to keep track of how many unique contexts there are.
    # Conveniently, n_contexts also serves to provide a unique context
    # index whenever the program encounters a new context. The dummy class
    # Namespace is to make it possible that we can refer to and update
    # n_contexts within inner functions
    # (both "contexts_increment" and "add_word")
    # inside this "GetContextArray" function.

    def contexts_increment():
        tmp = ns.n_contexts
        ns.n_contexts += 1
        return tmp

    contextdict = defaultdict(contexts_increment)
    # key: context (e.g., tuple ('of', '_', 'cat') as a 3-gram context for 'the'
    # value: context index (int)
    # This dict is analogous to worddict, where each key is a word (str)
    # and each value is a word index (int).

    # entries for sparse matrix
    rows = []  # row numbers are word indices
    cols = []  # column numbers are context indices
    values = []

    words_to_contexts = dict()
    contexts_to_words = dict()

    for word in worddict.keys():				# audrey  Feb. 23, 2018  WATCH OUT! THIS IS EVERY word IN wordlist!!
        words_to_contexts[word] = dict()

    def add_word(current_word, current_context, occurrence_count):
        word_no = worddict[current_word]				# This is why context_array has rows for words not encountered in any ngrams.
        context_no = contextdict[current_context]		# Apparently it will have enough rows to accommodate the highest worddict index encountered.
        rows.append(word_no)
        cols.append(context_no)
        
        # if we use 1, we assume "type" counts.
        # What if we use occurrence_count (--> "token" counts)?
        values.append(1)

        # update words_to_contexts and contexts_to_words
        if current_context not in words_to_contexts[current_word]:
            words_to_contexts[current_word][current_context] = 0

        if current_context not in contexts_to_words:
            contexts_to_words[current_context] = dict()
        if current_word not in contexts_to_words[current_context]:
            contexts_to_words[current_context][current_word] = 0

        words_to_contexts[current_word][current_context] += occurrence_count
        contexts_to_words[current_context][current_word] += occurrence_count

    for trigram, freq in trigram_to_freq_sorted:
        word1, word2, word3 = trigram

        context1 = ('_', word2, word3)
        context2 = (word1, '_', word3)
        context3 = (word1, word2, '_')

        if word1 in words_to_contexts:
            add_word(word1, context1, freq)
        if word2 in words_to_contexts:
            add_word(word2, context2, freq)
        if word3 in words_to_contexts:
            add_word(word3, context3, freq)

    for bigram, freq in bigram_to_freq_sorted:
        word1, word2 = bigram

        context1 = ('_', word2)
        context2 = (word1, '_')

        if word1 in words_to_contexts:
            add_word(word1, context1, freq)
        if word2 in words_to_contexts:
            add_word(word2, context2, freq)
    
    
    # FIND OUT SOME INFO 
    # Feb. 23, 2018  Is there anything different about words that get minuscule values on eigenvectors?
    # YES    For 47, tumbled, undue, unlimited:   The list of contexts is empty! 
    # How could it be???   Answer:None of their contexts have enough occurrences. Default value for min_context_count is 3.
    
    if verbose:
    	print("\nVerbose screen output for investigating occurrence of minuscule entries, positive and negative(!), on Eig0...  Feb.-March 2018")
    	print("This development was carried out for Brown corpus, max_word_types=7000, n_eigenvectors=6.")
    	print("If further investigation is needed in future, specifics may be changed as necessary.")
    	print("Screen output and spreadsheet 'eigenvector_data_for_excel.csv' should be used together.")
    	print("In get_context_array()...")
    	
    	print("\nThe following lines show words_to_contexts dictionary values for a selection of low-frequency words.")
    	print("ALL THE CONTEXTS FOR '47','tumbled', 'undue', and 'unlimited' OCCUR FEWER THAN THE REQUIRED MIN_CONTEXT_COUNT TIMES:")
    	print("for  104: ", list(words_to_contexts['104']))
    	print("for 1913: ", list(words_to_contexts['1913']))
    	print("for 1943: ", list(words_to_contexts['1943']))
    	print("for 20th: ", list(words_to_contexts['20th']))
    	print("for   47: ", list(words_to_contexts['47']))                    # no context with enough occurrences
    	print("for   55: ", list(words_to_contexts['55']))
    	print("for  700: ", list(words_to_contexts['700']))
    	print("for  7th: ", list(words_to_contexts['7th']))
    	print("for absorption: ", list(words_to_contexts['absorption']))
    	print("for tumbled: ", list(words_to_contexts['tumbled']))
    	print("for undue: ", list(words_to_contexts['undue']))
    	print("for unlimited: ", list(words_to_contexts['unlimited']))
    	print("####")
    	print("for ballistic: ", list(words_to_contexts['ballistic']))
    	print("for bombers: ", list(words_to_contexts['bombers']))
    	print("for objective: ", list(words_to_contexts['objective']))
    	print("for state: ", list(words_to_contexts['state']))
    	print("for long-range: ", list(words_to_contexts['long-range']))
    	print("contexts_to_words for ('long-range', '_'): ", list(contexts_to_words[('long-range', '_')]))
    	print("####")
    	print("for rourke: ", list(words_to_contexts['rourke']))
    	print("for incomplete: ", list(words_to_contexts['incomplete']))
    	print("####")
    	print("for phosphor: ", list(words_to_contexts['phosphor']))
    	print("contexts_to_words for ('_', 'screen'): ", list(contexts_to_words[('_', 'screen')]))
    	#print("contexts_to_words for ('_', 'chip', 'industry'): ", list(contexts_to_words[('_', 'chip', 'industry')]))
    	#print("contexts_to_words for ('_', 'chip'): ", list(contexts_to_words[('_', 'chip')]))
    	#print("contexts_to_words for ('the', '_', 'chip'): ", list(contexts_to_words[('the', '_', 'chip')]))
    	#print("contexts_to_words for ('the', '_'): ", list(contexts_to_words[('the', '_')]))
    	
    	print("\nTHESE 8 WORDS HAVE CONTEXTS WITH ENOUGH OCCURRENCES BUT STILL PRODUCE MINUSCULE VALUES ON Eig0 (5 of them negative)")
    	print("Here are  words_to_contexts  results for these 8.")
    	print("for berger: ", list(words_to_contexts['berger']))
    	print("for don: ", list(words_to_contexts['don']))
    	print("for virus: ", list(words_to_contexts['virus']))
    	print("for bargaining: ", list(words_to_contexts['bargaining']))
    	print("for payne: ", list(words_to_contexts['payne']))
    	print("for plasma: ", list(words_to_contexts['plasma']))
    	print("for greville: ", list(words_to_contexts['greville']))
    	print("for hank: ", list(words_to_contexts['hank']))
    	
    	
    	print("\n4 of the words checked above were correctible by filtering out use of any ngrams which include words from outside of the designated wordlist.")
    	print("(i.e., from beyond the max_word_types limit).")
    	print("COMMENT ADDED March 15, 2018: Rethinking led to removing the filtering code--")
    	print("prompted by discovery that 'rourke' and 'incomplete' took non-minuscule values but were being filtered by that code.")
    	print("The 4 words don, virus, plasma, and hank still get filtered, not here but in get_shared_context_matrix() because they have no shared contexts.")
    	print("rourke and incomplete remain in the wordlist because they do have shared contexts (even though those contexts involve words from beyond the wordlist).")
    	
    	
    	# audrey  ADDED on Feb. 24, 2018  REDUCE THE context_array
    	
    	print("\nSummarizing what was learned so far:")
    	print("A context is acceptable if it <<OMIT AS OF March 15, 2018 [uses only words from the designated wordlist and]>> occurs at least min_context_count times in the corpus.")
    	print("The specified entries for the csr matrix 'context_array' are constructed from acceptable contexts.")
    	print("Some words in the given wordlist have no acceptable contexts.")
    	print("By construction the context_array however has a row for <every> word in the given wordlist.")
    	print("Thus some rows have no specified entries. This accounts for most of the words with minuscule entries on Eig0.")
    	print("We will omit those rows from the context_array and those words from the original wordlist.")
    
    
    # audrey  Feb. 23, 2018   AT THIS POINT, DELETE WHEERE APPROPRIATE (worddict? words_to_contexts? rows?)
    #                         THOSE WORDS WHICH HAVE NO LISTED CONTEXTS
    #                         CHECK CAREFULLY FOR ANY OTHER PLACES  
    #                         AND CHECK NUMBERS WHEREVER POSSIBLE
    # Build the matrix first, and then reduce it by getnnz? Or can we reduce inputs before building matrix?
    
    # csr_matrix in scipy.sparse means compressed matrix
    #context_array = sparse.csr_matrix((values, (rows, cols)),
    #                                  shape=(len(worddict), ns.n_contexts + 1),	# ns.n_contexts is better. 
    #                                  dtype=np.int64)								
    
    # Also shape is determined automatically 
    # from highest word_no and context_no encountered.
    # We get extraneous rows because word_no is taken from worddict,
    # not from the words as they're encountered successively in acceptable ngrams.
    
    context_array = sparse.csr_matrix((values, (rows, cols)),
    								  dtype=np.int64)
    
    if verbose:
    	print("\nBefore getnnz is used to filter out rows and columns which have no explicitly specified entries:")
    	print("context_array.shape =", context_array.shape)
    	print("no_specified_entries_rows.shape =", context_array[context_array.getnnz(1)==0].shape)
    	print("no_specified_entries_cols.shape =", context_array[:,context_array.getnnz(0)==0].shape)
                                      
    # M = M[M.getnnz(1)>0][:,M.getnnz(0)>0] #GOOD
    context_array = context_array[context_array.getnnz(1)>0][:,context_array.getnnz(0)>0]
    
    if verbose:
    	print("After getnnz, context_array.shape =", context_array.shape)
    
    ctxt_supported_wordlist = [word for word in wordlist if list(words_to_contexts[word]) != [] ]
    if verbose:
    	print("\nctxt_supported_wordlist is those words from original wordlist with at least one acceptable context.")
    	print("lctxt_supported_wordlist) =", len(ctxt_supported_wordlist)) 

    # RECORD words_to_contexts AND contexts_to_words DICTIONARIES FOR USE BY STANDALONE PROGRAMS
    #outfile_wtoc_jsonpickle_name = "words_to_contexts_jsonpickle.txt"
    #outfile_wtoc_jsonpickle      =  open(outfile_wtoc_jsonpickle_name, "w")
    #serialstr = jsonpickle.encode(words_to_contexts)
    #print(serialstr, file=outfile_wtoc_jsonpickle)
    #outfile_wtoc_jsonpickle.close()
    
    #outfile_ctow_jsonpickle_name = "contexts_to_words_jsonpickle.txt"
    #outfile_ctow_jsonpickle      =  open(outfile_ctow_jsonpickle_name, "w")
    #serialstr = jsonpickle.encode(contexts_to_words)
    #print(serialstr, file=outfile_ctow_jsonpickle)
    #outfile_ctow_jsonpickle.close()

    
    return context_array, words_to_contexts, contexts_to_words, ctxt_supported_wordlist


def get_shared_context_matrix(context_array, ctxt_supported_wordlist, contexts_to_words=None, verbose=False):   
	# contexts_to_words is needed only for verbose screen output
	################
	if verbose:
		print("\nContinuing in get_shared_context_matrix()...")
		print("\nAT THIS POINT THERE ARE <<March 15, 2018 replace 4 by 8>> 8 REMAINING WORDS WITH MINUSCULE VALUES ON Eig0.")
		print("Here are   contexts_to_words  results for the contexts obtained by words_to_contexts for these 4.")
		print("for ('anthony', '_'): ", list(contexts_to_words[('anthony', '_')]))
		print("for ('edward', '_'): ", list(contexts_to_words[('edward', '_')]))
		print("for ('vincent', '_'): ", list(contexts_to_words[('vincent', '_')]))
		print("for (',', 'vincent', '_')", list(contexts_to_words[(',', 'vincent', '_')]))
		print("for ('collective', '_'): ", list(contexts_to_words[('collective', '_')]))
		print("for ('of', 'collective', '_')", list(contexts_to_words[('of', 'collective', '_')]))
		
		print("\nNow look at shared_context_matrix, making use of its related csr matrix (denoted here scm_csr)")
		
	# computing shared context master matrix
	#shared_context_matrix = context_array.dot(context_array.T).todense()
	scm_csr = context_array.dot(context_array.T)	#'scm' stands for 'shared_context_matrix'
	
	if verbose:
		print("scm_csr.shape =", scm_csr.shape)
		print("single_entry_rows.shape =", scm_csr[scm_csr.getnnz(1)==1].shape)
		print("single_entry_cols.shape =", scm_csr[:,scm_csr.getnnz(0)==1].shape)
		
		
		print("\nctxt_supported_wordlist.index('bargaining') =", ctxt_supported_wordlist.index('bargaining'))
		print("ctxt_supported_wordlist.index('payne') =", ctxt_supported_wordlist.index('payne'))
		print("ctxt_supported_wordlist.index('berger') =", ctxt_supported_wordlist.index('berger'))
		print("ctxt_supported_wordlist.index('greville') =", ctxt_supported_wordlist.index('greville'))
		
		barg_index = ctxt_supported_wordlist.index('bargaining')
		print("\nLook at indptr for rows near 'bargaining'")
		print("scm_csr.indptr[barg_index-1] =", scm_csr.indptr[barg_index-1])
		print("scm_csr.indptr[barg_index] =", scm_csr.indptr[barg_index])
		print("scm_csr.indptr[barg_index+1] =", scm_csr.indptr[barg_index+1])
		print("scm_csr.indptr[barg_index+2] =", scm_csr.indptr[barg_index+2])
		
		
		print("\nA row with a single nonzero entry corresponds to a word which shares contexts with no other word.") 
		print("Omit those rows from shared_context_matrix and those words from the given wordlist.")
		
	shared_ctxt_supported_wordlist = \
			[word for word in ctxt_supported_wordlist if
			(scm_csr.indptr[ctxt_supported_wordlist.index(word) + 1] - scm_csr.indptr[ctxt_supported_wordlist.index(word)]) > 1 ]
	
	if verbose:
		print("len(shared_ctxt_supported_wordlist) =", len(shared_ctxt_supported_wordlist))
		print("Result now is that each remaining word has shared contexts and eigenvectors have no anomalous entries.")
		#  POSSIBLY USE np.diff  No, what's used above seems clearer
		
	wordlist = shared_ctxt_supported_wordlist       # BE VERY CAREFUL
	n_words = len(shared_ctxt_supported_wordlist)
	
	# M = M[M.getnnz(1)>0][:,M.getnnz(0)>0] #GOOD
	scm_csr = scm_csr[scm_csr.getnnz(1) > 1][:,scm_csr.getnnz(0) > 1]    # audrey  March 1, 2018
	
	shared_context_matrix = scm_csr.todense()
	return shared_context_matrix, shared_ctxt_supported_wordlist
	

def compute_words_distance(coordinates):
    # the scipy pdist function is to compute pairwise distances
    return spatial.distance.squareform(
        #spatial.distance.pdist(coordinates, 'euclidean'))  # turns into a square matrix  word_list x word_list
        spatial.distance.pdist(coordinates, 'cosine'))  # turns into a square matrix  word_list x word_list    # AUDREY  2017_03_22
      
        #ON May 24, 2018 ~4:30 pm
        # The line that was in use is the 'cosine' line immediately above.
        # Trying now: use euclidean distance here.
        # Then normalize each row. REALIZED THIS IS WRONG PLACE. ROWNORMED RIGHT AFTER OBTAINING EIGENVECTORS.
        # First try just leaving all then in euclidean distance. May afterward try converting to cosine dist.
        # BEFORE CHANGING ANYTHING, SAVE A COPY OF OUTPUT - to compare nearest neighbors.
        # AND TRY AFTERWARD CHANGING n_eigenvectors TO A VALUE > 6 (current value).

def compute_closest_neighbors(word_distances, n_neighbors):
	sorted_neighbors = word_distances.argsort()
	# indices of sorted rows, low to high
    
    # For a given row, if distance between wordlist entry and a neighbor is < machine epsilon, 
	# sort may place the neighbor instead of the wordlist entry at the 0th position.
	# This actually happens for wordlist entry 'jr' and its neighbor 'etc', using cosine distance,
	# both of which occur if max_word_types=2000. Many more occur for higher values of max_word_types.
	
	# ADDITIONAL INFO from numpy.sort  documentation   says that 'mergesort' is stable   2018_01_06
	
	#print("\nChecking lines of sorted_neighbors array for correct 0th entry...")
	for linenum in range(len(sorted_neighbors)):    # len( ) returns number of rows
		if not (sorted_neighbors[linenum, 0] == linenum):
			line = sorted_neighbors[linenum]
			#print("linenum is ", linenum)
			#print("line is ", line)
			##current_loc = line.index(linenum)
			##current_loc = (line == linenum)
			current_loc = line.tolist().index(linenum)
			#print("current_loc of linenum is ", current_loc)
			line[0], line[current_loc] = linenum, line[0]
			#print("after exchange, line is ", line)
			#print()
	
	# truncate columns at n_neighbors + 1
	nearest_neighbors = sorted_neighbors[:, :n_neighbors + 1]
	return nearest_neighbors


def compute_graph( words_to_neighbors):
    graph = nx.Graph()
    for word in words_to_neighbors.keys():
        neighbors = words_to_neighbors[word]
        for neighbor in neighbors:
            graph.add_edge(word, neighbor)
    return graph
    
    
# Not in use but retained for now   June 16, 2019 
def Chicago_get_laplacian(affinity_matrix):
	n_words = affinity_matrix.shape[0]
	diam_array = compute_diameter_array(n_words, affinity_matrix)		# Revise - omit n_words parameter
	incidence_graph = compute_incidence_graph(n_words, diam_array, affinity_matrix)
	laplacian_matrix = compute_laplacian(diam_array, incidence_graph)
	del incidence_graph
	return laplacian_matrix, np.sqrt(diam_array)
	
	
####### HERE BEGINS DECOMPOSITION 
####### RETAINED HERE WHEN MUCH OTHER UNUSED CODE WAS REMOVED  JUNE 2019
####### EXPECT TO WORK ON THIS SOON
####### REMEMBER TO WORK THROUGH KEVIN'S SLIDES ON LATENT DIRICHLET ALLOCATION
####### MOSTLY FOR WORDBREAKER BUT ALSO FOR TOPIC MODELING   
def decomp_skdl(eigenarray):    #learn_dictionary_sk_dl(eigenarray):    #decomp_skdl(eigenarray):  
	# These letters are used below to identify array shapes
	#  C = length of wordlist
	#  N = dimension of wordvector space
	#  M = number of atoms
	# eigenarray is (C x N)
	print('DictionaryLearning (from sklearn) ...')
	
	# Set parameters
	atoms_param  = 7     #7  #10  #15  #20
	sparse_param = 2     #2  #3
	
	t0 = time.time()
	dl_object = DictionaryLearning(n_components=atoms_param, transform_n_nonzero_coefs=sparse_param, transform_algorithm='omp', fit_algorithm='lars')
	atoms_array = dl_object.fit(eigenarray).components_    # (M x N) 
	coeff_array = dl_object.transform(eigenarray)          # (C x M)
	dt = time.time() - t0
	
	s1 = "Algorithm: sklearn DictionaryLearning (using omp and lars)"
	s2 = "%d iterations" % dl_object.n_iter_
	s3 = "done in %.2f seconds" % dt
	header = "\n".join((s1, s2, s3))				# ALSO ERROR ???
	
	atoms = atoms_array.T	           # tranpose for consistency with other algorithms (esp. NMF)
	codes = coeff_array.T
	return atoms, codes, header
	
	
#DEPRECATED  
#def decomp_skdl(data_array, wordvecs, inds):   
#	D, coeffs_entire_wordlist, header = learn_dictionary_sk_dl(data_array)  # put the code inline instead
#	extracted_coeffs = coeffs_entire_wordlist[inds, :]
#	#codes = sparse_encode(wordvecs, D, algorithm='omp', n_nonzero_coefs=3)
#	coeffs = sparse_encode(wordvecs, D, algorithm='omp', n_nonzero_coefs=2)
#	print('sparse coeffs \n', coeffs)
#	print('extracted_coeffs \n', extracted_coeffs) 
#	
#	return D, coeffs, header
	
	
def decomp_ica(eigenarray):
	# These letters are used below to identify array shapes
	#  C = length of wordlist
	#  N = dimension of wordvector space
	#  M = number of atoms		# For ICA, M = N
	# eigenarray is (C x N)	
	
	print("running FastICA...")
	t0 = time.time()
	ica_object = FastICA(whiten=False)
	coeff_array = ica_object.fit_transform(eigenarray)     # C x M
	atoms_array = ica_object.components_                    # M x N    # unmixing array
	#atoms = ica_object.mixing_                              # N x M
	dt = time.time() - t0
	
	#print("\natoms_array:")
	#print(atoms_array)
	#print("\ncoeff_array:")
	#print(coeff_array)
	
	s1 = "Algorithm: FastICA(whiten=False)"
	s2 = "%d iterations" % ica_object.n_iter_
	s3 = "done in %.2f seconds" % dt
	header = "\n".join((s1, s2, s3))				# ALSO ERROR ???
	
	atoms = atoms_array.T		# tranpose for consistency with other algorithms (esp. NMF)
	codes = coeff_array.T
	return atoms, codes, header


# NMF - Nonnegative matrix factorization (nonnegativity and sparsity constraints)
	
def calculate_objval(V, W, H):					# "reconstruction error"
	return .5 * np.sum(np.square(V - W @ H))	# square each entry in the array, then add them all up

# NOT IN USE   November 2017	
def calculate_L1_setting(sprs, M, l2):			# NOTE: Define sparsity here!
	sqrtM = math.sqrt(M)
	k1 = (sprs + sqrtM - sprs*sqrtM) * l2		# obtained by solving the sparseness expression for its L1 variable,
	return k1									# then plugging in values sprs and l2 as appropriate.
	# This is based on Hoyer's definition of sparsity (Hoyer_04, page 1460):
	# For vector v of dim M,  sparseness(v) = (sqrtM - L1(v)/L2(v)) / (sqrtM - 1)
	
	
# Note: Every column vector h of H has same length M. 
# If M were passed in, it would avoid repeated calculation of M and its square root.
# Furthermore, (sprs + sqrtM - sprs*sqrtM) could be calculated once, passed to here instead of sprs_req. SHOULD DO THIS.
# However this way is safer, as M is not actually a parameter.
#MORE ON THIS FROM SEPT. 22, 2017 IN EVERNOTE Wordvectors3
#Wnew(:,i) = projfunc(Wnew(:,i),L1a*norms(i),(norms(i)^2),1); 
#    I’m just handling this differently; but made me realize I could compute  (sprs + sqrtM - sprs*sqrtM) 
#    and pass that value CLEARLY LABELLED AS DEP ON M AND SPRS instead of sprs_req
#    likewise maybe pass M also CLEARLY LABELLED as dimh or hdim

# NOT IN USE  November 2017
def hoyer_plus_kludges_apply_constraints(h, sprs, c_index):					
	# Find the vector g closest to h which has the following properties:
	#	all entries in g are nonnegative
	#   sparseness(g) = sprs

	verbose = False
	if c_index == 161:							# wordlist[161] = 'take'
		verbose = True							# A: k1, B: l2,  C: nonneg    Z: normal return   P, Q, R: exceptions
		
	M  = len(h)									# i.e., dimension of vector h
	rootM = math.sqrt(M)						# for convenience
	l2 = np.linalg.norm(h)						# L2 norm of h
	l1 = np.linalg.norm(h, ord=1)				# L1 norm of h
	k1 = calculate_L1_setting(sprs, M, l2)
	
	if verbose:
		print("\nIn apply_constraints, for c_index = ", c_index)
		print("Vector h:", h)
		print("l2 = ", l2, ",  	k1 = ",  k1, ",   l1 = ", l1)
	
	if np.all(h>=0):
		L1L2ratio = l1/l2
		h_sparseness = (rootM - L1L2ratio) / (rootM - 1)
		if h_sparseness >= sprs:
			#if True:
			#	print("c_index =", c_index, "  early out")
			if verbose:
				print("Returning h -- already nonneg,  h_sparseness =", h_sparseness)
			return h, h_sparseness
			

	# To obtain the desired sparseness (namely sprs), construct g s.t. L2_norm(g) = l2 and L1_norm(g) = k1.
	# Then fix any resulting negative entries in g to zero.
	# Repeat until both sparseness and nonneg criteria satisfied (requires at most M passes, by method of construction).
	
	# Inits
	g = h											
	zeroed_positions = np.array([], dtype = np.int)
	#if verbose:
	#	print("zeroed_positions is")
	#	print(zeroed_positions)
	#	print("length of zeroed_positions is ", len(zeroed_positions))
	num_zeroed_positions = 0
	j = 0                                       # to count the iterations
	
	while True and j<(M+1):
		if verbose:
			print("\nIn constraining loop j = ", j, ":")
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)
			print("g sparseness = ", (rootM - L1L2ratio) / (rootM - 1))
		#print("What to add to each coordinate:")
		#print("M - num_zeroed_positions = ", M - num_zeroed_positions)
		#print("sum(abs(g)) =", sum(abs(g)))
		#print("k1 - sum(abs(g)) =", k1 - sum(abs(g)))
		if verbose:
			print("A: (k1 - sum(abs(g)))/(M - num_zeroed_positions) = ", (k1 - sum(abs(g)))/(M - num_zeroed_positions))
		g = g + (k1 - sum(abs(g)))/(M - num_zeroed_positions)		# projection of g onto hyperplane determined by the k1 constraint
		g[zeroed_positions] = 0										# and the zeroed position constraints
		if verbose:
			print("A: g for k1  is", g)
		
		basepoint = np.ones(M) * k1/(M - num_zeroed_positions)		# special point in same hyperplane
		basepoint[zeroed_positions] = 0
		if verbose:
			print("B: basepoint is", basepoint)
		
		# Find the point z on the radiant from the basepoint through g such that L2_norm(z) = l2.
		# To set this up, represent z as   g + rho*w,  where rho is a scalar and w = g - basepoint.
		# Solve for rho:  (g + rho*w) @ transpose(g + rho*w) = l2^2
		
		w = g - basepoint
		if verbose:
			print("B: w is        ", w)
		
		
		a = w @ w.T
		b = 2 * (g @ w.T)
		c = (g @ g.T) - l2*l2
		
		#if verbose:
		#	print("a = ", a)
		#	print("b = ", b)
		#	print("c = ", c)
			
		radicand = b*b-4*a*c
		if verbose:
			print("B: radicand = ", radicand)
			
		if radicand <= 0:									# SHOULD MEAN k1 PLANE, l2 SPHERE DON"T INTERSECT
			if True:   #verbose:
				print("\nBad radicand at c_index = ", c_index, ";   radicand = ", radicand, "  a=", a, " b=", b, "  c=", c)
				#exit()   # Turned out that a==0 (so divide by 0), also b==0, apparently due to g identically == basepoint
		
		if np.all(w==0):
			rho = 0
		else:
			rho = (-b + cmath.sqrt(b*b-4*a*c).real)/ (2*a)	# N.B. WHEN radicand < 0, THIS THROWS STUFF AWAY;
		g = g + rho*w									 	# OBSERVATIONALLY SETS g TO basepoint. NOT GOOD APPROX TO g, and WRONG L2 
		# g now satisfies the sparseness criterion			# WHEN radicand = 0, SIMILAR BEHAVIOR?
		#  -- not necessarily all nonnegative yet
		if verbose:
			print("B: rho = ", rho)
			print("B: g for l2 is", g)
			print("B: Desired l2 =", l2, "\n   L2_norm(g) =", np.linalg.norm(g), "\n   L2_norm(basepoint) =", np.linalg.norm(basepoint))
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)
			print("B: g sparseness = ", (rootM - L1L2ratio) / (rootM - 1))	
		
		if np.all(g>=0):
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)
			g_sparseness = (rootM - L1L2ratio) / (rootM - 1)
			if verbose:
				print("Z: nonneg g for return is", g)
				print("Z: g sparseness =", g_sparseness, "\n")
			return g, g_sparseness
		
		
		# Set negative entries to zero
		zeroed_positions = np.where(g<=0)[0]						# includes positions previously zeroed
		num_zeroed_positions = len(zeroed_positions)
		if num_zeroed_positions == M:
			if True:   #verbose   #True
				print("P: At c_index =", c_index, ", ALL Zeroes, returning g =", g)
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)
			g_sparseness = (rootM - L1L2ratio) / (rootM - 1)
			if True:   #verbose   #True
				print("P: g sparseness =", g_sparseness)
			return g, g_sparseness
			
		g[zeroed_positions] = 0
		
		#elif num_zeroed_positions == (M - 1):
		if num_zeroed_positions == (M - 1):
			for i in range(M):
				if i not in zeroed_positions:
					g[i] = l2
				break;
			if verbose:    #True:   #verbose:
				print("Q: At c_index =", c_index, ", nonzeroes 1, returning g =", g)
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)		# should be 1
			g_sparseness = (rootM - L1L2ratio) / (rootM - 1)			# should be 1
			if verbose:    #True:   #verbose:
				print("Q: g sparseness = ", g_sparseness)
			return g, g_sparseness
		
		if num_zeroed_positions == (M - 2):				 # ADDED 2017_09_23   Sparsity has been attained
			#if True:									 # Calling this "2 nonzero shortcut". I've been undecided about it.
			#	print("R: At c_index =", c_index, ", nonzeroes 2 - return ")
			if verbose:   #True:   #verbose:
				#print("R: At c_index =", c_index, ", nonzeroes 2 - go on")
				print("R: At c_index =", c_index, ", nonzeroes 2 - return ")
				print("R: g before norm change =", g)
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)
			g_sparseness = (rootM - L1L2ratio) / (rootM - 1)
			if verbose:   #True:   #verbose:
				print("R: g sparseness =", g_sparseness)
			g = (l2/np.linalg.norm(g)) * g
			if verbose:   #True:   #verbose:
				print("R: g after norm change  =", g)
			L1L2ratio = np.linalg.norm(g, ord=1)/np.linalg.norm(g)
			g_sparseness = (rootM - L1L2ratio) / (rootM - 1)
			if verbose:   #True:   #verbose:
				print("R: g sparseness = ", (rootM - L1L2ratio) / (rootM - 1))
			return g, g_sparseness

		j = j+1
		if verbose:
			print("C: zeroed_positions is", zeroed_positions)
			print("   len(zeroed_positions) =", len(zeroed_positions))
			print("   num_zeroed_positions = ", num_zeroed_positions)
			print("C: g with negs fixed is", g)
		
		#if num_zeroed_positions == (M-1):
		#	print("NUMBER OF ZEROED POSITIONS = M-1 for c_index = ", c_index)
		#	#exit()
		# I think that if we get down to <=2 nonzero positions we should accept that g without going back through the loop.
		# Actually we should probably adjust to make its L2 norm be l2.
		# The g itself will actually have the correct k1; not sure about what L1 norm of the adjusted g will be.
		# What I care about is decomposition and approximation to be good, not Hoyer's definition of sparseness; 
		# it may however affect the algorithm.
		# 
		# This brings up the idea that you don't need the sparseness to be exactly equal to the sparsity parameter, 
		# rather to be less than or equal.  OOPS
		# So the l1 value you're after is allowed to be <= the k1 value calculated based on the parameter.
		# If the k1 plane is outside the sphere, reduce your l1 requirement (and project to a lower plane than k1);
		# there is some room for this l1 to be greater than (1.0 * given l2) and still have the plane cut inside the l2 sphere.
		# The requirement is less tight (I think) when M is larger.
		# Quite possibly given M and given l2 for a particular h, we could calculate an l1 specification where l1 would be < k1
		# but would not produce a negative radicand.

# THIS WORKS, NEEDS TO BE CLEANED UP  November 2017		 # PROBABLY DON'T WANT specified_index ARGUMENT
def one_more_zero(g, c_index, specified_index = None):   # in future 'sprs' may not be required 
	# EXPLANATION 
	# Given a vector g with at least 2 nonzero entries, this function returns a nearby vector g_hat (with the same L2 norm) 
	# with a zero in the position of the most negative nonzero coordinate in g.
	# As used in the NMf algorithm, each application of this function will advance by one step 
	# the attainment of a sparse non-negative approximation for the original h. 
	
	# IS the average of h1, h2, h3 computed with signs or abs values?
	 
	verbose = False
	#if c_index == 127:
	#	verbose = True
	#	print("\n@@@@ c_index == 127 @@@@")
	
	Q = len(g)						# dimension of vector g;  in context of NMF algorithm, Q <= M, and decreases with successive applications
	sqrtQ = math.sqrt(Q)			# for convenience
	if verbose:
		print("Q =", Q, "   sqrtQ =", sqrtQ)
	
	# As described in    , we induce sparsity by reducing the L1 norm while maintaining the L2 norm.
	# For finding g_hat, then, we are concerned with two types of structure in R^Q:
	#   a hyperplane normal to the vector <1, 1, ..., 1> (all elements of which have the same L1 norm)
	# Note that all non-negative vectors with the same L1 norm lie in a hyperplane normal to the vector <1, 1, ..., 1>.
	
	#u  = ((g>=0) + -1 * (g<0)) / sqrtQ
	#idx = list(g).index(min(g))
	
	u  = np.ones(Q)/sqrtQ			# <1, 1, ..., 1> normalized
	saved_norm = np.linalg.norm(g)
	g1 = g/saved_norm
	#mindx = list(g).index(min(g))
	
	if verbose:
		print("u =", u)
		print("g =", g)
		#print("index of min entry in g =", mindx)
		print("saved_norm of g =", saved_norm)
		print("g1 =", g1)
	
	
	# Let S be the unit sphere in dimension M, 
	# S1 the great circle on S determined by u and g1,
	# and P the plane containing S1 (so u, g1, and the origin belong to subspace P). 
	# Let L be the line in P perpendicular to u passing through the origin.
	# Let k be the projection of g onto L, and j the normalization of k.  Then
	
	# Think of u as radius of unit circle in y direction.
	
	y = np.dot(g1,u)*u				# projection of g1 onto u   (Q-dim)
	x = g1 - y						# projection of g1 perpendicular to y (in proper 2d-plane)   # N.B. On Nov. 4, 2017 realized had been   g1 - np.dot(g, u)*u  !!!
	x1 = x/np.linalg.norm(x)		# unit vector in x direction
	if verbose:
		print("np.dot(g1, u) =", np.dot(g1, u))        # MAY SIMPLY FIND AVERAGE (BUT CORRECTLY)
		print("y = np.dot(g1, u)*u =", y)
		print("x = g1 - y =", x)
		print("norm of x =", np.linalg.norm(x))
		print("x1 = normalized x =", x1)
		
	if specified_index != None:
		zindx = specified_index
	else:
		zindx = list(x).index(min(x))	# position of least entry in all these vectors: g, g1, x, x1		
	nu = x1[zindx]
	if verbose:
		print("index of min entry in x1 is", zindx, ";	nu = x1[", zindx, "] =", nu)

	# scalars	
	subcalc = Q * nu * nu
	_lambda = math.sqrt( subcalc/(subcalc+1) )
	_kappa  = math.sqrt( 1/(subcalc+1) )
	if verbose:
		print("Q * nu * nu =", subcalc)
		print("_lambda =", _lambda)
		print("_kappa  =", _kappa)

	# Q-dim 
	lambda_u = _lambda * u
	kappa_x1 = _kappa * x1
	with_plus_K  = lambda_u + kappa_x1
	with_minus_K = lambda_u - kappa_x1
		
	if  abs(with_plus_K[zindx]) <= abs(with_minus_K[zindx]):
		g1_hat = with_plus_K
	else:
		g1_hat = with_minus_K
		
	if verbose:
		print("_lambda * u =", _lambda*u)
		print("_kappa * x1 =", _kappa*x1)
		if g1_hat[zindx] == with_plus_K[zindx]:
			print("PLUS \ng1_hat =", g1_hat)
		else:
			print("MINUS \ng1_hat =", g1_hat)
		print("norm of g1_hat =", np.linalg.norm(g1_hat))
	
	if abs(g1_hat[zindx]) < 1e-10:
		g1_hat[zindx] = 0.0
	else:
		print("c_index =", c_index, ":\tg1_hat[zindx] =", g1_hat[miindx], ",\tabs value =", abs(g1_hat[zindx]), "\tNOT sufficiently close to 0.0")
	
	delete_idx = np.array(list(g1_hat[:zindx]) + list(g1_hat[zindx+1:]))
	if np.any(delete_idx < 0):
		print("c_index =", c_index, ":\tdelete_idx has a negative;   g =", g)
	elif np.any(delete_idx < 1e-10):
		print("c_index =", c_index, ":\tdelete_idx has another near-zero")    # This is probably OK, but for now may wish to know.
	#if np.all(except_idx >= 0):
	#	print("c_index =", c_index, ":  g =", g)
		
	g_hat = g1_hat * saved_norm
	if verbose:
		print("g_hat = g1_hat * saved_norm =", g_hat)
		
	return g_hat, zindx
	
	
def apply_constraints(g, level, c_index):
	# needs stopping condition
	#print("\nA: At top of level", level, ", g =\t", g)
	g_hat, zindx = one_more_zero(g, c_index)		# the second coord is irrelevant right now--just needed to fill it
	#print("B: one_more_zero(g) returned g_hat =", g_hat, ",\tzindx =", zindx)		# 'zindx' means the zeroed index	
	#print("C: len(g_hat) is", len(g_hat))
	if len(g_hat) <= 3:
		#print("At bottom of recursion: returning g_hat because length==3")
		return g_hat
	f = np.array(list(g_hat[:zindx]) + list(g_hat[zindx+1:]))		# This deletes the zeroed coordinate
	#print("D: Here is the reduced vector  f before apply_constraints(f):\t", f)
	f_rebuilt = apply_constraints(f, level+1, c_index)
	#print("\nBack to level", level)
	#print("E: Here is the rebuilt version of f returned by apply_constraints(f):\t", f_rebuilt)
	g_rebuilt = np.array(list(f_rebuilt[:zindx]) + list([0]) + list(f_rebuilt[zindx:]))
	#print("F: returning upward  g_rebuilt =\t", g_rebuilt)
	return g_rebuilt
	
	
	
#def apply_constraints(updated_h, previous_h, sprs_req, c_index):   #DO THE NORMALIZING IN THIS FUNCTION
#	h_hat = one_more_zero(updated_h, c_index)  
#	# AND SUBSPACE CONVERSIONS
#	#if doesn''t work:
#	#	h_hat = previous_h			# will not be sparser, but will be positive
#	#print(c_index)
#		
		

# NOW THAT constraints CODE WORKS, WORK HERE TO TRY TO GET GOOD RESULTS	
def decomp_nmf(eigenarray):
	# These letters are used below to identify array shapes  NO - THESE WILL BE DEFINED
	#  C = number of wordvectors
	#  N = dimension of wordvectors (i.e., wordvectors reside in R^N)
	#  M = number of atoms
	# eigenarray is (C x N)
	
	# Write description here
	# arg min W, H   Minimize "reconstruction error" - |  |, with 
	#           non-negat
	#           sparse
	#           ||w|| = 1
	#     alternating min
	# use mlwiki for structure     and cite
	# use Hoyer2004 paper and matlab code for algorithm  and cite
	# Optimizing H requires gradient descent because of sparseness and non-negativity constraints; for W  a simpler multiplicative update suffices.
	
	# Meaning of parameters
	# 	max_iter					limit on # of iterations
	# 	min_stepsize				for gradient descent: 
	#									Within each iteration, the stepsize is adaptively decreased till it results
	#									in an improvement over the preceding iteration for the objective function.
	#									The algorithm terminates when the step shrinks to min_stepsize with no further
	#									improvement to the objective.
	#	num_atoms					number of atoms
	#	sprs_req					specifies desired sparseness (for coefficients)
	#									For the specific measure of sparseness defined by Hoyer and 
	#									used in this program, see calculate_L1_setting() function above.
	#									       
	
	print("\nAt entry, wordvector for 'turn' is", eigenarray[412,:])
	print("CHECK THE SIGNS!")
	
	# Set parameters      possibly should send these in by keyword, with defaults
	max_iter = 40		#25  #40     #200	#5  
	min_stepsize = 1e-20		#1e-200
	sprs_req = .9		#.9	 #.8   #.85
	num_atoms = 8		#6	#!0	#3	#7	#8	#5	#12
	
	
	np.random.seed(5)		#5   # only for development if needed
	# ATTN: Hoyer rescales data to avoid potential overflow/underflow
	
	V = eigenarray.T		# N x C
	# V = skl_normalize(V, norm='l2', axis=0) 	# Consider this again later.
	
	# Convention: the following letters will be used consistently across arrays V, W, H 
	# to help clarify their construction
	N = V.shape[0]			# dimension of wordvectors (i.e., wordvectors reside in R^N)
	C = V.shape[1]			# number of wordvectors 
	M = num_atoms
	
	
	# Build initial version of W 
	# Requirements: each column should be a unit vector with first coord nonnegative
	
	# THIS IS FOR N==2 ONLY  (AS EXPERIMENT)   NOT LIKELY as of Sept. 2017
	#W = np.zeros((N, M))
	#increment = math.pi/M
	#start = -math.pi * .5
	#for i in range(M):
	#	theta = start + i*increment
	#	W[0,i] = math.cos(theta)
	#	W[1,i] = math.sin(theta)
		
	
	# RANDOMIZING APPROACH 
	W = np.zeros((N, M))
	W[0, :] = np.random.rand(1, M)					# 1st row: floats sampled from [0.0, 1.0)
	W[1:N, :] = -1.0 + 2*np.random.rand(N-1, M)		# remaining rows: floats sampled from [-1.0, 1.0)
	W = skl_normalize(W, norm='l2', axis=0)
	
	print("\nCompleted initial version of W")
	print("Should be N x M,   columns unit length, first coord nonnegative")
	print(W)
	
	# Build initial version of H
	# Requirements: all entries nonnegative; each column sparse
	H = np.random.rand(M, C)
	Htmp_sparseness = np.zeros(C)
	#for c in range(C):		#columnwise                   # TRY OMITTING THIS. October 23, 2017
	#	H[:,c], Htmp_sparseness[c] = apply_constraints(H[:,c], sprs_req, c)
		
	print("\nCompleted initial version of H. ")
	print("Should be M X C,  all entries nonnegative, column sparsity=", sprs_req)
	print(H)
	
	#print("\nInitially, H[:,259] = ", H[:,259])
	#print("Initially, H[:,50] = ", H[:,50])
	#print("Initially, H[:,161] = ", H[:,161])
	#print("V[:,161] - W @ H[:,161] =", V[:,161] - W @ H[:,161])
	
	## VERY DUBIOUS   November 24, 2017   Check removing Dec. 4, 2017
	#max_abs_V = max(abs(V.flatten()))
	#V = V/max_abs_V
	
	
	# init
	Err = calculate_objval(V, W, H)
	print("Initial value of objective =", Err)
	iter = 0		
	stepsizeW = 1
	stepsizeH = 1
	#min_stepsize_reached = False
	delta_obj = 1
	t0 = time.time()
	
	#Loop 1	
	while (stepsizeW >= min_stepsize) and stepsizeH >= min_stepsize and (abs(delta_obj)>1e-9) and (iter < max_iter):
	#while (stepsizeW >= min_stepsize) and (math.fabs(delta_obj)>1e-9) and (iter < max_iter):
	#while (stepsize >= min_stepsize) and (math.fabs(delta_obj)>1e-9) and (iter < 1):
		iter = iter + 1
		print("\niter = ", iter)
		delta_obj = -Err
				
		# ------------UPDATE W ----------------------
		# -------by gradient descent-----------------

		if False:
			
			Errtmp = Err									# init
			dW = (W @ H - V) @ H.T
			
			print("\nAt beginning of W UPDATE for iter", iter, ", Err =", Err)
			print("dW =")
			print(dW)
			#time.sleep(7)
			
			#max_entry = max(abs(dW.flatten()))
			#scale_factor = .01/max_entry									#1.0/max_entry	#.5/max_entry
			#dW = scale_factor * dW
			#print("max(dW) =", max_entry)
			#print("scale_factor * dW =") 
			#print(dW)
			
			#Loop2
			passnum = 0
			while not Errtmp < Err:							# don't step so far in the domain that the obj value fails to improve
				passnum = passnum + 1
				if stepsizeW < min_stepsize:					# -- but if it can't be improved, best solution (locally) has been reached
					#min_stepsize_reached = True
					print("passnum", passnum, ": min_stepsize reached for W without improving objective")
					break;
					
				stepsizeW = .5 * stepsizeW					# reduce stepsize
				Wtmp = W - stepsizeW * dW					# gradient descent step
				print("\nIn (Loop1) iter =", iter, ", (Loop 2) passnum =", passnum, ", stepsizeW =", stepsizeW)
				print("stepsizeW * dW =")
				print(stepsizeW * dW)
				print("Wtmp = W - stepsizeW * dW =")
				print(Wtmp)
				
				for m in range(M):
					if Wtmp[0,m] < 0:
						Wtmp[:,m], zindex = one_more_zero(Wtmp[:,m], m, specified_index=0)		# The 'm' argument is for tracking [for dev only]
				
				#INSERT
				#print("\nBefore norm operations")
				#print("W", W)		#print(W)
				Wnorms = np.linalg.norm(W, axis=0)
				#print("Wnorms:", Wnorms)
				Wnorms = np.expand_dims(Wnorms, axis=1)
				print("Wnorms:") 
				print(Wnorms)
				#print("H", H)
							
				#W = skl_normalize(W, norm='l2', axis=0)			# normalize each column (i.e., each atom)
				#H = H * (Wnorms @ np.ones((1, C)))				# Adjust H to preserve product W @ H
				#TO HERE
				
				
				##### THIS MUCH SHOULD BE DELETED #####
				Wtmp = skl_normalize(Wtmp, norm='l2', axis=0)			# normalize each column (i.e., each atom)
				print("Wtmp with columns normalized =") 
				print(Wtmp)
				checklengths  = [np.linalg.norm(Wtmp[:,m]) for m in range(M)]
				print("checklengths: ", checklengths)
				
				
				H = H * (Wnorms @ np.ones((1, C)))				# Adjust H to preserve product W @ H
				print("H after norm operations:")
				print(H)
				##### TO HERE #####
				
				#oldErrtmp = Errtmp
				Errtmp = calculate_objval(V, Wtmp, H)
				#print("\n(Loop1) iter =", iter, ", (Loop2) passnum =", passnum, ", stepsize =", stepsize, " - after apply_constraints()")
				print("Errtmp = ", Errtmp)
				#time.sleep(7)
				#print("avg sparseness = ",  np.mean(Htmp_sparseness))
				#if Errtmp > oldErrtmp:
				#    print("Errtmp increased in Loop2 from passnum", passnum-1, "to passnum", passnum, ";  was", oldErrtmp, ", now = ", Errtmp, "   stepsize = ", stepsize, "   iter = ", iter)
				#    stepsize = .001 * stepsize
				#    #time.sleep(10)
				
			if Errtmp<Err:
				W = Wtmp
				Err = Errtmp
				#delta_obj = delta_obj + Err						# change in obj value due to updates for both W and H 
				stepsizeW = stepsizeW * 1.2						# why?
			else:
				print("In W update, Err could not be reduced. Retained W unchanged after H update.")
			
			
			print("\nAfter W update, iter", iter, ":")
			print("   objval = Err =", Err)
			#print("   avg sparseness = ", H_avg_sparseness)
			print("   W = ")
			print(W)

		
		
		# ----by multiplicative update rule----------        try also gradient descent
		if False:											# was   if else:
			W = W * (V @ H.T) / (W @ H @ H.T)
			#W = W * (V @ H.T) / (W @ H @ H.T + 1e-9)		# '@' is linear algebra multiplication
											                # '*' and '/' are elementwise operations
											                				
											                
			#print("\nBefore norm operations")
			#print("W", W)		#print(W)
			Wnorms = np.linalg.norm(W, axis=0)
			#print("Wnorms:", Wnorms)
			Wnorms = np.expand_dims(Wnorms, axis=1)
			#print("Wnorms:", Wnorms)
			#print("H", H)
			
			W = skl_normalize(W, norm='l2', axis=0)			# normalize each column (i.e., each atom)
			H = H * (Wnorms @ np.ones((1, C)))				# Adjust H to preserve product W @ H
			#print("\nAfter norm operationa")
			#print("W", W)
			#print("H", H)
			#print("\nprod_after = ", W @ H)
			
			
			Err = calculate_objval(V, W, H)
			print("\nAfter W update and H norm adjustment, iter", iter, ":")
			print("   objval = Err =", Err)
			print("   W = ")
			print(W)
			print("   H = ")
			print(H)
			
			#print("adjusted after W update, H[:,259] = ", H[:,259])
			#print("adjusted after W update, H[:,50] = ", H[:,50])
			#print("adjusted after W update, H[:,2462] = ", H[:,2462])
			
		# ----by multiplicative update by portion if needed----------
		if True:
		
			full_update = W * (V @ H.T) / (W @ H @ H.T)
			portion = 2.0									# init
			Errtmp = Err
			
			#Loop2W
			passnumW = 0
			while not Errtmp < Err:							# Should we just require the Error to be non-increasing?
				passnumW = passnumW + 1
				if portion < min_stepsize:					# -- but if it can't be improved, best solution (locally) has been reached
					#min_stepsize_reached = True
					print("passnumW", passnumW, ": min_stepsize reached for W full_update portion without improving objective")
					break;
					
				portion = .5 * portion						# May want factor some value other than .5
				Wtmp = W + portion*(full_update - W)
				
				Wtmp_norms = np.linalg.norm(Wtmp, axis=0)
				Wtmp_norms = np.expand_dims(Wtmp_norms, axis=1)
				
				Wtmp = skl_normalize(Wtmp, norm='l2', axis=0)			# normalize each column (i.e., each atom)
				Htmp = H * (Wtmp_norms @ np.ones((1, C)))				# Adjust H to preserve product W @ H
				
				Errtmp = calculate_objval(V, Wtmp, Htmp)
				
				print("\nErrtmp =", Errtmp, "in (Loop1) iter =", iter, ", (Loop2W) passnumW =", passnumW, ", portion =", portion)
				
			if Errtmp<Err:
				W = Wtmp
				H = Htmp
				Err = Errtmp
			else:
				print("In W update, Err could not be reduced. Retained W unchanged.")
				
				
			print("\nErr =", Err, " at end of W update, iter", iter)
			print("W = ")
			print(W)
			print("H adjusted (by W norms) = ")
			print(H)
				
				
		# ------------UPDATE H ----------------------
		# ----by multiplicative update rule----------
		if False:
			H = H * (W.T @ V) / (W.T @ W @ H + 1e-9)		# '@' is linear algebra multiplication
			Htmp_sparseness = np.zeros(C)					# '*' and '/' are elementwise operations
			
			for c in range(C):
				H[:,c], Htmp_sparseness[c] = apply_constraints(H[:,c], sprs_req, c)    # INCLUDE 'c' FOR DEVELOPEMENT
				
			Err = calculate_objval(V, W, H)
			delta_obj = delta_obj + Err
			print("\nAfter H update,")
			print("   objval = ", Err)
			print("   avg sparseness = ", np.mean(Htmp_sparseness))
			print("   H = ")
			print(H)
			
			print("adjusted after H update, H[:,259] = ", H[:,259])
		
		
		# ------------UPDATE H ----------------------
		# -------by gradient descent----------------- 
		if True:											# else:
			dH = W.T @ (W @ H - V)
			#Errtmp = Err + 1								# init
			Errtmp = Err
			
			print("\nAt beginning of H UPDATE  for iter", iter, ", Err =", Err)
			print("dH =")
			print(dH)
			#time.sleep(7)
			
			#Loop2H
			passnumH = 0
			stepsizeH = 2.0 * stepsizeH						# 2017_11_26  This is an experiment to try not decreasing the stepsize unless necessary--
															# so not on first pass through Loop2H.
															# May also change .5 to something larger -- say .7 
															
			while not Errtmp < Err:							# don't step so far in the domain that the obj value fails to improve
				passnumH = passnumH + 1
				if stepsizeH < min_stepsize:					# -- but if it can't be improved, best solution (locally) has been reached
					#min_stepsize_reached = True
					print("passnum", passnum, ": min_stepsize reached for H without improving objective")
					break;
					
				stepsizeH = .5 * stepsizeH					# reduce stepsize
				Htmp = H - stepsizeH * dH					# gradient descent step
				Htmp_sparseness = np.zeros(C)
								
				
				print("\nIn (Loop1) iter =", iter, ", (Loop 2H) passnumH =", passnumH, ", stepsizeH =", stepsizeH, " - before apply_constraints()")
				for c in range(C):
				#for c in range(2):
				# modify each column to satisfy constraints
					#print("\nFor apply_constraints, c = ", c)
					#for m in range(M):		# OMIT - DON'T RETAIN ZEROES THROUGH GRAD DESCENT
					#if H[m,c]==0:
					#		Htmp[m,c] = 0
					Htmp[:,c] = apply_constraints(Htmp[:,c], 1, c)		# The '1' is the recursion level for apply_constraints() [for dev] . 
																		# The 'c' is for tracking image, esp for one_more_zero() [for dev]
																		# By now (Nov. 8, 2017), I think one_more_zero() is fine, 
																		# so c_index for verbose may soon be omitted.
					
					# OR MAY USE:
					# h = H[:,c]
					# htmp = Htmp[:,c]
					# htmp = htmp * (h>0)                    # (h>0) is a Boolean array (see Boolean mask)
					#   ... apply_constraints(htmp, sprs_req, c)
					#   ... apply_constraints(Htmp[:,c] * (H[:,c] > 0), sprs_req, c)     # more briefly
				#print("V[:,161] - W @ Htmp[:,161] =", V[:,161] - W @ Htmp[:,161])
				# exit()   # FOR TESTING one_more_zero()
					
				
				#oldErrtmp = Errtmp
				Errtmp = calculate_objval(V, W, Htmp)
				print("\n(Loop1) iter =", iter, ", (Loop2) passnumH =", passnumH, ", stepsizeH =", stepsizeH, " - after apply_constraints()")
				print("Errtmp = ", Errtmp)
				##time.sleep(7)
				##print("avg sparseness = ",  np.mean(Htmp_sparseness))
				#if Errtmp > oldErrtmp:
				    #print("Errtmp increased in Loop2 from passnum", passnum-1, "to passnum", passnum, ";  was", oldErrtmp, ", now = ", Errtmp, "   stepsizeH = ", stepsizeH, "   iter = ", iter)
				    #stepsizeH = .001 * stepsizeH
				    ##time.sleep(10)
					##print("Errtmp increased! Exit now")
					##s1 = "Algorithm NMF - exit from Update H because Errtmp increased"
					##s2 = "Err = %f   oldErrtmp = %f   Errtmp = %f" % (Err, oldErrtmp, Errtmp)
					##s3 = "iter = %d   stepsize = %f" % (iter, stepsize)
					##seq = (s1, s2, s3)
					##header = "\n".join(seq)
					##return W, Htmp, header
					###exit()
					
				
			if Errtmp<Err:
				H = Htmp
				#H_avg_sparseness = np.mean(Htmp_sparseness)
				Err = Errtmp
				delta_obj = delta_obj + Err						# change in obj value due to updates for both W and H 
				#stepsizeH = stepsizeH * 1.2					# COMMENTED OUT ON 2017_11_26 - See 2017_11_26 COMMENT ABOVE	# why?
			else:
				print("In H update, Err could not be reduced. Retained H unchanged after W update.")
			
			
			print("\nAfter H update, iter", iter, ":")
			print("   objval = Err =", Err)
			#print("   avg sparseness = ", H_avg_sparseness)
			print("   H = ")
			print(H)
			

	dt = time.time() - t0
	
	s1 = "Algorithm: Nonnegative matrix factorization with sparsity"
	s2 = "%d iterations" % iter
	s3 = "done in %.2f seconds" % dt
	header = "\n".join((s1, s2, s3))				# ALSO ERROR ???
		
	#return atoms, codes, header
	return W, H, header
	
	

def learn_sparse_repr(eigenarray):
	# EDIT TO SPECIFY ALGORITHM (convenience during development)

	alg_label = "nmf"		#"ica"    #"sk_dl"
	#atoms, codes, header = decomp_skdl(eigenarray)
	#atoms, codes, header = decomp_ica(eigenarray)
	atoms, codes, header  = decomp_nmf(eigenarray)
	
	return atoms, codes, header, alg_label
	


def study_decomposition(wordlist, eigenarray, atoms, codes, header, alg_label, timestamp, num_neighbors=20):
	# N.B. eigenarray included here only to get wordvector coordinates 
	# in order to exhibit both wordvector and its decomposition

	#EDIT TO SPECIFY WORDS (convenience during development)
	the_words = ['start', 'turn', 'work', 'call', 'look', 'give', 'find', 'bring', 'take', 'do']
	#the_words = ['city', 'basis', 'lady', 'dog', 'region', 'memory', 'cat']   # pure_nouns
	the_inds = [wordlist.index(word) for word in the_words]
	the_wordvecs = eigenarray[the_inds, :]
	print('\n', the_words)
	#print(the_wordvecs)
	
	the_coeffs = codes[:, the_inds]
	
	# OUTPUT TO FILE     
	outfilename = "sparse_repr_info." + alg_label + "." + timestamp.strftime("%Y_%m_%d.%H_%M") + ".csv"
	outfile = open(outfilename, mode='w')
	print(header, file=outfile)
	
	C = len(the_words)
	N = atoms.shape[0]         # dimension of wordvector space
	M = atoms.shape[1]         # number of atoms
	
	#vec_dim   = D.shape[1]          # length of wordvector = length of atom = n_eigenvectors
	#num_atoms = D.shape[0]          # same as codes[1]
	#num_words = len(the_words)      # same as codes.shape[0]
	
	print('\nWORDVECTORS', file=outfile)
	for c in range(C):
		print(the_words[c], end='', file=outfile)
		for n in range(N):
			print(', %.15f' % the_wordvecs[c, n], end='', file=outfile)
		print(file=outfile)
	
	print('\nDICTIONARY', file=outfile)    # Columns are atoms
	for n in range(N):	                   # nth line will show nth coordinate in wordvector space
		for m in range(M):	               # across every atom
			print(', %.15f' % atoms[n, m], end='', file=outfile)
		print(file=outfile)
	
	print('\nCOEFFICIENTS for decomposition', file=outfile)
	for c in range(C):
		print(the_words[c], end='', file=outfile)
		for m in range(M):
			print(', %.15f' % the_coeffs[m, c], end='', file=outfile)
		print(file=outfile)
		
	atoms_to_neighbors = dict()
	longer_array = np.concatenate((eigenarray, atoms.T), axis=0)
	
	# TEMPORARY    ALSO CHANGE M+2 BACK TO M   satisfied for now; may check more axis vectors later
	#two_more = np.array(([1,0], [0,1]))
	#longer_array = np.concatenate((longer_array, two_more), axis=0)
	# TEMPORARY
	
	distances = compute_words_distance(longer_array)
	nearest_neighbors_idxform = compute_closest_neighbors(distances, num_neighbors)
	
	neighbors_of_atoms_idxform = nearest_neighbors_idxform[len(wordlist):, 1:]
	
	#for i in range(len(wordlist), len(wordlist)+M):
	#	line = nearest_neighbors[i]
	#	atom = "atom_" + str(i)
	#	neighbors_idx = line[1:]
	#	neighbors = [wordlist[idx] for idx in neighbors_idx]			# NOTE - An atom could occur among the neighbors
	#	atoms_to_neighbors  NO   make an array   Can do this in matrix form  Fix it tomorrow.
	
	print('\n\nWORD NEIGHBORS for atoms', file=outfile)
	for t in range(num_neighbors):
		for m in range(M):
		#for m in range(M+2):
			tth_neighbor_idxform = neighbors_of_atoms_idxform[m, t]
			if tth_neighbor_idxform >= len(wordlist):
				print("For t =", t, "and m =", m, "in 'WORD NEIGHBORS for atoms', the index is beyond length of wordlist.") 
				#print(',atom_%d' % m, end='', file=outfile)
				which_atom = tth_neighbor_idxform - len(wordlist)
				print(',atom_%d' % which_atom, end='', file=outfile)
			else:
				print(',%s' % wordlist[tth_neighbor_idxform], end='', file=outfile)  # replace by the word
		print(file=outfile)
	

	outfile.close()
	
# INCORPORATED INTo study_decomposition()
def investigate_atoms(wordlist, eigenvectors, atoms, num_neighbors=20):
	longer_array = np.concatenate((eigenvectors, atoms.T), axis=0)
	num_eig = eigenvectors.shape[0]
	num_atoms = atoms.shape[1]
	num_entries = longer_array.shape[0]
	print("number of eigenvectors =", num_eig)
	print("number of atoms =", num_atoms)
	print("longer_array.shape[0] =", num_entries)
	print("longer_array.shape[1] =", longer_array.shape[1])
	for i in range(num_entries - num_atoms - 5, num_entries):
		print(longer_array[i])
		
	print()
	
	distances = compute_words_distance(longer_array)
	nearest_neighbors = compute_closest_neighbors(distances, num_neighbors)
	
	words_to_neighbors = dict()
	words_to_neighbor_distances = dict()
	
	for i in range(num_entries - num_atoms, num_entries):
		line = nearest_neighbors[i]
		word_idx, neighbors_idx = line[0], line[1:]
		word = "atom" + str(i)
		neighbors = [wordlist[idx] for idx in neighbors_idx]			# NOTE - An atom could occur among the neighbors
		neighbor_distances = [distances[i, idx] for idx in neighbors_idx]   # AUDREY  2017_03_28  IS THIS RIGHT? word_distances[word_idx, idx] for idx in neighbors_idx
		words_to_neighbors[word] = neighbors
		words_to_neighbor_distances[word] = neighbor_distances
		print(word, ": \t", neighbors)
		
####### HERE ENDS DECOMPOSITION   AS OF JUNE 2019


def discretize(vectors, alg_id, common_data_for_spreadsheets, copy=True, max_svd_restarts=30, n_iter_max=25, random_state=None, log=False):
	
	wordlist = common_data_for_spreadsheets[0]
	diameter = common_data_for_spreadsheets[1]
	output_dir = common_data_for_spreadsheets[2]
	timestamp_string = common_data_for_spreadsheets[3]
	
	if log == True:
		# CREATE WORKBOOK AND SET UP FORMATS (using xlsxwriter)
		workbook = xlsxwriter.Workbook(output_dir + alg_id  + '.n=' + str(vectors.shape[1]) + timestamp_string + ".xlsx")
		
		LUT = [
			'#646464',    #nice charcoal
			'#96addb',    #blue violet
			'#ffff00',	  #yellow       '#B27FB2', #light magenta         #FF00FF',	#magenta		#c0c0c0', #silver
			'#ff00ff',    #fuchsia
			'#2c8840',    #deeper green
			'#5ed1b7',    #nice aqua
			'#0000ff',  #blue
			'#c0c0c0',  #silver			'#ffff00',  #yellow
			'#00ffff',  #aqua
			'#800000',  #maroon
			'#008000',  #green
			'#6666ff',  #was navy  #000080
			'#808000',  #olive
			'#800080',  #purple
			'#008080',  #teal
			'#808080',  #gray
			'#00ff00',  #lime
			'#8B4513',  #saddlebrown
			'#d15eb7',
			'#2c5088',
			'#004040',
			'#400040',
			#'#ff0000',  #red
			#'#fa3fc9',  #pinker fuchsia
			]
			
		fill_format = []
		for i in range(len(LUT)):
			fill_format.append(workbook.add_format({'bg_color': LUT[i]}))
			#marker_format.append( workbook.add_format({'border': {'color': LUT[i]}, 'fill': {'color': LUT[i]}) )
			
		# DOWN TO HERE
		bold = workbook.add_format({'bold': True})
		float_format = workbook.add_format({'num_format':'0.00000000'})
		merge_format = workbook.add_format({'align': 'center', 'bold': True})
	
	#############################
	# THE ALGORITHM BEGINS HERE #
	#############################
	random_state = sklearn.utils.check_random_state(random_state)
	vectors = sklearn.utils.as_float_array(vectors, copy=copy)
	eps = np.finfo(float).eps
	n_samples, n_components = vectors.shape
	
	###### I THINK THIS STEP IS NOT IN Yu-Shi PAPER.  AT 12:33 A.M. Dec. 14, 2018, COMMENT IT OUT!
	# Normalize the eigenvectors to an equal length of a vector of ones.
	# Reorient the eigenvectors to point in the negative direction with respect
	# to the first element.  This may have to do with constraining the
	# eigenvectors to lie in a specific quadrant to make the discretization
	# search easier.
	
	#norm_ones = np.sqrt(n_samples)
	#for i in range(vectors.shape[1]):
		#vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones    #This line IS in Shi code. Try with.FALSE!!
		#if vectors[0, i] != 0:
			#vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])
		
	# Normalize the rows of the eigenvectors.  Samples should lie on the unit
	# hypersphere centered at the origin.  This transforms the samples in the
	# embedding space to the space of partition matrices.
	#vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]
	vectors = skl_normalize(vectors, norm='l2', axis=1)		# row => unit-length
	
	
	svd_restarts = 0
	has_converged = False
	
	# If there is an exception we try to randomize and rerun SVD again
	# do this max_svd_restarts times.
	while (svd_restarts < max_svd_restarts) and not has_converged:
		# Initialize first column of rotation matrix with a row of the
		# eigenvectors
		rotation = np.zeros((n_components, n_components))
		rotation[:, 0] = vectors[random_state.randint(n_samples), :].T
		
		# To initialize the rest of the rotation matrix, find the rows
		# of the eigenvectors that are as orthogonal to each other as
		# possible
		make_rotation_column = np.zeros(n_samples)    # changed name from 'c' to 'make_rotation_column' ('c' occurs in spreadsheet code below) 
		for j in range(1, n_components):
			# Accumulate c to ensure row is as orthogonal as possible to
			# previous picks as well as current one
			make_rotation_column += np.abs(np.dot(vectors, rotation[:, j - 1]))
			rotation[:, j] = vectors[make_rotation_column.argmin(), :].T
			
			
		if log==True:
			# ADD worksheet0 TO SHOW INITIAL VECTOR MATRIX AND ROTATION MATRIX
			worksheet0 = workbook.add_worksheet('0')
			(C, N) = vectors.shape
			worksheet0.set_column(5, 2*N+5, 12)
			
			# Column headings
			worksheet0.write(0, 0, 'Index', bold)		# A1
			worksheet0.write(0, 1, 'Wordlist', bold)	# B1
			worksheet0.write(0, 2, 'Cluster', bold)		# C1
			worksheet0.write(0, 3, 'Diameter', bold)	# D1
			
			for n, col in zip(range(N), range(5,N+5)):
				worksheet0.write(0, col, "SoftIndVec_" + str(n), bold)  #Do rows sum to 1?
				
			# Data
			for c in range(C):
				row = c + 1
				worksheet0.write(row, 0, c)
				worksheet0.write(row, 1, wordlist[c])
				worksheet0.write(row, 2, 'not yet')
				worksheet0.write(row, 3, diameter[c])
				
				#skip one column
				for n, col in zip(range(N), range(5,N+5)):
					worksheet0.write(row, col, vectors[c, n], float_format)
					
			for n in range(N):
				row = n + 1
				for m, col in zip(range(N), range(N+6, 2*N+6)):
					worksheet0.write(row, col, rotation[n, m], float_format)
			# END OF worksheet0 SECTION
		
		
		last_objective_value = 0.0
		n_iter = 0
		
		while not has_converged:
			n_iter += 1
			
			#t_discrete = np.dot(vectors, rotation)   # use of 'discrete' name was misleading
			#labels = t_discrete.argmax(axis=1)
			
			t_continuous = np.dot(vectors, rotation)
			labels = t_continuous.argmax(axis=1)
			
			vectors_discrete = csc_matrix(
				(np.ones(len(labels)), (np.arange(0, n_samples), labels)),
				shape=(n_samples, n_components))
				
			t_svd = vectors_discrete.T * vectors
			
			try:
				U, S, Vh = np.linalg.svd(t_svd)
				svd_restarts += 1
			except LinAlgError:
				print("SVD did not converge, randomizing and trying again")
				break
				
			ncut_value = 2.0 * (n_samples - S.sum())
			print("ncut_value = ", ncut_value)
			if ((abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max)):
				has_converged = True
			else:
				# otherwise calculate rotation and continue
				last_objective_value = ncut_value
				rotation = np.dot(Vh.T, U.T)
				
			
			if log==True:
				# ADD A WORKSHEET TO RECORD THE CONTINUOUS AND DISCRETE MATRICES FOR THIS ITERATION
				worksheet1 = workbook.add_worksheet('iter '+str(n_iter))
				(C, N) = vectors.shape
				worksheet1.set_column(5, 2*N+5, 12)         # 1st column, last column, width
				
				# Column headings
				worksheet1.write(0, 0, 'Index', bold)		# A1
				worksheet1.write(0, 1, 'Wordlist', bold)	# B1
				worksheet1.write(0, 2, 'Cluster', bold)		# C1
				worksheet1.write(0, 3, 'Diameter', bold)	# D1
				
				for n, col in zip(range(N), range(5,N+5)):
					worksheet1.write(0, col, "SoftIndVec_" + str(n), bold)  #Do rows sum to 1?
					
				for n, col in zip(range(N), range(N+6, 2*N+6)):
					worksheet1.write(0, col, "IndVec_" + str(n), bold)
					
				# Data
				for c in range(C):
					row = c + 1
					worksheet1.write(row, 0, c)
					worksheet1.write(row, 1, wordlist[c])
					worksheet1.write(row, 2, labels[c], fill_format[ labels[c] ])  # IS THIS CORRECT?
					worksheet1.write(row, 3, diameter[c])
					
					#skip one column
					for n, col in zip(range(N), range(5,N+5)):
						worksheet1.write(row, col, t_continuous[c, n], float_format)    		#soft_partition[c, n]
						
					#skip one column
					for n, col in zip(range(N), range(N+6, 2*N+6)):
						worksheet1.write(row, col, vectors_discrete[c, n], float_format)		#partition[c, n]
				# END OF worksheet SECTION
			
	
	if not has_converged:
		raise LinAlgError('SVD did not converge')
		
	if log==True:
		workbook.close()
		
	# MAYBE SHOULD RETURN vectors also  (which would have unit-length rows)
	#return vectors_discrete.toarray(), labels
	return t_continuous, labels
	
	
def spectral_clustering_menu(eigenvectors, eigenvalues, common_data_for_spreadsheets, random_seed=None):
	# For any matrix named as eigenvectors_xxx in this program, 
    # each column is an eigenvector of some particular sort of laplacian
    #               (or the result of an operation on these eigenvectors)
    # each row represents a word as a tuple in that coordinate system
    
    # THINGS TO KEEP IN MIND
    # weighted k-means
    # any other weighting scheme; rescaling eigenvectors
    # regularize
    # nonnegative constraint
    # data spectroscopy
    # hierarchical scheme, repeated n=2
    # hierarchical scheme with uncertainty, 3-way partition
    # use coherence criterion w/any algorithm
    
	# PRELIMINARIES
	(C, N) = eigenvectors.shape
	diameter  = common_data_for_spreadsheets[1]
	sqrt_diam = np.sqrt(diameter)	# 1-dim ndarray
	eigenvectors_unitrows = skl_normalize(eigenvectors, norm='l2', axis=1)
	clustercenters_placeholder = np.zeros((N, N))	# for use by algorithms other than kmeans
	
		
	# PREPARE INPUT DATA FOR ASSORTED ALGORITHMS
	eigenvectors_sym = eigenvectors								# columns are unit-length eigenvectors of L_sym
	eigenvectors_rw  = eigenvectors / sqrt_diam[:, np.newaxis]	# divide each element in row i by sqrt_diam[i]	
																# columns are e-vectors of L_rw with e-values same as sym
																# not unit length
																
	eigenvectors_sym_rw_unitrows = eigenvectors_unitrows		# normalizing rows of e-vectors_sym and e-vectors_rw gives same entries
	eigenvectors_rw_1          = skl_normalize(eigenvectors_rw, norm='l2', axis=0)
	eigenvectors_rw_1_unitrows = skl_normalize(eigenvectors_rw_1, norm='l2', axis=1)
	
	# This section is specifically for diffusion clustering.
	# For now, start from laplacian. In future work directly from affinity matrix.
	diffusion_embedding_rw = np.empty_like(eigenvectors_rw)			# init
	diffusion_eigenvalues = np.ones_like(eigenvalues) - eigenvalues
	print("\neigenvalues:", eigenvalues)
	print("diffusion_eigenvalues:", diffusion_eigenvalues)
	
	num_timesteps = 1    # 5 timesteps, 12 eigs   usually 1    3   20    30
	timestep_coeffs = np.zeros((num_timesteps, N))   # can I specify format? 
	timestep_coeffs[0, :] = diffusion_eigenvalues
	for i in range(1, num_timesteps):
		timestep_coeffs[i, :] = timestep_coeffs[i-1, :] * diffusion_eigenvalues		# element-wise
		print("\n", timestep_coeffs[i, :])

	
	#diffusion_embedding_rw = eigenvectors_rw @ np.diag(diffusion_eigenvalues)		# 1 timestep
	#diffusion_embedding_rw_1 = eigenvectors_rw_1 @ np.diag(diffusion_eigenvalues)
	
	diffusion_embedding_rw = eigenvectors_rw @ np.diag(timestep_coeffs[num_timesteps-1, :])		# 5 timesteps  then 10 timesteps
	diffusion_embedding_rw_1 = eigenvectors_rw_1 @ np.diag(timestep_coeffs[num_timesteps-1, :])
	
	print("\neigenvectors_rw:")
	print(eigenvectors_rw[0:5, :])
	print("\ndiffusion_embedding_rw")
	print(diffusion_embedding_rw[0:5, :])    # first 5 words
	print("\n")
		
	print("\neigenvectors_rw_1:")
	print(eigenvectors_rw_1[0:5, :])
	print("\ndiffusion_embedding_rw_1")
	print(diffusion_embedding_rw_1[0:5, :])
	print("\n")
	
	# raise SystemExit    # March 7, 2019
	
	
	# CLUSTERING ALGORITHMS 
	###################################################################################
	# KMeans   Use the object-oriented version for access to cluster centers.    
	# 4 input variants: sym requires unit rows [Ng et al.], rw, rw_1, rw_1 unit rows   Try also sym - separates high-diam words
	# rw unit rows not needed: same as sym unit rows.                                  DO WE NEED TO MAKE COPIES? (see def discretize)
	###################################################################################
	
	#kmeans_clustering
	nreq_clusters = 9		# N  3  8  9  11  15  19  # should eventually be a user parameter setting
	
	diffusion_kmeans_clustering_rw = sklearn.cluster.KMeans(n_clusters=nreq_clusters, random_state=random_seed).fit(diffusion_embedding_rw)
	diffusion_clusterlabels_rw  = diffusion_kmeans_clustering_rw.labels_
	diffusion_clustercenters_rw = diffusion_kmeans_clustering_rw.cluster_centers_
	generate_eigenvector_spreadsheet('diffusion_kmeans.rw', diffusion_embedding_rw, nreq_clusters, diffusion_clusterlabels_rw, diffusion_clustercenters_rw, common_data_for_spreadsheets)
	
	diffusion_kmeans_clustering_rw_1 = sklearn.cluster.KMeans(n_clusters=nreq_clusters, random_state=random_seed).fit(diffusion_embedding_rw_1)
	diffusion_clusterlabels_rw_1  = diffusion_kmeans_clustering_rw_1.labels_
	diffusion_clustercenters_rw_1 = diffusion_kmeans_clustering_rw_1.cluster_centers_
	generate_eigenvector_spreadsheet('diffusion_kmeans.rw_1', diffusion_embedding_rw_1, nreq_clusters, diffusion_clusterlabels_rw_1, diffusion_clustercenters_rw_1, common_data_for_spreadsheets)
	
	#kmeans_clustering_sym_rw_unitrows = sklearn.cluster.KMeans(n_clusters=nreq_clusters, random_state=random_seed).fit(eigenvectors_sym_rw_unitrows)
	#clusterlabels_sym_rw_unitrows  = kmeans_clustering_sym_rw_unitrows.labels_
	#clustercenters_sym_rw_unitrows = kmeans_clustering_sym_rw_unitrows.cluster_centers_
	#generate_eigenvector_spreadsheet('kmeans.sym_rw_unitrows', eigenvectors_sym_rw_unitrows, nreq_clusters, clusterlabels_sym_rw_unitrows, clustercenters_sym_rw_unitrows, common_data_for_spreadsheets)
	
	#kmeans_clustering_rw = sklearn.cluster.KMeans(n_clusters=nreq_clusters, random_state=random_seed).fit(eigenvectors_rw)
	#clusterlabels_rw  = kmeans_clustering_rw.labels_
	#clustercenters_rw = kmeans_clustering_rw.cluster_centers_
	#generate_eigenvector_spreadsheet('kmeans.rw', eigenvectors_rw, nreq_clusters, clusterlabels_rw, clustercenters_rw, common_data_for_spreadsheets)
	
	#kmeans_clustering_rw_1 = sklearn.cluster.KMeans(n_clusters=nreq_clusters, random_state=random_seed).fit(eigenvectors_rw_1)
	#clusterlabels_rw_1  = kmeans_clustering_rw_1.labels_
	#clustercenters_rw_1 = kmeans_clustering_rw_1.cluster_centers_
	#generate_eigenvector_spreadsheet('kmeans.rw_1', eigenvectors_rw_1, nreq_clusters, clusterlabels_rw_1, clustercenters_rw_1, common_data_for_spreadsheets)
	
	#kmeans_clustering_rw_1_unitrows = sklearn.cluster.KMeans(n_clusters=nreq_clusters, random_state=random_seed).fit(eigenvectors_rw_1_unitrows)
	#clusterlabels_rw_1_unitrows  = kmeans_clustering_rw_1_unitrows.labels_
	#clustercenters_rw_1_unitrows= kmeans_clustering_rw_1_unitrows.cluster_centers_
	#generate_eigenvector_spreadsheet('kmeans.rw_1_unitrows',  eigenvectors_rw_1_unitrows, nreq_clusters, clusterlabels_rw_1_unitrows, clustercenters_rw_1_unitrows, common_data_for_spreadsheets)
		
	
	###################################################################################
	# discretize                check code carefully against algorithm in paper
	# Iteration within this algorithm begins with a matrix with rows of length one
	# (derived from a laplacian eigenvector matrix). To enforce this, the first step
	# in the discretize() code (obtained from sklearn) is to row-normalize the input.  
	# For insurance, we retain this line, noting here that using either a given 
	# eigenvector matrix or its row-normalized form as input produces the same result.	 
	###################################################################################
	#nreq_clusters = N
	
	# NOT YET TRIED
	#diffusion_discr_rot_rw, diffusion_discr_clusterlabels_rw = discretize(diffusion_embedding_rw, 'discretize_LOG.sym_rw_unitrows', common_data_for_spreadsheets, random_state=random_seed, log=False)
	#generate_eigenvector_spreadsheet('diffusion_discr_rw', diffusion_discr_rot_rw, nreq_clusters, diffusion_discr_clusterlabels_rw, clustercenters_placeholder, common_data_for_spreadsheets)
	
	#diffusion_discr_rot_rw_1, diffusion_discr_clusterlabels_rw_1 = discretize(diffusion_embedding_rw_1, 'discretize_LOG.sym_rw_unitrows', common_data_for_spreadsheets, random_state=random_seed, log=False)
	#generate_eigenvector_spreadsheet('diffusion_discr_rw_1', diffusion_discr_rot_rw_1, nreq_clusters, diffusion_discr_clusterlabels_rw_1, clustercenters_placeholder, common_data_for_spreadsheets)
	
	
	#rotated_vectors_sym_rw_unitrows, clusterlabels_sym_rw_unitrows = discretize(eigenvectors_sym_rw_unitrows, 'discretize_LOG.sym_rw_unitrows', common_data_for_spreadsheets, random_state=random_seed, log=False)
	#generate_eigenvector_spreadsheet('discretize.sym_rw_unitrows', rotated_vectors_sym_rw_unitrows, nreq_clusters, clusterlabels_sym_rw_unitrows, clustercenters_placeholder, common_data_for_spreadsheets)
	
	#rotated_vectors_rw_1_unitrows, clusterlabels_rw_1_unitrows = discretize(eigenvectors_rw_1_unitrows, 'discretize_LOG.rw_1_unitrows', common_data_for_spreadsheets, random_state=random_seed, log=False)
	#generate_eigenvector_spreadsheet('discretize.rw_1_unitrows', rotated_vectors_rw_1_unitrows, nreq_clusters, clusterlabels_rw_1_unitrows, clustercenters_placeholder, common_data_for_spreadsheets)
	
	
# This function no longer called directly. Covered instead by code inside "spectral_clustering_menu()"
# Retain as of June 16, 2019  for revising spectral_clustering_menu( )
def spectral_clustering_sym(eigenvectors, common_data_for_spreadsheets, assign_labels='kmeans', random_seed=None):
	# For any matrix named as eigenvectors_xxx in this program, 
    #  each column is an eigenvector of some particular sort of laplacian
    #  each row represents a word as a tuple in that coordinate system
    
    # At input, columns are L_sym unit-length eigenvectors.
    # L_rw eigenvectors 
    
    # 
    # Get the word vectors for clustering
    wordcoords_sym = skl_normalize(eigenvectors, norm='l2', axis=1)		# row => unit-length
    
    # Step 3: apply clustering algorithm
    if assign_labels == 'kmeans':
    	# Use the object-oriented version for access to cluster centers. Mostly we take the default parameter values.
    	kmeans_clustering = sklearn.cluster.KMeans(n_clusters=eigenvectors.shape[1], random_state=random_seed).fit(wordcoords_sym)
    	clusterlabels  = kmeans_clustering.labels_
    	clustercenters = kmeans_clustering.cluster_centers_
    	
    if assign_labels == 'discretize':
    	wordcoords_sym, clusterlabels = discretize(wordcoords_sym, 'sym.', common_data_for_spreadsheets, random_state=random_seed)
    	N = eigenvectors.shape[1]
    	clustercenters = np.zeros((N, N))	# relevant only for kmeans
    
    return wordcoords_sym, clusterlabels, clustercenters
    
    
# This function no longer called directly. Covered instead by code inside "spectral_clustering_menu()"
# Retain as of June 16, 2019  for revising spectral_clustering_menu( )
def spectral_clustering_rw(eigenvectors, sqrt_diam, common_data_for_spreadsheets, assign_labels='kmeans', random_seed=None):
	# For any matrix named as eigenvectors_xxx in this program, 
    #  each column is an eigenvector of some particular sort of laplacian
    #  each row represents a word as a tuple in that coordinate system
	
	# Step 1: Obtain eigenvectors of L_rw     [solution of "generalized eigenvector problem":  Lu = lambda D u]
	eigenvectors_rw = eigenvectors / sqrt_diam[:, np.newaxis]			    # At input, columns are L_sym unit-length eigenvectors.
	eigenvectors_rw = skl_normalize( eigenvectors_rw, norm='l2', axis=0 )   # Consider whether or not to normalize columns
	
	# Step 2: Get the word vectors for clustering 
	wordcoords_rw = eigenvectors_rw		# algorithm does not call for row-based modification
	#wordcoords_rw = skl_normalize( eigenvectors_rw, norm='l2', axis=1 )	# Try row-normalization instead of column-normalization
																			# Note changes column 0 => non-constant
																			# Note this occurs inside discretize() code
	
	# Step 3: apply clustering algorithm
	if assign_labels == 'kmeans':          # NOTE  Yu-Shi may imply that rows should be unit length here also. Try it.
		# Use the object-oriented version for access to cluster centers. Mostly we take the default parameter values.
		kmeans_clustering = sklearn.cluster.KMeans(n_clusters=eigenvectors.shape[1], random_state=random_seed).fit(wordcoords_rw)
		clusterlabels  = kmeans_clustering.labels_
		clustercenters = kmeans_clustering.cluster_centers_
		
	if assign_labels == 'discretize':
		wordcoords_rw, clusterlabels = discretize(wordcoords_rw, 'rw.', common_data_for_spreadsheets, random_state=random_seed)
		N = eigenvectors.shape[1]			#
		clustercenters = np.zeros((N, N))	# relevant only for kmeans
		
	return wordcoords_rw, clusterlabels, clustercenters
	


#def generate_eigenvector_spreadsheet(algorithm, eigenvectors, nreq_clusters, cluster_labels, cluster_centers, wordlist, diameter, output_dir, timestamp_string):
def generate_eigenvector_spreadsheet(alg_id, eig_input, nreq_clusters, cluster_labels, cluster_centers, common_data_for_spreadsheets):
	# Now handled through xlsxwriter
	
	# PRELIMINARIES
	wordlist = common_data_for_spreadsheets[0]
	diameter = common_data_for_spreadsheets[1]
	output_dir = common_data_for_spreadsheets[2]
	timestamp_string = common_data_for_spreadsheets[3]
	
	(C, N) = eig_input.shape
	print("In generate_eigenvector_spreadsheet, C = ", C, ",  N =", N)    
	
	#eig_rel = np.ones_like(eig_input)
	#for k in range(N):
		#eig_rel[:,k] = eig_input[:,k] / eig_input[:,0]
	
	
	# USING xlsxwriter
	workbook = xlsxwriter.Workbook(output_dir + alg_id + ".eig_data.nreq=" + str(nreq_clusters) + ".n=" + str(N) + timestamp_string + ".xlsx")
		
	# THIS IS NEW  November 8, 2018
	LUT = [
		'#646464',    #nice charcoal
		'#96addb',    #blue violet
		'#ffff00',	  #yellow       '#B27FB2', #light magenta         #FF00FF',	#magenta		#c0c0c0', #silver
		'#ff00ff',    #fuchsia
		'#2c8840',    #deeper green
		'#5ed1b7',    #nice aqua
		'#0000ff',  #blue
		'#c0c0c0',  #silver			'#ffff00',  #yellow
		'#00ffff',  #aqua
		'#800000',  #maroon
		'#008000',  #green
		'#6666ff',  #was navy  #000080
		'#808000',  #olive
		'#800080',  #purple
		'#008080',  #teal
		'#808080',  #gray
		'#00ff00',  #lime
		'#8B4513',  #saddlebrown
		'#d15eb7',
		'#2c5088',
		'#004040',
		'#400040',
		#'#ff0000',  #red
		#'#fa3fc9',  #pinker fuchsia
		]
		
	fill_format = []
	for i in range(len(LUT)):
		fill_format.append(workbook.add_format({'bg_color': LUT[i]}))
		#marker_format.append( workbook.add_format({'border': {'color': LUT[i]}, 'fill': {'color': LUT[i]}) )
		
	# DOWN TO HERE
	bold = workbook.add_format({'bold': True})
	float_format = workbook.add_format({'num_format':'0.00000000'})
	merge_format = workbook.add_format({'align': 'center', 'bold': True})
	
	worksheet1 = workbook.add_worksheet('Wordlist order')
	worksheet2 = workbook.add_worksheet('Sort by coord value')
	worksheet3 = workbook.add_worksheet('cluster.coord value')
	worksheet4 = workbook.add_worksheet('cluster.wordlist index')
	worksheet5 = workbook.add_worksheet('cluster.diameter')
	worksheet6 = workbook.add_worksheet('cluster.no 2nd sort')
	worksheet7 = workbook.add_worksheet()  # this one is for cluster centers and anything else I record, 
	# maybe means as I calculate. Also 2d scatter plot of all clusters.
	
	##############
	# WORKSHEET1 #
	##############
	
	worksheet1.set_column(5, 2*N+4, 12)         # 1st column, last column, width
	
	# Column headings
	worksheet1.write(0, 0, 'Index', bold)		# A1
	worksheet1.write(0, 1, 'Wordlist', bold)	# B1
	worksheet1.write(0, 2, 'Cluster', bold)		# C1
	worksheet1.write(0, 3, 'Diameter', bold)	# D1
	
	for n, col in zip(range(N), range(5,N+5)):
		worksheet1.write(0, col, "Eigenvector" + str(n), bold)
		
	#for n, col in zip(range(1,N), range(N+6, 2*N+5)):
		#worksheet1.write(0, col, "Eig" + str(n) + " / Eig0", bold)
	
	# Data
	for c in range(C):
		row = c + 1		
		worksheet1.write(row, 0, c)
		worksheet1.write(row, 1, wordlist[c])
		worksheet1.write(row, 2, cluster_labels[c], fill_format[ cluster_labels[c] ])
		worksheet1.write(row, 3, diameter[c])
		
		#skip one column  
		for n, col in zip(range(N), range(5,N+5)):
			worksheet1.write(row, col, eig_input[c, n], float_format)
			
		#skip one column
		#for n, col in zip(range(1,N), range(N+6, 2*N+5)):
			#worksheet1.write(row, col, eig_rel[c, n], float_format)  
			## Recall from above that eig_rel[c,n] = eig_input[c, n]/eig_input[c,0]
			
	##############
	# WORKSHEET2 #
	##############
	## SORT EACH EIGENVECTOR FIRST BY COORD VALUE (INCREASING), THEN BY CLUSTER, USING LEXSORT
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix
	#lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.lexsort((diameter, cluster_labels, eig_input[:,n]))
	
	#for n in range(1,N):
		#lex_sortation_rel[:,n] = np.lexsort((diameter, cluster_labels, eig_rel[:,n]))
	
	
	## SORT EACH EIGENVECTOR BY COORD VALUE (INCREASING), USING ARGSORT 
	#arg_sortation = np.argsort(eig_input, axis=0, kind='mergesort')    # C x N matrix
	#arg_sortation_rel =  np.argsort(eig_rel, axis=0, kind='mergesort')    # also C x N, disregard leftmost column
	
	
	# Default value for column width is too small to display desired number of digits.
	# Total number of columns used (on this worksheet) = 1+ N*5 + (N-1)*5 -1 = (2N-1)*5.
	# In xlsxwriter, indexing starts at 0.
	worksheet2.set_column(0, (2*N-1)*5-1, 11)		# 1st column, last column, width 
	
	## Column headings
	worksheet2.write(0, 0, 'Diameter', bold)
	
	for n in range(N):
		col = 1 + n*5
		worksheet2.merge_range(0, col, 0, col+2, "Eig"+str(n), merge_format)
		worksheet2.write(0, col+3, 'Eig'+str(n)+' w/labels', bold)
		
	#for n in range(1,N):
		#col = 1+ N*5 + (n-1)*5
		#worksheet2.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		#worksheet2.write(0, col+3, 'Eig'+str(n)+'/Eig0 w/labels', bold)
	
		
	## Data
	for c in range(C):
		row = c + 1
		
		indx = lex_sortation[c,0]
		#indx = arg_sortation[c,0]
		worksheet2.write(row, 0, diameter[indx])
		
		for n in range(N):
			#indx = arg_sortation[c,n]
			indx = lex_sortation[c,n]
			col = 1+n*5
			worksheet2.write(row, col, indx)
			worksheet2.write(row, col+1, wordlist[indx])
			worksheet2.write(row, col+2, eig_input[indx,n], float_format)
			worksheet2.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
		#for n in range(1,N):
			##indx = arg_sortation_rel[c,n]
			#indx = lex_sortation_rel[c,n]
			#col = 1 + N*5 + (n-1)*5
			#worksheet2.write(row, col, indx)
			#worksheet2.write(row, col+1, wordlist[indx])
			#worksheet2.write(row, col+2, eig_rel[indx,n], float_format)
			#worksheet2.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
	# Save the min and max values for eig_input to set up axes for plots in Worrksheet7
	if N>1:
		print("N =", N)
		minindex_by_eig = lex_sortation[0,:]		# slices
		maxindex_by_eig = lex_sortation[C-1,:]
		print("minindex_by_eig: ", minindex_by_eig)
		print("maxindex_by_eig: ", maxindex_by_eig)
		#print("Eig1: ", eig_input[minindex_by_eig[1], 1], eig_input[maxindex_by_eig[1], 1])
		#print("Eig2: ", eig_input[minindex_by_eig[2], 2], eig_input[maxindex_by_eig[2], 2])
		
	
	##############
	# WORKSHEET3 #
	##############
	## SORT EACH EIGENVECTOR FIRST BY CLUSTER, THEN BY COORD VALUE (INCREASING), USING LEXSORT
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix  
	#lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.lexsort((diameter, eig_input[:,n], cluster_labels))
		
	#for n in range(1,N):
		#lex_sortation_rel[:,n] = np.lexsort((diameter, eig_rel[:,n], cluster_labels))
	
	
	# xlsxwriter instructions are similar to those for Worksheet2, above
	worksheet3.set_column(0, (2*N-1)*5-1, 11)   #set column width
	
	## Column headings
	for n in range(N):
		col = 1 + n*5
		worksheet3.merge_range(0, col, 0, col+2, "Eig"+str(n), merge_format)
		worksheet3.write(0, col+3, 'Clusters @Eig'+str(n), bold)
		
	#for n in range(1,N):
		#col = 1+ N*5 + (n-1)*5
		#worksheet3.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		#worksheet3.write(0, col+3, 'Clusters @Eig'+str(n)+'/Eig0', bold)
	
	
	## Data  	#Note that the cluster_labels column is same for every eigenvector, 
				#hence could be generated once and then copied.
				
	for c in range(C):
		row = c + 1
		
		for n in range(N):    # for each eigenvector:
			indx = lex_sortation[c,n]
			col = 1 + n*5
			worksheet3.write(row, col, indx)
			worksheet3.write(row, col+1, wordlist[indx])
			worksheet3.write(row, col+2, eig_input[indx,n], float_format)
			worksheet3.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
		#for n in range(1,N):
			#indx = lex_sortation_rel[c,n]
			#col = 1 + N*5 + (n-1)*5
			#worksheet3.write(row, col, indx)
			#worksheet3.write(row, col+1, wordlist[indx])
			#worksheet3.write(row, col+2, eig_rel[indx,n], float_format)
			#worksheet3.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
	
	## Plots
	for n in range(N):
		clstr_pcwise_chart = workbook.add_chart({'type': 'scatter'})
		clstr_pcwise_chart.add_series({
			'name':       'Eig'+str(n)+' - coord value',
			'categories': ['Wordlist order', 1, 0, C, 0],	              #'Wordlist order' is name of Sheet1
			'values':     ['cluster.coord value', 1, 5*n+3, C, 5*n+3],    #'cluster.coord value' is name of Sheet3
			'marker':     {'type': 'short_dash', 'size': 2, 'border': {'color': 'red'}, 'fill': {'color': 'red'}},
		})
		
		worksheet3.insert_chart(2*n+10, 5*n+1, clstr_pcwise_chart)
		
	
	##############
	# WORKSHEET4 #
	##############
	## SORT EACH EIGENVECTOR FIRST BY CLUSTER, THEN BY WORDLIST INDEX (INCREASING), USING LEXSORT
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix  
	#lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.lexsort((np.arange(C), cluster_labels))
		
	#for n in range(1,N):
		#lex_sortation_rel[:,n] = np.lexsort((np.arange(C), cluster_labels))
		
	
	worksheet4.set_column(0, (2*N-1)*5-1, 11)   #set column width
	
	## Column headings
	for n in range(N):
		col = 1 + n*5
		worksheet4.merge_range(0, col, 0, col+2, "Eig"+str(n), merge_format)
		worksheet4.write(0, col+3, 'Clusters @Eig'+str(n), bold)
		
	#for n in range(1,N):
		#col = 1+ N*5 + (n-1)*5
		#worksheet4.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		#worksheet4.write(0, col+3, 'Clusters @Eig'+str(n)+'/Eig0', bold)
		
	
	## Data  	#Note that the cluster_labels column is same for every eigenvector, 
				#hence could be generated once and then copied.
				
	for c in range(C):
		row = c + 1
		
		for n in range(N):    # for each eigenvector:
			indx = lex_sortation[c,n]
			col = 1 + n*5
			worksheet4.write(row, col, indx)
			worksheet4.write(row, col+1, wordlist[indx])
			worksheet4.write(row, col+2, eig_input[indx,n], float_format)
			worksheet4.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
		#for n in range(1,N):
			#indx = lex_sortation_rel[c,n]
			#col = 1 + N*5 + (n-1)*5
			#worksheet4.write(row, col, indx)
			#worksheet4.write(row, col+1, wordlist[indx])
			#worksheet4.write(row, col+2, eig_rel[indx,n], float_format)
			#worksheet4.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
	
	## Plots
	for n in range(N):
		clstr_pcwise_chart = workbook.add_chart({'type': 'scatter'})
		clstr_pcwise_chart.add_series({
			'name':       'Eig'+str(n)+' - wordlist index',
			'categories': ['Wordlist order', 1, 0, C, 0],
			'values':     ['cluster.wordlist index', 1, 5*n+3, C, 5*n+3],
			'marker':     {'type': 'short_dash', 'size': 2, 'border': {'color': 'blue'}, 'fill': {'color': 'blue'}},
		})
		
		worksheet4.insert_chart(2*n+10, 5*n+1, clstr_pcwise_chart)
		
	
	##############
	# WORKSHEET5 #
	##############
	## SORT EACH EIGENVECTOR FIRST BY CLUSTER, THEN BY DIAMETER (INCREASING), USING LEXSORT
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix  
	#lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.lexsort((diameter, cluster_labels))
		
	#for n in range(1,N):
		#lex_sortation_rel[:,n] = np.lexsort((diameter, cluster_labels))
		
	
	worksheet5.set_column(0, (2*N-1)*5-1, 11)   #set column width
	
	## Column headings
	for n in range(N):
		col = 1 + n*5
		worksheet5.merge_range(0, col, 0, col+2, "Eig"+str(n), merge_format)
		worksheet5.write(0, col+3, 'Clusters @Eig'+str(n), bold)
		
	#for n in range(1,N):
		#col = 1+ N*5 + (n-1)*5
		#worksheet5.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		#worksheet5.write(0, col+3, 'Clusters @Eig'+str(n)+'/Eig0', bold)
		
	
	## Data  	#Note that the cluster_labels column is same for every eigenvector, 
				#hence could be generated once and then copied.
				
	for c in range(C):
		row = c + 1
		
		for n in range(N):    # for each eigenvector:
			indx = lex_sortation[c,n]
			col = 1 + n*5
			worksheet5.write(row, col, indx)
			worksheet5.write(row, col+1, wordlist[indx])
			worksheet5.write(row, col+2, eig_input[indx,n], float_format)
			worksheet5.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
		#for n in range(1,N):
			#indx = lex_sortation_rel[c,n]
			#col = 1 + N*5 + (n-1)*5
			#worksheet5.write(row, col, indx)
			#worksheet5.write(row, col+1, wordlist[indx])
			#worksheet5.write(row, col+2, eig_rel[indx,n], float_format)
			#worksheet5.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
	
	## Plots
	for n in range(N):
		clstr_pcwise_chart = workbook.add_chart({'type': 'scatter'})
		clstr_pcwise_chart.add_series({
			'name':       'Eig'+str(n)+' - diameter',
			'categories': ['Wordlist order', 1, 0, C, 0],
			'values':     ['cluster.diameter', 1, 5*n+3, C, 5*n+3],
			'marker':     {'type': 'short_dash', 'size': 2, 'border': {'color': 'green'}, 'fill': {'color': 'green'}},
		})
		
		worksheet5.insert_chart(2*n+10, 5*n+1, clstr_pcwise_chart)
		
	
	##############
	# WORKSHEET6 #
	##############
	## SORT EACH EIGENVECTOR BY CLUSTER, NO SECONDARY SORT.  WHAT IS THE ORDER, AND WHY DO WE SEE STRUCTURE?
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix  
	#lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.argsort(cluster_labels)		# Note - for Worksheet6, all columns are same; could simplify.
		
	#for n in range(1,N):
		#lex_sortation_rel[:,n] = np.argsort(cluster_labels)
		
	
	worksheet6.set_column(0, (2*N-1)*5-1, 11)   #set column width
	
	## Column headings
	for n in range(N):
		col = 1 + n*5
		worksheet6.merge_range(0, col, 0, col+2, "Eig"+str(n), merge_format)
		worksheet6.write(0, col+3, 'Clusters @Eig'+str(n), bold)
		
	#for n in range(1,N):
		#col = 1+ N*5 + (n-1)*5
		#worksheet6.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		#worksheet6.write(0, col+3, 'Clusters @Eig'+str(n)+'/Eig0', bold)
		
	
	## Data  	#Note that the cluster_labels column is same for every eigenvector, 
				#hence could be generated once and then copied.
				
	for c in range(C):
		row = c + 1
		
		for n in range(N):    # for each eigenvector:
			indx = lex_sortation[c,n]
			col = 1 + n*5
			worksheet6.write(row, col, indx)
			worksheet6.write(row, col+1, wordlist[indx])
			worksheet6.write(row, col+2, eig_input[indx,n], float_format)
			worksheet6.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
		#for n in range(1,N):
			#indx = lex_sortation_rel[c,n]
			#col = 1 + N*5 + (n-1)*5
			#worksheet6.write(row, col, indx)
			#worksheet6.write(row, col+1, wordlist[indx])
			#worksheet6.write(row, col+2, eig_rel[indx,n], float_format)
			#worksheet6.write(row, col+3, cluster_labels[indx], fill_format[ cluster_labels[indx] ])
			
	
	## Plots
	for n in range(N):
		clstr_pcwise_chart = workbook.add_chart({'type': 'scatter'})
		clstr_pcwise_chart.add_series({
			'name':       'Eig'+str(n)+' - no 2nd sort',
			'categories': ['Wordlist order', 1, 0, C, 0],
			'values':     ['cluster.no 2nd sort', 1, 5*n+3, C, 5*n+3],
			'marker':     {'type': 'short_dash', 'size': 2, 'border': {'color': 'purple'}, 'fill': {'color': 'purple'}},
		})
		
		worksheet6.insert_chart(2*n+10, 5*n+1, clstr_pcwise_chart)
		
	
	##############
	# WORKSHEET7 #
	##############
	## RECORD MEAN OF EACH CLUSTER'S COORD VALUES ON EACH EIGENVECTOR
	cluster_ids, card_per_cluster = np.unique(cluster_labels, return_counts=True)
	J = len(cluster_ids)   # cluster_labels has length C. cluster_ids is a short list--the set from which the labels are drawn.
	K = nreq_clusters
	print("K =", K)
	print("J =", J)
	print("cluster_ids: ", cluster_ids)
	print("card_per_cluster: ", card_per_cluster, "\n")
	
	if J==K:
		padded_card_per_cluster = card_per_cluster
	else:
		padded_card_per_cluster = np.zeros(K)
		for id in cluster_ids:
			padded_card_per_cluster[id] = card_per_cluster[list(cluster_ids).index(id)]
		
	
	# BE SURE COUNTS MATCH CATEGORIES IN ORDER   COULD COUNT AT SAME TIME AS SUM TO BE SURE
	coord_sum_per_cluster_per_eigenvector  = np.zeros((K, N))		# WAS  np.zeros((N, N))
	coord_mean_per_cluster_per_eigenvector = np.zeros((K, N))		# WAS  np.zeros((N, N))
	
	for c in range(C):
		for n in range(N):
			
			coord_sum_per_cluster_per_eigenvector[cluster_labels[c],n] = \
				coord_sum_per_cluster_per_eigenvector[cluster_labels[c],n] + eig_input[c,n]
				
	#for k in range(K):     for j in cluster_ids 
		#if k in cluster_ids:
			#kindx = list(cluster_ids).index(k)
			#print("k =", k, ",  kindx =", kindx)
			
			#for n in range(N):
				#coord_mean_per_cluster_per_eigenvector[k,n] = \
					#coord_sum_per_cluster_per_eigenvector[k,n] / card_per_cluster[kindx]
					
	for id in cluster_ids:
		for n in range(N):
			coord_mean_per_cluster_per_eigenvector[id,n] = \
				coord_sum_per_cluster_per_eigenvector[id,n] / padded_card_per_cluster[id]
#			
#				coord_mean_per_cluster_per_eigenvector[cluster_ids[k],n] = \
#					coord_sum_per_cluster_per_eigenvector[cluster_ids[k],n] / card_per_cluster[k]
				

	## XlsxWriter instructions
	worksheet7.set_column(0, N+1, 11)	#set column width
	worksheet7.write(0, 0, 'Coordinate mean per cluster per eigenvector', bold)
	
	worksheet7.write(2, 0, 'A.  Computed from cluster output as recorded on Worksheet3   ' + timestamp_string, bold)
	
	## Column headings
	for n, col in zip(range(N), range(1, N+1)):
		worksheet7.write(3, col, 'Eig'+str(n), bold)
		
	## Data   # EASIER TO INCLUDE ROW FOR EMPTY CLUSTER THAN TO EXCLUDE ITS LABEL AND ADJUST ROWS FOR REMAINING LABELS 
	for m, row in zip(range(K), range(4, 4+K)):              # DO NOT USE J ON THIS LINE. 
		worksheet7.write(row, 0, 'Cluster'+str(m), bold)
		for n, col in zip(range(N), range(1, N+1)):
			worksheet7.write(row, col, coord_mean_per_cluster_per_eigenvector[m,n], float_format)
	
	#   
	
	row = K + 5
	worksheet7.write(row, 0, 'B.  Cluster centers returned by KMeans   ' + timestamp_string, bold)
		
	## Column headings
	for n, col in zip(range(N), range(1, N+1)):
		worksheet7.write(row+1, col, 'Eig'+str(n), bold)
			
	## Data
	for m, row in zip(range(K), range(7+K, 7+2*K)):
		worksheet7.write(row, 0, 'Cluster'+str(m), bold)
		for n, col in zip(range(N), range(1, N+1)):
			worksheet7.write(row, col, cluster_centers[m,n], float_format)
			
	
	## 2d CLUSTER PLOTS
	#if N>2:
	for n in range(1,N):
		heig = n-1        # eig number for horizontal axis
		veig = n          # eig number for vertical axis
		hmin = eig_input[minindex_by_eig[heig], heig]
		hmax = eig_input[maxindex_by_eig[heig], heig]
		vmin = eig_input[minindex_by_eig[veig], veig]
		vmax = eig_input[maxindex_by_eig[veig], veig]
		
		cluster_chart_2d = workbook.add_chart({'type': 'scatter'})
		cluster_chart_2d.set_title({'name': 'Eig' + str(heig) + " x Eig" + str(veig)})
		
		cluster_chart_2d.set_x_axis({'min': min(1.1*hmin, -0.115), 'max': max(1.1*hmax, 0.115)})
		cluster_chart_2d.set_y_axis({'min': min(1.1*vmin, -0.115), 'max': max(1.1*vmax, 0.115)})
		
		#cluster_chart_2d.set_x_axis({'num_format': 0x02})	#These lines had no effect. Using Excel interactively instead.
		#cluster_chart_2d.set_y_axis({'num_format': 0x02})	#Select any number along axis, then use Format/Axis menu.
		#cluster_chart_2d.set_x_axis({'major_unit': 2})
		
		cluster_chart_2d.set_x_axis({'crossing': 0.0})
		cluster_chart_2d.set_y_axis({'crossing': 0.0})
		
		cluster_chart_2d.set_x_axis({'major_gridlines': {'visible': False}})
		cluster_chart_2d.set_y_axis({'major_gridlines': {'visible': False}})
		
		
		#rowbase = 0
		#for k in range(K):    # for each cluster
			#cluster_chart_2d.add_series({
				#'categories': ['cluster.wordlist index', rowbase+1, 3+5*heig,  rowbase + card_per_cluster[k], 3+5*heig],
				#'values':     ['cluster.wordlist index', rowbase+1, 3+5*veig,  rowbase + card_per_cluster[k], 3+5*veig],
				#'marker':     {'type': 'short_dash', 'size': 9, 'border': {'color': LUT[cluster_ids[k]]}, 'fill': {'color': LUT[cluster_ids[k]]}},
				#'x_axis':     {'num_format': 0x02},
			#})
			#rowbase = rowbase + card_per_cluster[k]
			
		rowbase = 0
		for id in cluster_ids:
			cluster_chart_2d.add_series({
				'categories': ['cluster.wordlist index', rowbase+1, 3+5*heig,  rowbase + padded_card_per_cluster[id], 3+5*heig],
				'values':     ['cluster.wordlist index', rowbase+1, 3+5*veig,  rowbase + padded_card_per_cluster[id], 3+5*veig],
				'marker':     {'type': 'short_dash', 'size': 9, 'border': {'color': LUT[id]}, 'fill': {'color': LUT[id]}},
				'x_axis':     {'num_format': 0x02},
			})
			rowbase = rowbase + padded_card_per_cluster[id]	
			
		worksheet7.insert_chart(7 + 2*K + (n-1)*15, 1, cluster_chart_2d)
		#worksheet7.insert_chart(7+N+K + (n-1)*15, 1, cluster_chart_2d)
		
	
	# Plots with heig always Eig0   starting with Eig0 x Eig2, since we already have Eig0 x Eig1
	for n in range(2,N):
		heig = 0        # eig number for horizontal axis
		veig = n          # eig number for vertical axis
		hmin = eig_input[minindex_by_eig[heig], heig]
		hmax = eig_input[maxindex_by_eig[heig], heig]
		vmin = eig_input[minindex_by_eig[veig], veig]
		vmax = eig_input[maxindex_by_eig[veig], veig]
		
		cluster_chart_2d = workbook.add_chart({'type': 'scatter'})
		cluster_chart_2d.set_title({'name': 'Eig' + str(heig) + " x Eig" + str(veig)})
		
		cluster_chart_2d.set_x_axis({'min': min(1.1*hmin, -0.115), 'max': max(1.1*hmax, 0.115)})
		cluster_chart_2d.set_y_axis({'min': min(1.1*vmin, -0.115), 'max': max(1.1*vmax, 0.115)})
		
		#cluster_chart_2d.set_x_axis({'num_format': 0x02})	#These lines had no effect. Using Excel interactively instead.
		#cluster_chart_2d.set_y_axis({'num_format': 0x02})	#Select any number along axis, then use Format/Axis menu.
		#cluster_chart_2d.set_x_axis({'major_unit': 2})
		
		cluster_chart_2d.set_x_axis({'crossing': 0.0})
		cluster_chart_2d.set_y_axis({'crossing': 0.0})
		
		cluster_chart_2d.set_x_axis({'major_gridlines': {'visible': False}})
		cluster_chart_2d.set_y_axis({'major_gridlines': {'visible': False}})
		
		
		#rowbase = 0
		#for k in range(K):    # for each cluster
			#cluster_chart_2d.add_series({
				#'categories': ['cluster.wordlist index', rowbase+1, 3+5*heig,  rowbase + card_per_cluster[k], 3+5*heig],
				#'values':     ['cluster.wordlist index', rowbase+1, 3+5*veig,  rowbase + card_per_cluster[k], 3+5*veig],
				#'marker':     {'type': 'short_dash', 'size': 9, 'border': {'color': LUT[cluster_ids[k]]}, 'fill': {'color': LUT[cluster_ids[k]]}},
				#'x_axis':     {'num_format': 0x02},
			#})
			#rowbase = rowbase + card_per_cluster[k]
			
		rowbase = 0
		for id in cluster_ids:
			cluster_chart_2d.add_series({
				'categories': ['cluster.wordlist index', rowbase+1, 3+5*heig,  rowbase + padded_card_per_cluster[id], 3+5*heig],
				'values':     ['cluster.wordlist index', rowbase+1, 3+5*veig,  rowbase + padded_card_per_cluster[id], 3+5*veig],
				'marker':     {'type': 'short_dash', 'size': 9, 'border': {'color': LUT[id]}, 'fill': {'color': LUT[id]}},
				'x_axis':     {'num_format': 0x02},
			})
			rowbase = rowbase + padded_card_per_cluster[id]
		
		worksheet7.insert_chart(7 + 2*K + (n-1)*15, 9, cluster_chart_2d)
		#worksheet7.insert_chart(7+N+K + (n-1)*15, 9, cluster_chart_2d)
		
	
	# Plots with heig always Eig1   starting with Eig1 x Eig3, since we already have Eig1 x Eig2
	for n in range(3,N):
		heig = 1        # eig number for horizontal axis
		veig = n          # eig number for vertical axis
		hmin = eig_input[minindex_by_eig[heig], heig]
		hmax = eig_input[maxindex_by_eig[heig], heig]
		vmin = eig_input[minindex_by_eig[veig], veig]
		vmax = eig_input[maxindex_by_eig[veig], veig]
		
		cluster_chart_2d = workbook.add_chart({'type': 'scatter'})
		cluster_chart_2d.set_title({'name': 'Eig' + str(heig) + " x Eig" + str(veig)})
		
		cluster_chart_2d.set_x_axis({'min': min(1.1*hmin, -0.115), 'max': max(1.1*hmax, 0.115)})
		cluster_chart_2d.set_y_axis({'min': min(1.1*vmin, -0.115), 'max': max(1.1*vmax, 0.115)})
		
		#cluster_chart_2d.set_x_axis({'num_format': 0x02})	#These lines had no effect. Using Excel interactively instead.
		#cluster_chart_2d.set_y_axis({'num_format': 0x02})	#Select any number along axis, then use Format/Axis menu.
		#cluster_chart_2d.set_x_axis({'major_unit': 2})
		
		cluster_chart_2d.set_x_axis({'crossing': 0.0})
		cluster_chart_2d.set_y_axis({'crossing': 0.0})
		
		cluster_chart_2d.set_x_axis({'major_gridlines': {'visible': False}})
		cluster_chart_2d.set_y_axis({'major_gridlines': {'visible': False}})
		
		
		#rowbase = 0
		#for k in range(K):    # for each cluster
			#cluster_chart_2d.add_series({
				#'categories': ['cluster.wordlist index', rowbase+1, 3+5*heig,  rowbase + card_per_cluster[k], 3+5*heig],
				#'values':     ['cluster.wordlist index', rowbase+1, 3+5*veig,  rowbase + card_per_cluster[k], 3+5*veig],
				#'marker':     {'type': 'short_dash', 'size': 9, 'border': {'color': LUT[cluster_ids[k]]}, 'fill': {'color': LUT[cluster_ids[k]]}},
				#'x_axis':     {'num_format': 0x02},
			#})
			#rowbase = rowbase + card_per_cluster[k]
			
		rowbase = 0
		for id in cluster_ids:
			cluster_chart_2d.add_series({
				'categories': ['cluster.wordlist index', rowbase+1, 3+5*heig,  rowbase + padded_card_per_cluster[id], 3+5*heig],
				'values':     ['cluster.wordlist index', rowbase+1, 3+5*veig,  rowbase + padded_card_per_cluster[id], 3+5*veig],
				'marker':     {'type': 'short_dash', 'size': 9, 'border': {'color': LUT[id]}, 'fill': {'color': LUT[id]}},
				'x_axis':     {'num_format': 0x02},
			})
			rowbase = rowbase + padded_card_per_cluster[id]
		
		worksheet7.insert_chart(7 + 2*K + (n-1)*15, 17, cluster_chart_2d)
		#worksheet7.insert_chart(7+N+K + (n-1)*15, 17, cluster_chart_2d)
		

	
	workbook.close()
	



#def decomposition_ica(data_array, wordlist, timestamp):
	#outfilename = "decomp_ica_info.sk_dl." + timestamp.strftime("%Y_%m_%d.%H_%M") + ".csv"
	#outfile = open(outfilename, mode='w')
	
#def sparse_repr(data_array, wordlist, timestamp):
	##sparse_repr_sk_dl(data_array, wordlist, timestamp)
	#decomposition_ica(data_array, wordlist, timestamp)
	
	
def check_eigen_match(laplacian, eigenvalues, eigenvectors):
	#eig1 = eigenvectors[:, 1]
	lapl_times_eigenvectors = laplacian @ eigenvectors
	print("eigenvalues:", eigenvalues)
	#print("eig1 =", eig1[0:5])
	print("lambda0_times_eigenvector0 =", eigenvalues[0] * eigenvectors[0:6, 0])
	print("lambda1_times_eigenvector1 =", eigenvalues[1] * eigenvectors[0:6, 1])
	print("lambda2_times_eigenvector2 =", eigenvalues[2] * eigenvectors[0:6, 2])
	print("laplacian_times_eigenvectors:")
	print(lapl_times_eigenvectors[0:6, 0:3])
	print("\n")




# NOTE THAT I'M NOW SETTING n_eigenvectors    AUDREY   2017_04_06
# These values are default settings. The real values come from the startup settings.
def run(unigram_counter=None, bigram_counter=None, trigram_counter=None,
		max_word_types=1000, n_neighbors=9, n_eigenvectors=11,
		min_context_count=3):

    output_dir = "DevOutput/"     # THIS IS FOR spreadsheet PARTICULARLY spectral clustering DEVELOPMENT. o.w. my storage area
    if not os.path.exists(output_dir):
    	os.mkdir(output_dir)
    	
    timestamp = datetime.datetime.now()
    timestamp_string = timestamp.strftime(".%Y_%m_%d.%H_%M")
    
    word_freq_pairs = double_sorted(unigram_counter.items(),
                                    key=lambda x: x[1], reverse=True)
    	
	#debugging 1st example found where the wordlist entry is displaced from 0th spot in list of its nearest neighbors
    #print("\nword_freq_pairs from line 476 of manifold.py:\n")
    #print(word_freq_pairs[: 1415])  # ('difficulty', 77), ('editor', 77), ('election', 77), ('follows', 77), ('governor', 77), ('jr', 77), ('judge', 77)]
    #print()

    if len(word_freq_pairs) > max_word_types:
        wordlist = [word for word, _ in word_freq_pairs[: max_word_types]]
    else:
        wordlist = [word for word, _ in word_freq_pairs]
        
    # INSERTED ON FEB. 21, 2018 TO SEE WHAT HAPPENS WHEN OMIT WORDS WHICH WERE ASSIGNED  MINUSCULE VALUES ON EIG_0 
    #fd = open('datasets/Wordlist6856.2018_02_20.txt')    # READ IN A DIFFERENT WORDLIST
    #contents = fd.read()
    #wordlist = contents.splitlines()
    ##for j in range(5):
    ##	print(wordlist[j])
    # END OF INSERTION  FEB. 21, 2018
    
    # ATTENTION: THIS SHOULD BE DONE PROGRAMMATICALLY. AND IF NOT IT SHOULD BE EXTERNAL.  November 20, 2018
    wordlist = remove_outliers(wordlist)


    # SET UP THE GRAPH
    # computing the context array
    # also words_to_contexts and contexts_to_words dicts
    context_array, words_to_contexts, contexts_to_words, ctxt_supported_wordlist = get_context_array(
        wordlist, bigram_counter, trigram_counter, min_context_count, verbose=False)
        
    shared_context_matrix, shared_ctxt_supported_wordlist = get_shared_context_matrix(context_array, ctxt_supported_wordlist, contexts_to_words, verbose=False  ) 
    # These parameters are needed only for verbose screen output: contexts_to_words
    del context_array
    
    #####  NOTE NOTE NOTE wordlist is modified here #####
    n_words = len(shared_ctxt_supported_wordlist)	# Safer than using shared_ctxt_supported_wordlist below
    wordlist = shared_ctxt_supported_wordlist		# and leaving wordlist unchanged and available (for accidental misuse!)
    
    
    # OBJECTIVE:  minimize NCut
    # NP-hard, so first do relaxed optimization 
    # Consider laplacian L and two normalized versions: L_rw = D^(-1) * L  and  L_sym = D^(-1/2) * L * D^(-1/2)
    # Solution for relaxed objective is (equivalently)
    #    "generalized eigen-decomp" of L   (i.e, solutions for  Lu = lambda Du)
    #    eigen-decomp of L_rw
    #    D^(-1/2) * eigen-decomp of L_sym
    
    
    # COMPUTE RELAXED SOLUTION ~ "soft clustering"    [L_sym version]
    laplacian, sqrt_diam = csgraph_laplacian(shared_context_matrix, normed=True, return_diag=True)   # returns L_sym
    #laplacian, sqrt_diam = Chicago_get_laplacian(shared_context_matrix)
    diameter = np.square(sqrt_diam)		# ndarray
        
    eigenvalues, eigenvectors = linalg.eigsh(laplacian, k=n_eigenvectors, which='SM')  #N.B. eigs complex; eigsh not
    eigenvectors = (_deterministic_vector_sign_flip(eigenvectors.T)).T                 #standardization convention; no other effect
    
    # Collect information to pass to spreadsheets
    common_data_for_spreadsheets = [wordlist, diameter, output_dir, timestamp_string]
    
    # DISCRETE ("hard") CLUSTERING
    # Passage from soft to hard stage may proceed from eigen-decomp of either L_rw (Shi-Malik) or L_sym (Ng+others). Results will not be identical.
    # Alternative algorithms  ref. Ulrike von Luxburg. A Tutorial on Spectral Clustering.
    # These alternatives are organized as separate functions for clarity and ease of separate modification.
    
    spectral_clustering_menu(eigenvectors, eigenvalues, common_data_for_spreadsheets, random_seed=1)
    # spectral_clustering_menu( ) replaces next several lines
    #
    #wordcoords_sym, clusterlabels_sym, clustercenters_sym = spectral_clustering_sym(eigenvectors, common_data_for_spreadsheets, assign_labels='kmeans', random_seed=1)
    ##wordcoords_sym, clusterlabels_sym, clustercenters_sym = spectral_clustering_sym(eigenvectors, assign_labels='discretize', random_seed=1)
    #generate_eigenvector_spreadsheet('sym', wordcoords_sym, clusterlabels_sym, clustercenters_sym, common_data_for_spreadsheets)
    ##eigenvector_clusterwise_plots('sym', wordcoords_sym, clusterlabels_sym, diameter, output_dir, timestamp_string)
    
    #wordcoords_rw, clusterlabels_rw, clustercenters_rw = spectral_clustering_rw(eigenvectors, sqrt_diam, common_data_for_spreadsheets, assign_labels='kmeans', random_seed=1)
    #generate_eigenvector_spreadsheet('rw', wordcoords_rw, clusterlabels_rw, clustercenters_rw, common_data_for_spreadsheets)
    ##eigenvector_clusterwise_plots('rw', wordcoords_rw, clusterlabels_rw, diameter, output_dir, timestamp_string)
    #
    
    raise SystemExit    # October 26, 2018
    
    # Computing distances between words
    
    #word_distances = compute_words_distance(coordinates)
    word_distances = compute_words_distance(eigenvectors)
    #print("\nword_distances:")   # audrey  2016_12_05
    #print(word_distances)  # This is a matrix  word_list x word_list  with entries the distances.   audrey  2016_12_05
    #del coordinates
    #del eigenvalues    # AUDREY  2017_04_22  Retain eigenvalues to use later in print_wordvectors_info()

    # computing nearest neighbors now
    nearest_neighbors = compute_closest_neighbors(word_distances, n_neighbors)     # correction for misplaced word entry is handled here
    # print("\nnearest_neighbors:")   # audrey  2016_12_05
    # print(nearest_neighbors)   # These are indices, into wordlist    audrey  2016_12_05
                                 # We need indices to retrieve both words and distances
                               
 
    words_to_neighbors = dict()
    words_to_neighbor_distances = dict()
     
    for i in range(len(wordlist)):
        line = nearest_neighbors[i]
        word_idx, neighbors_idx = line[0], line[1:]
        word = wordlist[word_idx]
        neighbors = [wordlist[idx] for idx in neighbors_idx]
        neighbor_distances = [word_distances[i, idx] for idx in neighbors_idx]   # AUDREY  2017_03_28  IS THIS RIGHT? word_distances[word_idx, idx] for idx in neighbors_idx
        #neighbor_distances = [word_distances[word_idx, idx] for idx in neighbors_idx]   # AUDREY  2017_03_28  IS THIS RIGHT? word_distances[word_idx, idx] for idx in neighbors_idx
        words_to_neighbors[word] = neighbors
        words_to_neighbor_distances[word] = neighbor_distances
        
    
    return words_to_neighbors, words_to_contexts, contexts_to_words
    
    
