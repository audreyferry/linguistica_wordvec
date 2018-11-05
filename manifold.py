# -*- encoding: utf8 -*-
 
from collections import defaultdict

#from sympy import *   # may just Matrix??
#xfrom sympy.matrices import Matrix


from scipy import (sparse, spatial)
from scipy.sparse import linalg
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
	

def compute_diameter_array(n_words, shared_context_matrix):
#def normalize(n_words, shared_context_matrix):		#changed name to avoid confusion with sklearn's normalize function 2017_08_02
    arr = np.ones(n_words, dtype=np.int64)
    for word_no in range(n_words):
        arr[word_no] = np.sum(shared_context_matrix[word_no,:]) - \
                       shared_context_matrix[word_no, word_no]
    return arr   # NOTE. 


def compute_incidence_graph(n_words, diameter, shared_context_matrix):
	#incidence_graph = np.asarray(shared_context_matrix, dtype=np.int64)
	incidence_graph = np.asarray(-1 * shared_context_matrix, dtype=np.int64)    # audrey  MULT. BY -1   April 28, 2018
	for word_no in range(n_words):
		incidence_graph[word_no, word_no] = diameter[word_no]
	return incidence_graph


def compute_laplacian(diameter, incidence_graph):
    d = np.sqrt(np.outer(diameter, diameter))
    # we want to NOT have div-by-zero errors,
    # but if d[i,j] = 0 then incidence_graph[i,j] = 0 too.
    d[d == 0] = 1

    # broadcasts the multiplication, so A[i,j] = B[i,j] * C[i, j]
    laplacian = (1 / d) * incidence_graph
    print("\nHere is the previously incorrect laplacian:")   #TEMPORARY  April 27, 2018
    print(laplacian[:11, :6])
    print("Are there any negative entries?")
    
    return laplacian


#def sym_compute_eigenvalues(laplacian, n_eigenvectors):
#	sym_laplacian = Matrix(laplacian)
#	return(list(sym_laplacian.eigenvals().keys()))
	

def compute_eigenvectors(laplacian, n_eigenvectors):
    # csr_matrix in scipy means compressed matrix
    laplacian_sparse = sparse.csr_matrix(laplacian)

    # linalg is the linear algebra module in scipy.sparse
    # eigs takes a matrix and
    # returns (array of eigenvalues, array of eigenvectors)
    
    #return linalg.eigs(laplacian_sparse)          # AUDREY   k=6 by default
    #return linalg.eigs(laplacian_sparse, k=2)
    #return linalg.eigs(laplacian_sparse, k=11)    # AUDREY   2017_03_27
    
    #return linalg.eigs(laplacian_sparse, k=n_eigenvectors)
    return linalg.eigs(laplacian_sparse, k=n_eigenvectors, which='SM')    # AUDREY  Added which='SM"  April 28, 2018
    #return linalg.eigsh(laplacian_sparse, k=n_eigenvectors)
    
    

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


def make_heatmap(wordlist, words_to_neighbors, words_to_neighbor_distances, timestamp, max_heatmap_words=400):    #audrey  2016_12_12
    x_entries = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Span = 25    # 50
    Overlap = 5
    
    heatmap_dirname = 'HEATMAPS.' + timestamp.strftime("%Y_%m_%d.%H_%M")
    # audrey  2017_01_29   alt., could pass in output dir from lexicon.py,
    # or set up member of Lexicon to hold heatmaps and do output there
    
    os.mkdir(heatmap_dirname)
    current_dir = os.getcwd()
    os.chdir(heatmap_dirname)
    
    print("Heatmaps...")	# if wordvector stuff were a separate module, this would be handled in class Lexicon, on lexicon.py
    i = 0
    while i < max_heatmap_words:
    	if (i + Span + Overlap > max_heatmap_words):
    		Overlap = 0
    		
    	y_entries = wordlist[i:i+Span+Overlap]
    	#print("\nHeatmap  words:")
    	#print(y_entries)   #  audrey  2016_12_12
    	
    	z_entries = list()
    	#print("\nHeatmap  word and (adjusted) neighbor_distances, ROUNDED, then check ORIGINAL :")
    	for word in y_entries:
    		rounded_distances_list = list()
    		for val in words_to_neighbor_distances[word]:
    			rounded_distances_list.append(round(val, 4))   # more legible via hoverinfo
    		z_entries.append(rounded_distances_list)
    		#print("\n", word, rounded_distances_list)  #audrey  2016_12_13
    		#print("\n", word, words_to_neighbor_distances[word])  #audrey  2016_12_13
    	
    	text_entries = list()
    	#print("\nHeatmap text entries:")
    	for word in y_entries:
    		text_entries.append(words_to_neighbors[word])
    		#print("\n", word, words_to_neighbors[word])
    	
    	annot_entries = go.Annotations()
    	for n, row in enumerate(text_entries):
    		for m, val in enumerate(row):
    			annot_entries.append(go.Annotation(text=text_entries[n][m], x=x_entries[m], y=y_entries[n],
    														xref='x1', yref='y1', showarrow=False,
    														font=dict(size=11, color='white' if z_entries[n][m] < 0.074 else 'black')))
    														
    	trace = go.Heatmap(x=x_entries, y=y_entries, z=z_entries, text=text_entries, hoverinfo='y+text+z', colorscale='Hot', zmin=0, zmax=0.15)
    	upperbound = i+Span
    	title_string = 'Distances: original - Words: ' + str(i) + ' to ' + str(i+Span)
    	fig = go.Figure(data=go.Data([trace]))
    	fig['layout'].update(
    		title=title_string,
    		annotations=annot_entries)
    	
    	fname = 'heatmap_uniform_colorscale.annots_and_hover.formatted.' + str(i) + 'to' + str(i+Span) + timestamp.strftime("%Y_%m_%d.%H_%M" + ".html")
    	plotly.offline.plot(fig, filename=fname, auto_open=False)   # Trying to get graph not to display, but to record to html file
    	
    	i = i+Span
    
    # RETURN FROM HEATMAP DIRECTORY TO DIRECTORY FROM WHICH THIS FUNCTION WAS CALLED
    os.chdir(current_dir)
	
	

def plot_multiclasses_yz(classes_vecs, color_per_class, eigshape, label, timestamp):  # ADD LEGEND   # TRY Nouns, then clean those
    thefigure = plt.figure()
    title_string = eigshape + "     " + label + "     " + timestamp.strftime("%Y_%m_%d.%H_%M")
    plt.title(title_string)
    ax = plt.gca()
    
    class_labels = label.split('_')
    
    for i in range(len(classes_vecs)):
    	vecs = classes_vecs[i]
    	origin_yz_array = np.zeros((len(vecs), 2))
    	vecs_yz_array = vecs[:,1:3]     #  MAY NEED TO BE normalized  ALSO FOR xy AND xz! 
    	
    	arrows_array = np.concatenate((origin_yz_array, vecs_yz_array), axis=1)
    	X0, Y0, U, V = zip(*arrows_array)
    	ax.quiver(X0, Y0, U, V, angles='xy', scale_units='xy', scale=1, color=color_per_class[i], width=.004, label=class_labels[i])
    	
    # EXTENT OF AXES 
    ax.set_xlim([-.1, .1])            #([-.03, .03])	#([-.05, .05])   # or ([-.05, 1.05])    standard for both axes: ([-1.05, 1.05])
    ax.set_ylim([-.1, .1])            #([-.03, .03])	#([-.05, .05])   # or ([-1.05, .05])    
    
    plt.legend(loc='best')		# 'upper left'
    
    plt.draw()
    # plot.show()
    
    filename_string = 'Multivecs_yz.' + eigshape + '.' + label + timestamp.strftime(".%Y_%m_%d.%H_%M") + '.png'
    plt.savefig(filename_string)
    plt.close(thefigure)
    	


def plot_multiclasses_xz(classes_vecs, color_per_class, eigshape, label, timestamp):  # ADD LEGEND   # TRY Nouns, then clean those
    thefigure = plt.figure()
    title_string = eigshape + "     " + label + "     " + timestamp.strftime("%Y_%m_%d.%H_%M")
    plt.title(title_string)
    ax = plt.gca()
    
    class_labels = label.split('_')
    
    for i in range(len(classes_vecs)):
    	vecs = classes_vecs[i]
    	origin_xz_array = np.zeros((len(vecs), 2))
    	vecs_xz_array = vecs[:,(0,2)]
    	
    	arrows_array = np.concatenate((origin_xz_array, vecs_xz_array), axis=1)
    	X0, Y0, U, V = zip(*arrows_array)
    	ax.quiver(X0, Y0, U, V, angles='xy', scale_units='xy', scale=1, color=color_per_class[i], width=.004, label=class_labels[i])
    	
    # EXTENT OF AXES 
    ax.set_xlim([-.1, .1])              #([-.03, .03])	#([-.05, .05])   # or ([-.05, 1.05])    standard for both axes: ([-1.05, 1.05])
    ax.set_ylim([-.1, .1])             #([-.03, .03])	#([-.05, .05])   # or ([-1.05, .05])    for non-unit vecs, try [-.05, .05]  for both axes
    
    plt.legend(loc='best')	# 'upper left'
    
    plt.draw()
    # plot.show()
    
    filename_string = 'Multivecs_xz.' + eigshape + '.' + label + timestamp.strftime(".%Y_%m_%d.%H_%M") + '.png'
    plt.savefig(filename_string)
    plt.close(thefigure)
    	


def plot_multiclasses_xy(classes_vecs, color_per_class, eigshape, label, timestamp):  # ADD LEGEND   # TRY Nouns, then clean those
    thefigure = plt.figure()
    title_string = eigshape + "     " + label + "     " + timestamp.strftime("%Y_%m_%d.%H_%M")
    plt.title(title_string)
    ax = plt.gca()
    
    class_labels = label.split('_')
    
    for i in range(len(classes_vecs)):
    	vecs = classes_vecs[i]
    	origin_xy_array = np.zeros((len(vecs), 2))
    	vecs_xy_array = vecs[:,:2]
    	
    	arrows_array = np.concatenate((origin_xy_array, vecs_xy_array), axis=1)
    	X0, Y0, U, V = zip(*arrows_array)
    	ax.quiver(X0, Y0, U, V, angles='xy', scale_units='xy', scale=1, color=color_per_class[i], width=.004, label=class_labels[i])
    	
    # EXTENT OF AXES 
    if eigshape[-1]=='2':  # REAL CONDITION SHOULD BE LENGTH OF VECTORS; FOR NOW UNIT VECTORS <=> n_eigenvectors==2
    	ax.set_xlim([-1.05, 1.05])   # or ([-.05, 1.05])    standard for both axes: ([-1.05, 1.05])
    	ax.set_ylim([-1.05, 1.05])   # or ([-1.05, .05])    for non-unit vecs, try [-.05, .05]  for both axes
    else:
    	ax.set_xlim([-.1, .1])        #([-.03, .03])	# ([-.05, .05])		# [-.08, .08]
    	ax.set_ylim([-.1, .1])        #([-.03, .03])	# ([-.05, .05])
    
    plt.legend(loc='upper left')		# 'best' came out upper right. Upper left looks better.
    
    plt.draw()
    # plot.show()
    
    filename_string = 'Multivecs_xy.' + eigshape + '.' + label + timestamp.strftime(".%Y_%m_%d.%H_%M") + '.png'
    plt.savefig(filename_string)
    plt.close(thefigure)
    	


def plot_LR_pair(leftvecs, rightvecs, eigshape, label, timestamp, leftwords,
				 show_left=True, show_right=True, show_diff_LR=True, show_diff_origin=False):
    
    #plt.ioff()    # if necessary
    
    thefigure = plt.figure()
    title_string = eigshape + "     " + label + "     " + timestamp.strftime("%Y_%m_%d.%H_%M")
    plt.title(title_string)
    ax = plt.gca()
    
    if show_diff_LR==True or show_diff_origin==True:
    	contrast_width = .002
    else:
    	contrast_width = .005
    	
    some_colors   = ['blue', 'green', 'crimson', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'sienna', 'darkcyan', 'darkgoldenrod', 'deeppink', 'darkslategray', 'blueviolet', 'seagreen', 'orange']
    #some_colors   = ['blue', 'green', 'crimson', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'sienna', 'darkcyan', 'darkgoldenrod', 'deeppink', 'darkslategray', 'blueviolet', 'seagreen', 'orange', 'blue', 'blue', 'red', 'red']  #N.B. blue, red at end NOT related to blue, red used in basic list
    #some_colors = ['cyan', 'cyan', 'black', 'black', 'crimson', 'crimson']   # This is for the 3 strong verbs
    # May customize the color list to display a particular behavior. 
    # For example, to highlight both state_stated and state_statement superimposed on some previous plot,
    # may add, say, 'olive' TWICE to the color list, at the indices corresponding to the two 'state_ ' pairs. 
    # Note, however, that if such an additional item is marked with a color that is part of the original list, 
    # it may be simply because that color stands out strongly, and not necessarily meant to indicate a relation
    # to a previous item of that color. 
    
    origin_2d_array = np.zeros((len(leftvecs), 2))     # inner parenthesis needed because 'shape' is a tuple
    left_2d_array   = leftvecs[:,:2]
    right_2d_array  = rightvecs[:,:2]
    diff_2d_array   = right_2d_array - left_2d_array
    
    
    # CREATE ARROWS
    if show_left==True:
    	arrows_array_lwords = np.concatenate((origin_2d_array, left_2d_array), axis=1)
    	X0, Y0, Ul, Vl = zip(*arrows_array_lwords)
    	ax.quiver(X0, Y0, Ul, Vl, angles='xy', scale_units='xy', scale=1, color=some_colors, width=contrast_width)
    	
    if show_right==True:
    	arrows_array_rwords = np.concatenate((origin_2d_array, right_2d_array), axis=1)
    	X0, Y0, Ur, Vr = zip(*arrows_array_rwords)
    	ax.quiver(X0, Y0, Ur, Vr, angles='xy', scale_units='xy', scale=1, color=some_colors, width=contrast_width)   #.001  #.005  #.003

    if show_diff_LR==True:
    	arrows_array_diff = np.concatenate((left_2d_array, diff_2d_array), axis=1)
    	Xl, Yl, Ud, Vd = zip(*arrows_array_diff)
    	ax.quiver(Xl, Yl, Ud, Vd, angles='xy', scale_units='xy', scale=1, color=some_colors)
    
    if show_diff_origin==True:
    	arrows_array_diff0 = np.concatenate((origin_2d_array, diff_2d_array), axis=1)
    	X0, Y0, Ud, Vd = zip(*arrows_array_diff0)
    	ax.quiver(X0, Y0, Ud, Vd, angles='xy', scale_units='xy', scale=1, color=some_colors)
    
    
    # EXTENT OF AXES     U horizontal, V vertical
    U_limits = (0,)    
    V_limits = (0,)
    
    if show_left==True:
    	U_limits = U_limits + Ul
    	V_limits = V_limits + Vl
    	
    if show_right==True:
    	U_limits = U_limits + Ur
    	V_limits = V_limits + Vr
    	
    if show_diff_LR==True:
    	if show_left==False:
    		Ul = tuple(left_2d_array[:,0])
    		Vl = tuple(left_2d_array[:,1])
    		U_limits = U_limits + Ul
    		V_limits = V_limits + Vl
    	if show_right==False:
    		Ur = tuple(right_2d_array[:,0])
    		Vr = tuple(right_2d_array[:,1])
    		U_limits = U_limits + Ur
    		V_limits = V_limits + Vr
    		
    if show_diff_origin==True:
    	U_limits = U_limits + Ud
    	V_limits = V_limits + Vd
    
    u_low  = min(U_limits) - .05            # .05 for unitvectors; .005 for prenorm vectors
    u_high = max(U_limits) + .05
    v_low  = min(V_limits) - .05
    v_high = max(V_limits) + .05
    #same_low = min(u_low, v_low)
    #same_high = max(u_high, v_high)

    ax.set_xlim([u_low, u_high])
    ax.set_ylim([v_low, v_high])
    #ax.set_xlim([same_low, same_high])		# to get 1:1 aspect ratio if desired
    #ax.set_ylim([same_low, same_high])
    
    legend_entries = list()
    range_limit = min(len(leftwords), len(some_colors))
    for i in range(range_limit):
    	i_patch = mpatches.Patch(color=some_colors[i], label=leftwords[i])   
    	legend_entries.append(i_patch)
    plt.legend(handles=legend_entries, bbox_to_anchor=(1.04, 0.5), loc='center left')  
    #bbox_to_anchor coords have 0 as bottom of plot, 1 as top; similarly for left-right
    
    plt.draw()
    # plt.show()  # WILL COMPUTATION PROCEED?  
    
    filename_string = 'Diffvecs.' + eigshape + '.' + label + timestamp.strftime(".%Y_%m_%d.%H_%M") + '.png'
    plt.savefig(filename_string, bbox_inches='tight')   # kwarg has good effect of not cutting off some of the legend.
    plt.close(thefigure)


def words_to_wordvectors(outfile, manifold_wordlist, eigenvectors, the_words):
	n_examples = len(the_words)
	
	inds = [manifold_wordlist.index(word) for word in the_words]
	vecs = eigenvectors[inds, :]
	#vecs = vecs.real    # WILL THIS GET RID OF THE 'ComplexWarning'??  YES. Prefer to fix it upstream however [upon return from compute_eigenvectors()], since there is also an earlier ComplexWarning.
	lengths = [np.linalg.norm(v) for v in vecs]   #LIST
	
	#unitvecs = np.ones(vecs.shape)
	#for i in range(n_examples):
	#	unitvecs[i,:] = vecs[i,:] / lengths[i]				# Raises ComplexWarnng if vecs have imaginary part (even if 0.j, as in our case)
	unitvecs = skl_normalize(vecs, norm='l2')                 # August 8, 2017 -- got scikit-learn and normalize to work; "sl" for scikit-learn
	checklengths  = [np.linalg.norm(v) for v in unitvecs]
	
	print("words:   ", the_words, file=outfile)
	print("inds:    ", inds, file=outfile)
	print("\nvecs:\n", vecs, file=outfile)
	print("\nlengths: ", lengths, file=outfile)
	print("\nunitvecs:\n", unitvecs, file=outfile)
	print("\nchecklengths: ", checklengths, file=outfile)
	#print("\nvecs (check that vecs unchanged after calculation of unitvecs:\n", leftvecs, file=outfile)  # TO BE SURE THESE ARE UNCHANGED
	print(file=outfile)
	
	
	# FIND BETTER COMPUTATION FOR UNITVECS (ABOVE)
	#unitvecs = np.divide(vecs, np.linalg.norm(vecsOI))   # doesn't work
	#unitvecs = np.multiply(vecs * 1/lengths)   # doesn't work
	
	# NOT YET ABLE TO IMPORT scikit-learn    # August 8, 2017 -- got scikit-learn and normalize to work
	#vecs_normalized = skl_normalize(vecs, norm='l2')
	#print("\nNOW TRY SKLEARN VERSION", file=outfile)
	#print("vecs_normalized:\n", vecs_normalized, file=outfile)
	#print("vecs (check that unchanged after calc of unitvecs:\n", vecs, file=outfile)  # TO BE SURE THESE ARE UNCHANGED
	
	#return unitvecs, vecs.real  # Fixed upstream instead (see above).
	return unitvecs, vecs
	
	

def wordvector_multiclasses_info(eigenvalues, eigenvectors, manifold_wordlist, classes_words, eigshape, label, timestamp):
	outfilename = "wordvector_multiclasses_info." + eigshape + '.' + label + '.' + timestamp.strftime("%Y_%m_%d.%H_%M") + ".cosine"
	outfile = open(outfilename, mode='w')
	
	print("eigenvalues:", file=outfile)
	print(eigenvalues, file=outfile)
	print(file=outfile)
	print("Shape of eigenvectors matrix: ", eigenvectors.shape, file=outfile)
	
	print("\nCLASSES  " + label, file=outfile)
	
	labels = label.split('_')
	#print("\nlabels: ", labels)
	classes_unitvecs = []
	classes_vecs = []
	
	#for (cls, label) in ??
	for i in range(len(classes_words)):
		print("\n\n[" + labels[i] + "]  WORDS TO WORDVECTORS\n", file = outfile)
		unitvecs, vecs = words_to_wordvectors(outfile, manifold_wordlist, eigenvectors, classes_words[i])
		classes_unitvecs.append(unitvecs)
		classes_vecs.append(vecs)
	
	return classes_unitvecs, classes_vecs
    


# NOTE - Analogies deal with pairs of related words. In this investigation, the relation will generally be between a base form and a form with a different grammatical function.
# By convention, the left word will be the base form--most often, the left having a null suffix and the right a standard suffix.
# For instance:
#      <Verb forms>  left - present (usually infinitive),   right - past   [Exception: checked both  (do, did) and  (does, did)]
#      <adj_adv>     left - adjective,  right - adverb
def wordvector_LR_info(eigenvalues, eigenvectors, manifold_wordlist, lwords, rwords, eigshape, label, timestamp):    # 'l' and 'r' for left and right.
    outfilename = "wordvector_pair_info." + eigshape + '.' + label + '.' + timestamp.strftime("%Y_%m_%d.%H_%M") + ".cosine"
    outfile = open(outfilename, mode='w')
    
    print("eigenvalues:", file=outfile)
    print(eigenvalues, file=outfile)
    print(file=outfile)
    print("Shape of eigenvectors matrix: ", eigenvectors.shape, file=outfile)
    
    print("\nANALOGY " + label, file=outfile)
    n_examples = len(lwords)
    print("n_examples = " + str(n_examples) + "\n", file = outfile)
    

    print("\nLEFT WORDS TO WORDVECTORS", file = outfile)
    leftunitvecs, leftvecs = words_to_wordvectors(outfile, manifold_wordlist, eigenvectors, lwords)
    
    print("\nRIGHT WORDS TO WORDVECTORS", file = outfile)
    rightunitvecs, rightvecs = words_to_wordvectors(outfile, manifold_wordlist, eigenvectors, rwords)
    
    
    print("\nINVESTIGATE BEHAVIOR OF lword_rword CORRESPONDENCE ACROSS THE SET OF EXAMPLES", file=outfile)
    print("For reference, line-number/word-pair identification shown here for matrices below \n(list order currently not significant):\n", file=outfile)
    for i in range(n_examples):
    	print("(" + str(i) + ", " + lwords[i] + "_" + rwords[i] + ")", file=outfile)
    
    
    # VECTOR SUBTRACTION
    diffvecs = rightunitvecs - leftunitvecs    # for each row, do elementwise subtraction
    print("\n\ndiffvecs  (that is, rightunitves - leftunitvecs):\n", diffvecs, file = outfile)
    print(file=outfile)
    print(file=outfile)
    
    cosdistarray = spatial.distance.squareform(spatial.distance.pdist(diffvecs, 'cosine'))
    print(file=outfile)
    print("cosdistarray (squareform)  < for diffvecs >\n", cosdistarray, file=outfile)
    print(file=outfile)
    
    print("\n\nFor convenience, pairwise cosine distance between diffvecs shown also as table (Last column is degrees)\n", file=outfile)
    for i in range(n_examples):
    	for j in range(i+1, n_examples):
    		#print("cosine distance " + str(i) + "_" + str(j) + ": \t", cosdistarray[i,j], file=outfile)
    		#print("cosine distance " + str(i) + "_" + str(j) + " \t" + lwords[i] + "," + lwords[j] + ": \t" + str(cosdistarray[i,j]) + "   \t" + str(np.rad2deg(np.arccos(1-cosdistarray[i,j]))), file=outfile)
    		print("cosine distance " + str(i) + "_" + str(j) + " \t" + str(cosdistarray[i,j]) + "   \t" + str(np.rad2deg(np.arccos(1-cosdistarray[i,j]))), file=outfile)
    print(file=outfile)    
    #print("cosine distance 01: ", cosdistarray[0,1], file=outfile)   # spatial.distance.cosine(diffvecs[0,:], diffvecs[1,:])
    
    # NEW ON May 12, 2017
    print("\ncosine distance between left wordvector and diffvec (last column is degrees)\n", file=outfile)
    for i in range(n_examples):
    	cosine_distance = spatial.distance.cosine(-leftunitvecs[i,:], diffvecs[i,:])
    	degrees = np.rad2deg(np.arccos(1-cosine_distance))
    	print(str(i) + "\t" + str(cosine_distance) + "\t" + str(degrees), file=outfile)
    print(file=outfile)
    
    	
    # For PLOT
    # This shows info for quiver(). It's shown here strictly for display, and for diffvecs only. 
    # See plot_LR_pair() for the calculations that are actually used.
    origin_array = np.zeros((n_examples, 2))     # shape is a 2-tuple
    print("origin_array:\n", origin_array, file=outfile)
    
    arrows_array = np.concatenate((origin_array, diffvecs[:,:2]), axis=1)
    print("\narrows_array for diffvecs:\n", arrows_array, file=outfile)
    
    print("\n\nSEE ACCOMPANYING PLOT:", file=outfile)
    plot_filename_string = 'Diffvecs.' + eigshape + '.' + label + timestamp.strftime(".%Y_%m_%d.%H_%M") + '.png'
    print(plot_filename_string, file=outfile)
    
    outfile.close()
    return leftunitvecs, rightunitvecs, leftvecs, rightvecs


def i_plot_set_of_categories(category_to_ivecs, filename, category_to_color):
	thefigure = plt.figure()
	plt.title(filename)
	ax = plt.gca()
	
	for cat in category_to_ivecs:
		#print("in plot, cat =", cat)
		vector_2d_array = category_to_ivecs[cat]      # unitvecs???		
		origin_2d_array = np.zeros_like(vector_2d_array)
		
		arrows_array = np.concatenate((origin_2d_array, vector_2d_array), axis=1)
		X0, Y0, U, V = zip(*arrows_array)
		ax.quiver(X0, Y0, U, V, angles='xy', scale_units='xy', scale=1, color=category_to_color[cat], label=cat)
		
	ax.set_xlim([-1.05, 1.05])
	ax.set_ylim([-1.05, 1.05])
	
	plt.legend(loc='upper left')		# 'best' came out upper right. Upper left looks better.
	
	plt.draw()
	plt.savefig(filename)
	plt.close(thefigure)

		

def i_txt_set_of_categories(i, category_to_words, category_to_inds, category_to_ivecs, category_to_iunitvecs, eigshape, filename):
	outfile = open(filename, mode='w')
	print("Eigenvectors matrix:", eigshape, file=outfile)    # eigshape
	print("\nx_axis: coordinate", str(0), file=outfile)
	print("y_axis: coordinate", i, file=outfile)
	
	for cat in category_to_words:
		print("\n[", cat, "]", file=outfile)
		C = len(category_to_words[cat])
		words = category_to_words[cat]
		inds  = category_to_inds[cat]
		i_vecs = category_to_ivecs[cat]
		i_unitvecs = category_to_iunitvecs[cat]
		lengths       = [np.linalg.norm(v) for v in i_vecs]   		#LIST
		checklengths  = [np.linalg.norm(v) for v in i_unitvecs]
		
		# Print column headings, then vector data
		print('\n%s, %s, %s, , %s, , %s, , %s \n'
			% ('      index', '     word\t', '      i_vecs\t', '     lengths\t', '  i_unitvecs\t', 'checklengths\t'), file=outfile)
		for c in range(C):
			print('%d, %s, %.15f, %.15f, %.4f, , %.15f, %.15f, %.4f'
			% (inds[c], words[c], i_vecs[c,0], i_vecs[c,1], lengths[c], i_unitvecs[c,0], i_unitvecs[c,1], checklengths[c]), file=outfile)
		
	outfile.close()


def i_plot_set_of_words(i_vectors, wordset, color_list, filename):
	thefigure = plt.figure()
	plt.title(filename)
	ax = plt.gca()				# documentation: "Get the current axes, creating one if necessary"
	
	#origin_2d_array = np.zeros((len(i_vectors), 2))     # inner parenthesis needed because 'shape' is a tuple
	vector_2d_array = i_vectors
	origin_2d_array = np.zeros_like(vector_2d_array)
	
	arrows_array = np.concatenate((origin_2d_array, vector_2d_array), axis=1)
	X0, Y0, U, V = zip(*arrows_array)
	ax.quiver(X0, Y0, U, V, angles='xy', scale_units='xy', scale=1, color=color_list)
	
	ax.set_xlim([-1.05, 1.05])	# MAY WANT TO CHANGE TO GET 1:1 ASPECT ratio
	ax.set_ylim([-1.05, 1.05])
	
	legend_entries = list()
	range_limit = min(len(wordset), len(color_list))
	for k in range(range_limit):
		k_patch = mpatches.Patch(color=color_list[k], label=wordset[k])
		legend_entries.append(k_patch)
	plt.legend(handles=legend_entries, bbox_to_anchor=(1.04, 0.5), loc='center left')
	#bbox_to_anchor coords have 0 as bottom of plot, 1 as top; similarly for left-right
	
	plt.draw()
	plt.savefig(filename, bbox_inches='tight')   # kwarg has good effect of not cutting off some of the legend.
	plt.close(thefigure)
	

def i_txt_set_of_words(i, category, words, inds, i_vecs, i_unitvecs, eigshape, filename):  #replace wordset by category_to_words with only one category
	outfile = open(filename, mode='w')	
	print("Eigenvectors matrix:", eigshape, file=outfile)    # eigshape
	print("\nx_axis: coordinate", str(0), file=outfile)
	print("y_axis: coordinate", i, file=outfile)
	
	print("\n[", category, "]", file=outfile)
	C = len(words)
	lengths       = [np.linalg.norm(v) for v in i_vecs]   		#LIST
	checklengths  = [np.linalg.norm(v) for v in i_unitvecs]	
	
	# Print column headings, then vector data
	print('\n%s, %s, %s, , %s, , %s, , %s \n' 
		% ('      index', '     word\t', '      i_vecs\t', '     lengths\t', '  i_unitvecs\t', 'checklengths\t'), file=outfile)
	for c in range(C):
		#print('%d, %s, %.15f, %.15f, %.4f, , %.15f, %.15f, %.4f' 
		print('%d, %s, %.30f, %.30f, %.4f, , %.30f, %.30f, %.4f'
		% (inds[c], words[c], i_vecs[c,0], i_vecs[c,1], lengths[c], i_unitvecs[c,0], i_unitvecs[c,1], checklengths[c]), file=outfile)		
	
	outfile.close()


def make_wordvector_plots(eigenvalues, eigenvectors, wordlist, timestamp):
	print("Word vector plots...")    # Wordvector stuff should be separate module, and then this line would be handled within Lexicon class, on lexicon.py
	
	# alphabetize the colors??  group by resemblance, so easier to compare??
	colors_16 = ['black', 'blue', 'blueviolet', 'crimson', 'cyan', 'darkcyan', 'darkgoldenrod', 'darkslategray', 'deeppink', 'green', 'lime', 'magenta', 'orange', 'seagreen', 'sienna', 'yellow']

	# SET UP EXAMPLES
	#  Each list presented here contains words from a lnguistic category of some sort;
	#  as used in comments below, the word "category" means simply one of these lists.
	
	present_verbs    = ['do',   'work',   'take',  'does', 'find',  'look',   'give',  'turn',   'provide',  'call',   'move',  'hold', 'bring',   'start',   'hear',  'ask']
	past_verbs       = ['did',  'worked', 'took',  'did',  'found', 'looked', 'gave',  'turned', 'provided', 'called', 'moved', 'held', 'brought', 'started', 'heard', 'asked']
	past_prtcpls     = ['done', 'worked', 'taken', 'done', 'found', 'looked', 'given', 'turned', 'provided', 'called', 'moved', 'held', 'brought', 'started', 'heard', 'asked']
	present_prtcpls  = ['doing', 'working', 'taking', 'doing', 'finding', 'looking', 'giving', 'turning', 'providing', 'calling', 'moving', 'holding', 'bringing', 'starting', 'hearing', 'asking']
	
	strpresent  = ['do',   'give',  'take']
	strpast     = ['did',  'gave',  'took']
	strpastpart = ['done', 'given', 'taken']
	strprpart   = ['doing', 'giving', 'taking']
	
	# (Variants)
	present_verbs_1 = ['do',  'work',   'take', 'does', 'find',  'look',   'give', 'turn',   'provide',  'call',   'move',  'hold', 'bring',   'start',   'hear',  'ask',   'state',  'state',     'involve',  'involve']       # Add 'blue', 'blue', 'red', 'red' to some_colors        ,     'move',  'move'
	past_verbs_1    = ['did', 'worked', 'took', 'did',  'found', 'looked', 'gave', 'turned', 'provided', 'called', 'moved', 'held', 'brought', 'started', 'heard', 'asked', 'stated', 'statement', 'involved', 'involvement']    # so blue will show   state_stated and state_statement  , 'moved', 'movement'   																																																						     # and red will show   involves_involved and involves_involvement
	# present_verbs_2 = ['do',  'work',   'take', 'does', 'find',  'look',   'give', 'turn',   'provide',  'call',   'move',  'hold', 'bring',   'start',   'hear',  'gold']         # Any different behavior re final pair?
	# past_verbs_2    = ['did', 'worked', 'took', 'did',  'found', 'looked', 'gave', 'turned', 'provided', 'called', 'moved', 'held', 'brought', 'started', 'heard', 'science']
	strpresent_1 = ['do',  'do',   'give', 'give',  'take', 'take']     # Use colors  cyan cyan black black red red
	strpastpp_1  = ['did', 'done', 'gave', 'given', 'took', 'taken']
	strpres_2  = ['do',  'doing', 'give', 'giving', 'take', 'taking']
	strpast_2  = ['did', 'done',  'gave', 'given',  'took', 'taken']
	
	modal1 = ['will',  'can',   'may',   'must', 'shall']                           # ORDERED BY WOrDLIST    2017_07_04
	modal2 = ['would', 'could', 'might', 'must', 'should']
	modal_verbs = ['would', 'will', 'can', 'could', 'may', 'must', 'might', 'should', 'shall']    # ORDERED BY WOrDLIST    2017_07_04
	
	nouns_regverbs = ['state', 'work', 'look', 'turn', 'call', 'move', 'start']		# ORDERED BY WORDLIST  2017_06_25
	nouns_strverbs = ['take', 'find', 'give', 'hold']								# ORDERED BY WORDLIST  2017_06_25
	nouns_morestrverbs = ['say', 'lead', 'break', 'spoke', 'buy']					# ORDERED BY WORDLIST  2017_06_25
	nouns_prptcpls = ['doing', 'working', 'moving', 'giving', 'hearing', 'finding', 'calling']	# ORDERED BY WORDLIST  2017_06_25
		
	more_strpresent  = ['go',   'say',  'lead', 'speak',  'write',   'break',  'buy',    'seek']
	more_strpast     = ['went', 'said', 'led',  'spoke',  'wrote',   'broke',  'bought', 'sought']
	more_strpastpart = ['gone', 'said', 'led',  'spoken', 'written', 'broken', 'bought', 'sought']
	more_strprespart = ['going', 'saying', 'leading', 'speaking', 'writing', 'breaking', 'buying', 'seeking']
	
	adj = ['great',   'hard',   'particular',   'final',   'natural',   'actual',   'previous',   'quick',   'slow']     # what about 'mighti'? 'heavi', 'happi', 'notabl', 'simpl'  # ORDERED BY WORDLIST  2017_06_26
	adv = ['greatly', 'hardly', 'particularly', 'finally', 'naturally', 'actually', 'previously', 'quickly', 'slowly'] # gracefully
	
	moreadj = ['little', 'small', 'social', 'big', 'political', 'economic', 'red', 'heavy', 'pretty', 'bright', 'quiet', 'soft']	# ORDERED BY WORDLIST  2017_06_27
	#adj_comp
	#adj_sup
	
	moreadv1 = ['very', 'simply', 'suddenly', 'nearly', 'clearly', 'easily', 'closely', 'fairly', 'truly', 'thoroughly', 'strongly', 'gently', 'roughly', 'happily'] 
	moreadj1 = ['very', 'simple', 'sudden',   'near',   'clear',   'easy',   'close',   'fair',   'true',  'thorough',   'strong',   'gentle', 'rough',   'happy']     #removed 'approximate'  #7459 in wordlist
	extraadv = ['successfully', 'rapidly', 'quietly', 'softly', 'regularly', 'vigorously']    # needs 7500  socially, politically, economically
	extraadj = ['successful',   'rapid',   'quiet',   'soft',   'regular',   'calm',   'vigorous']      
	
	verb0    = ['state',     'move',     'agree',     'manage',     'adjust',     'announce',     'engage',     'enjoy',     'involve']       #, 'abandon',     'attain']    # involve is #3471; involves is #2697
	verbment = ['statement', 'movement', 'agreement', 'management', 'adjustment', 'announcement', 'engagement', 'enjoyment', 'involvement']   #, 'abandonment', 'attainment' # ORDERED BY WorDLIST  2017_06_25
	
	verb0past = ['stated', 'agreed', 'enjoyed', 'managed', 'announced', 'adjusted', 'engaged', 'involved', 'moved']
	# verb0pastptcpl same as verb0past
	verb0subset    = ['state',   'enjoy',    'adjust',    'involve',   'move']
	# verb0presptcpl = ['stating', 'enjoying', 'adjusting', 'involving', 'moving']   #adjusting is #7438   #moving is #9748
	verb0presptcpl = ['stating', 'enjoying', 'involving']   
	
	masc_noun   = ['he',  'man',   'boy',  'father', 'husband', 'brother', 'uncle', 'male']     # colt(5247)   # ordered by wordlist 2017_06_25
	fem_noun    = ['she', 'woman', 'girl', 'mother', 'wife',    'sister',  'aunt',  'female']   # filly(8707)
	
	pure_nouns = ['city', 'basis', 'lady', 'dog', 'region', 'memory', 'cat']   		# ordered by wordlist 2017_06_25
	tion = ['action', 'information', 'question', 'position', 'education', 'section', 'attention', 'production', 'addition', 'function', 'tradition', 'application', 'connection', 'portion', 'dictionary']  # ALREADY IN ORDER AS OF 2016_06_25
	
	subj_pron = ['i',  'you', 'he',  'she', 'it', 'we', 'they']
	obj_pron  = ['me', 'you', 'him', 'her', 'it', 'us', 'them']
	poss_adj  = ['my', 'your', 'his', 'her', 'its', 'our', 'their']
	poss_pron = ['mine', 'yours', 'his', 'hers', 'its', 'ours', 'theirs']
	
	articles = ['the', 'a']
	det = ['the', 'this', 'that', 'these', 'those', 'a', 'any', 'another', 'other'] 
	
	prep = ['of', 'to', 'in', 'for', 'with', 'on', 'by', 'about', 'over', 'after', 'before', 'through', 'under', 'except', 'below', 'despite']
	
	# ADDED January 4, 2018
	coord_conj = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']
	correl_conj = ['both', 'either', 'neither', 'whether', 'and', 'or', 'nor', 'but']
	sub_conj_a  = ['after', 'although', 'as', 'because', 'before', 'if', 'lest', 'once', 'only', 'provided', 'since']
	sub_conj_s  = ['since', 'so', 'that', 'than', 'though', 'till', 'unless', 'until', 'when']   # removed 'supposing' - #26532
	sub_conj_w  = ['when', 'whenever', 'where', 'whereas', 'wherever', 'whether', 'while']
	
	misc = ['just', 'fully', 'quite', 'indeed', 'still', 'yet']   #picked up from looking up “even”  in dictionary.com
	
	# MOVED UPWARD - IMMEDIATELY FOLLOWING PRODUCTION OF EIGENVECTORS  
	## STANDARDIZE EIGENVECTORS FOR CONSISTENCY OF PLOTS     update for n_eigenvectors=3   (and standardize for any n)
	#test_ind = wordlist.index('the')
	#test_vec = eigenvectors[test_ind, :]
	#print("test_vec:", test_vec)	# standardize to  (positive, negative) coordinates for 'the'
	#if (test_vec[0] < 0):
	#	eigenvectors[:,0] = -1 * eigenvectors[:,0]
	#if (test_vec[1] > 0):
	#	eigenvectors[:,1] = -1 * eigenvectors[:,1]
	#if (len(test_vec)>2):
	#	if (test_vec[2] < 0):
	#		eigenvectors[:,2] = -1 * eigenvectors[:,2]
	
	universal_category_to_words = \
	{
	'present_verbs'   : ['do',   'work',   'take',  'does', 'find',  'look',   'give',  'turn',   'provide',  'call',   'move',  'hold', 'bring',   'start',   'hear',  'ask'],
	'past_verbs'      : ['did',  'worked', 'took',  'did',  'found', 'looked', 'gave',  'turned', 'provided', 'called', 'moved', 'held', 'brought', 'started', 'heard', 'asked'],
	'past_prtcpls'    : ['done', 'worked', 'taken', 'done', 'found', 'looked', 'given', 'turned', 'provided', 'called', 'moved', 'held', 'brought', 'started', 'heard', 'asked'],
	'present_prtcpls' : ['doing', 'working', 'taking', 'doing', 'finding', 'looking', 'giving', 'turning', 'providing', 'calling', 'moving', 'holding', 'bringing', 'starting', 'hearing', 'asking'],
	'adv'             : ['greatly', 'hardly', 'particularly', 'finally', 'naturally', 'actually', 'previously', 'quickly', 'slowly'],
	'adj'             : ['great',   'hard',   'particular',   'final',   'natural',   'actual',   'previous',   'quick',   'slow'],
	'det'             : ['the', 'this', 'that', 'these', 'those', 'a', 'any', 'another', 'other'],
	'coord_conj'      : ['for', 'and', 'nor', 'but', 'or', 'yet', 'so'],
	'rel_eig'         : ['wires', 'correctly', 'farms', 'greene']
	}
	
	universal_category_to_color = \
	{
	'present_verbs'   : 'black', 
	'past_verbs'      : 'blue', 
	'past_prtcpls'    : 'blueviolet',  
	'present_prtcpls' : 'crimson', 
	'adv'             : 'cyan', 
	'adj'             : 'darkcyan',
	'det'             : 'darkgoldenrod', 
	'coord_conj'      : 'darkslategray'
	}
	
	
	# -------------------------------------------------
	#   UNCOMMENT AND EDIT TO SPECIFY WHAT TO DISPLAY
	# -------------------------------------------------
	C = eigenvectors.shape[0]		# length(wordlist)
	N = eigenvectors.shape[1]		# number of eigenvectors
	eigshape = str(C) + "x" + str(N)
	
	# OBSERVE LINGUISTIC CATEGORY BY ITS ELEMENTS (one color per element)  [new, Jan. 2018]
	#dirname = "set_of_words"
	#if not os.path.exists(dirname):
	#	os.mkdir(dirname)
	#os.chdir(dirname)	# return required!
	
	#cat = 'rel_eig'
	#idstring_a = '-set_of_words.' + eigshape + '.' + cat + timestamp.strftime(".%Y_%m_%d.%H_%M")
	#color_list = colors_16
	#words = universal_category_to_words[cat]
	#inds = [wordlist.index(word) for word in words]
	#vecs = eigenvectors[inds, :]
	#for i in range(1,N):
	#	i_vecs = vecs[:,[0,i]]
	#	i_unitvecs = skl_normalize(i_vecs, norm='l2')
	#	i_plot_set_of_words(i_unitvecs, words, color_list, str(i)+idstring_a+'.png')
	#	#i_txt_set_of_words(i, i_vecs, i_unitvecs, str(i)+idstring_a+'.txt', wordset, inds)
	#	i_txt_set_of_words(i, cat, words, inds, i_vecs, i_unitvecs, eigshape, str(i)+idstring_a+'.csv')
		
	#os.chdir("..")	
	
	# OBSERVE PAIR OF CATEGORIES BY RELATED WORDS (one color per word pair)
	leftwords  = present_verbs
	rightwords = past_verbs    # if intend to plot only leftwords: shapes must match because diff gets calculated (unnecessarily) - so may use same category as for leftwords
	label = 'present_past'
	
	#leftunitvecs, rightunitvecs, leftprenorm_vecs, rightprenorm_vecs = wordvector_LR_info(eigenvalues, eigenvectors, wordlist, leftwords, rightwords, eigshape, label, timestamp)
	## UNITVECS
	#plot_LR_pair(leftunitvecs, rightunitvecs, eigshape, label, timestamp, leftwords, show_left=True, show_right=True, show_diff_LR=True, show_diff_origin=False)
	##		(i) uses default parameters (see function definition)
	##      (ii) may pass prenorm vectors if desired instead of unitvecs
	##      (iii) leftwords provides labels for legend
	## VECS (not unit length)
	#plot_LR_pair(4*leftprenorm_vecs, 4*rightprenorm_vecs, eigshape, label, timestamp, leftwords, show_left=True, show_right=False, show_diff_LR=False, show_diff_origin=False)
	
	
	# OBSERVE CATEGORY LOCATIONS      # OBSERVE CATEGORY LOCATIONS section.  This section is under development Jan 12, 2018. Will replace OBSERVE MULTIPLE CATEGORIES (below). 
	dirname = "set_of_categories"
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	os.chdir(dirname)	# return required!
		 
	categories_list = ['adv', 'det']
	idstring_b = '-set_of_categories.' + eigshape + '.' + '_'.join(categories_list) + timestamp.strftime(".%Y_%m_%d.%H_%M")
	#category_color_dict =
	category_to_words = dict()
	category_to_inds  = dict()
	category_to_vecs  = dict()
	for cat in categories_list:
		category_to_words[cat] = universal_category_to_words[cat]
		category_to_inds[cat] = [wordlist.index(word) for word in category_to_words[cat]]	# N.B. The use of '[' and ']' make this a list!
		category_to_vecs[cat] = eigenvectors[category_to_inds[cat], :]     					# all N coordinates (for each word in the given category)
		#print("category =", cat)
		#print("category_to_inds =", category_to_inds[cat])
		#print("category_to_vecs =", category_to_vecs[cat])
		#print()
	
	
	#PRINT THIS STUFF TO CHECK IT (AND TO GET THE DICTIONARY SYNTAX CORRECT)
	
	for i in range(1,N):
	#	print("i =", i)
		category_to_ivecs = dict()
		category_to_iunitvecs = dict()
		for cat in categories_list:
			category_to_ivecs[cat] = category_to_vecs[cat][:,[0,i]]	# only the 0th and ith coordinates
			category_to_iunitvecs[cat] = skl_normalize(category_to_ivecs[cat], norm='l2')
			#print("cat =", cat)
			#print("category_to_ivecs[cat] =", category_to_ivecs[cat])
			#print()
		i_plot_set_of_categories(category_to_iunitvecs, str(i)+idstring_b+'.png', universal_category_to_color)
		i_txt_set_of_categories(i, category_to_words, category_to_inds, category_to_ivecs, category_to_iunitvecs, eigshape, str(i)+idstring_b+'.csv')
	os.chdir("..")
	

	# OBSERVE MULTIPLE CATEGORIES (one color per category)
	#classes_words = [present_verbs, past_verbs, adj, adv,     verb0,   verbment,   det, strpastpart]       # list of lists
	#color_per_class = ['blue', 'cyan', 'magenta', 'deeppink', 'lime', 'seagreen', 'brown',  'black' ]	# may choose related shades as appropriate
	#label = "present_past_adj_adv_verb0_verbment_det_strpastpart"
	#some_colors   = ['blue', 'green', 'crimson', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'sienna', 'darkcyan', 'darkgoldenrod', 'deeppink', 'darkslategray', 'blueviolet', 'seagreen', 'orange']
	
	# TURN THIS ON TO TEST  JULY 21, 2018
	classes_words = [present_verbs, present_prtcpls, past_verbs, past_prtcpls, prep]				#[pure_nouns, verbment, tion, adj, moreadj]
	color_per_class = ['blue', 'lime',  'cyan', 'cyan', 'black']		#['blue', 'mediumturquoise', 'dimgray', 'deeppink', 'blueviolet']     # gray?  cornflowerblue and darkblue?  lightsteelblue?
	label = "pres_prptcpls_past_pastptcpls_prep"						#"pure_ment_tion_adj_moreadj"
	
	classes_unitvecs, classes_vecs = wordvector_multiclasses_info(eigenvalues, eigenvectors, wordlist, classes_words, eigshape, label, timestamp)
	# PLOT UNITVECS   Not so applicable for 3-dim. Is there any reason to like unitvecs here even for 2-dim?
	#plot_multiclasses_xy(classes_unitvecs, color_per_class, eigshape, label, timestamp)
	#plot_multiclasses_yz(classes_unitvecs, color_per_class, eigshape, label, timestamp)
	##plot_multiclasses_3d(classes_unitvecs, color_per_class, eigshape, label, timestamp)
	
	# PLOT VECS (not unit length)
	plot_multiclasses_xy(classes_vecs, color_per_class, eigshape, label, timestamp)
	plot_multiclasses_xz(classes_vecs, color_per_class, eigshape, label, timestamp)
	plot_multiclasses_yz(classes_vecs, color_per_class, eigshape, label, timestamp)
	##plot_multiclasses_3d(classes_vecs, color_per_class, eigshape, label, timestamp)
    

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
	
	
def eigenvector_clusterwise_plots(algorithm, eigenvectors, cluster_labels, diameter, output_dir, timestamp_string):
	
	# Purpose: understand and improve clustering.
	# For each eigenvector, plot entries sorted first by cluster label, then within each cluster as follows: 
	#    a - by value (incr)             
	#    b - by wordlist index (incr)
	#    c - by diameter (incr)
	#    d - not specified
	
	(C, N) = eigenvectors.shape
	x_entries = np.arange(C)
	
	# a: secondary sort by value
	for n in range(N):		# nth eigenvector
		ya_entries = eigenvectors[np.lexsort((np.arange(C), eigenvectors[:,n], cluster_labels)), n]
		plot_as_specified(x_entries, ya_entries, 'blue', output_dir + algorithm + ".eig" + str(n) + "._value" + timestamp_string, scatter=True)
		print(np.lexsort((np.arange(C), eigenvectors[:,n], cluster_labels)))
		
	# b: secondary sort by wordlist index	[compute order once, same for all eigs; similarly for c and d below]
	pointers_b = np.lexsort((np.arange(C), cluster_labels))
	print("pointers_b:  ", pointers_b)
	for n in range(N):
		yb_entries = eigenvectors[pointers_b, n]
		plot_as_specified(x_entries, yb_entries, 'green', output_dir + algorithm + ".eig" + str(n) + "._index" + timestamp_string, scatter=True)
		
	# c: secondary sort by diameter
	pointers_c = np.lexsort((diameter, cluster_labels))
	print("pointers_c:  ", pointers_c)
	for n in range(N):
		yc_entries = eigenvectors[pointers_c, n]
		plot_as_specified(x_entries, yc_entries, 'goldenrod', output_dir + algorithm + ".eig" + str(n) + "._diam" + timestamp_string, scatter=True)
		
	# d: secondary sort not specified  - what is the order, and why do we see structure??
	pointers_d = np.argsort(cluster_labels)
	print("pointers_d:  ", pointers_d)
	for n in range(N):
		yd_entries = eigenvectors[pointers_d, n]
		plot_as_specified(x_entries, yd_entries, 'red', output_dir + algorithm + ".eig" + str(n) + ".__" + timestamp_string, scatter=True)
		
	
	# 2d CLUSTER PLOT     may want to pass in number of clusters   use K sim to C, N
	K = N	# K is number of clusters. num_clusters same as num_eigenvectors for initial development, but not necessary.
	LUT = [
	'#646464',    #nice charcoal
	'#96addb',    #blue violet
	'#00ff00',    #lime   #Try lime instead  If good, change here and at Google Sheets  '#c0c0c0',    #silver
	'#ff00ff',    #fuchsia
	'#2c8840',    #deeper green
	'#5ed1b7',    #nice aqua
	'#0000ff',  #blue
	'#ffff00',  #yellow
	'#00ffff',  #aqua
	'#800000',  #maroon
	'#008000',  #green
	'#6666ff',  #was navy  #000080
	'#808000',  #olive
	'#800080',  #purple
	'#008080',  #teal
	'#808080',  #gray
	'#c0c0c0',  #silver	 #Switched positions of lime and silver because silver didn't show up well on plots  '#00ff00',  #lime
	'#d15eb7',
	'#2c5088',
	'#004040',
	'#400040',
	#'#ff0000',  #red
	#'#fa3fc9',  #pinker fuchsia
	]
	
	
	# copied from "plot_as_specified"
	fig, ax = plt.subplots()
	title_string = output_dir + algorithm + ".2dClusters" + timestamp_string
	plt.title(title_string)
	#if not ylimits==None:    Work on this  -- for outliers
	#	plt.ylim(ylimits)
	
	
	for k in range(K):
		print("Cluster " + str(k))
		pointers_clstr = np.asarray([ptr for ptr in pointers_b if cluster_labels[ptr]==k])
		print(pointers_clstr)
		print("length of pointers_clstr =", len(pointers_clstr))	# number of elts of Cluster k
		# so far, so good !
		
		x_coords = eigenvectors[pointers_clstr, 1]   # coords in wordlist order on eig1 for Cluster k 
		y_coords = eigenvectors[pointers_clstr, 2]   # in wordlist order on eig2
		ax.scatter(x_coords, y_coords, marker='.', s=1, color=LUT[k])   # Note: s is special parameter for scatter function (in points^^2 float)
		#plot_as_specified(x_coords, y_coords, LUT[k], output_dir + algorithm + ".2dCluster" + str(k) + timestamp_string, scatter=True)
		
	fig.savefig(title_string + ".png")
	plt.close(fig)



def NOT_NOW_eigenvector_clusterwise_plots(algorithm, eigenvectors, cluster_labels, diameter, output_dir, timestamp_string):
	# SEE Worksheet3 in generate_eigenvector_spreadsheet()
	(C, N) = eigenvectors.shape
	
	## SORT EACH EIGENVECTOR FIRST BY CLUSTER, THEN BY COORD VALUE (INCREASING), USING LEXSORT
	#lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix  
	#lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	#for n in range(N):
	#	lex_sortation[:,n] = np.lexsort((np.arange(C), eigenvectors[:,n], cluster_labels))
		
	#for n in range(1,N):
		#lex_sortation_rel[:,n] = np.lexsort((diameter, eig_rel[:,n], cluster_labels))
		
		
	# On each eigenvector, sort entries
	#    a - by cluster (incr)
	# then within each cluster
	#    b - by value (incr)
	#    c - by wordlist index (decr)
	#    d - by diameter (incr)
	
	# Possible to do this more efficiently. For a,c,d, index order determined by lexsort is same across all eigenvectors.
	# May do with top level the type of sort, for each eigenvector.
	
	x_entries = np.arange(C)
	for n in range(N):		# nth eigenvector
		ya_entries = eigenvectors[np.argsort(cluster_labels), n]
		plot_as_specified(x_entries, ya_entries, 'r', output_dir + algorithm + ".eig._." + str(n) + timestamp_string, scatter=True)
		#print(np.argsort(cluster_labels))
		
		yb_entries = eigenvectors[np.lexsort((np.arange(C), eigenvectors[:,n], cluster_labels)), n]
		#yb_entries = eigenvectors[lex_sortation[:,n], n]    # condense from above
		plot_as_specified(x_entries, yb_entries, 'b', output_dir + algorithm + ".eig._value" + str(n) + timestamp_string, scatter=True)
		#print(np.lexsort((np.arange(C), eigenvectors[:,n], cluster_labels)))
		
		yc_entries = eigenvectors[np.lexsort((np.arange(C), cluster_labels)), n]    # for decreasing order, use no,arange(C)[::-1]
		plot_as_specified(x_entries, yc_entries, 'g', output_dir + algorithm + ".eig._index." + str(n) + timestamp_string, scatter=True)
		#print(np.lexsort((np.arange(C)[::-1], cluster_labels)))
		
		yd_entries = eigenvectors[np.lexsort((diameter, cluster_labels)), n]
		plot_as_specified(x_entries, yd_entries, 'goldenrod', output_dir + algorithm + ".eig._diam." + str(n) + timestamp_string, scatter=True)
		#print(np.lexsort((diameter, cluster_labels)))
		
	# CLUSTER PLOT   Need to work on this to get colors.
	if (N>2):
		x_coords = eigenvectors[:,1]
		y_coords = eigenvectors[:,2]
		plot_as_specified(x_coords, y_coords, 'magenta', output_dir + algorithm + ".2d" + timestamp_string, scatter=True)
		
		
		
		
		
		



def eigenvector_plots(wordlist, eigenvectors, timestamp_string):
	(C, N) = eigenvectors.shape        # YES! Use this elsewhere also
	#print("C =", C, " and N =", N)
	
	# NON-RELATIVIZED VERSION HERE; RELATIVIZED VERSION BELOW

	dirname = "eigenvector_plots"
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	os.chdir(dirname)	# return required!
	
	for i in range(N):		# ith eigenvector
		
		# EIGENVECTOR, SHOWING VALUES IN COORDINATE ORDER
		x_entries = np.arange(C)		# enumeration of vector coords understood to be in wordlist order
		y_entries = eigenvectors[:,i]	# corresponding values
		
		# See "Adjustment for outliers" note in "RELATIVIZED VERSION" section below
		if i==3:
			y_entries[wordlist.index('bombers')]   = 0	
			y_entries[wordlist.index('ballistic')] = 0				
		plot_as_specified(x_entries, y_entries, 'b', str(i)+"-eigenvector_scatter"+timestamp_string, scatter=True)
		
		# SORTED EIGENVECTOR, SHOWING VALUES IN ASCENDING ORDER 
		x_entries = np.arange(C)				# enumeration understood to be of values in ascending order
		y_entries = sorted(eigenvectors[:,i])	# values in ascending order
		#if i==2:
		#	y_entries[-1] = y_entries[-3]		# last entry             # oops -- the 2 zeroes which were forced above were sorted as zeroes
		#	y_entries[-2] = y_entries[-3]		# next-to-last entry				
		plot_as_specified(x_entries, y_entries, 'limegreen', str(i)+"-sorted_eig"+timestamp_string, scatter=True)
		
		# EIGENVECTOR SORTED AS ABOVE BY INCREASING VALUE, BUT SHOWING INSTEAD THE WORD COORDINATE WHERE EACH VALUE OCCURS 
		x_entries = np.arange(C)										# enumeration understood to be of values in ascending order
		y_entries = np.argsort(eigenvectors[:,i], kind='mergesort')		# wordlist indices corresponding to increasing eigenvector values
		
		plot_as_specified(x_entries[:2500], y_entries[:2500], 'm', str(i)+"-argsort_low2500"+timestamp_string)
		plot_as_specified(x_entries[2500:5000], y_entries[2500:5000], 'm', str(i)+"-argsort_mid2500"+timestamp_string)
		plot_as_specified(x_entries[5000:7000], y_entries[5000:7000], 'm', str(i)+"-argsort_high2000"+timestamp_string)
		
		# NOTE 1 - Hardcoded numbers here of course are specific for max_word_types = 7000
		# NOTE 2 - View particular subregions, in line or scatter plot form, as desired.
		# NOTE 3 - Use of both "sorted" and "argsort" isn't efficient; but for clarity, leave as is for now.
				
	os.chdir("..")
	
	# RELATIVIZED VERSION 
	dirname = "relative_eigenvector_plots"
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	os.chdir(dirname)	# return required!
	
	for i in range(1,N):		# ith eigenvector relative to eig_0
		eigi_rel_eig0 = eigenvectors[:,i] / eigenvectors[:,0]	# elementwise division; INVESTIGATE SMALL VALUES ON EIG0
																# MAY ALSO MAKE YLIMITS (AS USED BELOW) UNNECESSARY
		
		# Adjustment for outliers
		# The huge span in y direction due to these outliers obscures any distinction among entries other than these two.
		# Resetting the y-value for these outliers is only for plotting, and makes observance of the generic behavior possible. 
		# See file eigenvector_data_for_excel.csv for the unadjusted values.
		if i==2 or i==3:
			eigi_rel_eig0[wordlist.index('bombers')]   = 0		# index of 'bombers'
			eigi_rel_eig0[wordlist.index('ballistic')] = 0		# index of 'ballistic'
		
		
		# EIGENVECTOR, SHOWING VALUES IN COORDINATE ORDER
		x_entries = np.arange(C)		# enumeration of vector coords understood to be in wordlist order
		y_entries = eigi_rel_eig0		# corresponding values 	
		#plot_as_specified(x_entries, y_entries, 'r', str(i)+"-rel_eigenvector_scatter"+timestamp_string, scatter=True, ylimits=(-12,8))
		plot_as_specified(x_entries, y_entries, 'r', str(i)+"-rel_eigenvector_scatter"+timestamp_string, scatter=True)
		
		# SORTED EIGENVECTOR, SHOWING VALUES IN ASCENDING ORDER 
		x_entries = np.arange(C)				# enumeration understood to be of values in ascending order
		y_entries = sorted(eigi_rel_eig0)		# values in ascending order
		#plot_as_specified(x_entries, y_entries, 'darkorange', str(i)+"-rel_sorted_eig"+timestamp_string, scatter=True, ylimits=(-12,8))
		plot_as_specified(x_entries, y_entries, 'darkorange', str(i)+"-rel_sorted_eig"+timestamp_string, scatter=True)
		
		# EIGENVECTOR SORTED AS ABOVE BY INCREASING VALUE, BUT SHOWING INSTEAD THE WORD COORDINATE WHERE EACH VALUE OCCURS 
		x_entries = np.arange(C)										# enumeration understood to be of values in ascending order
		y_entries = np.argsort(eigi_rel_eig0, kind='mergesort')			# wordlist indices ordered by increasing values
		
		plot_as_specified(x_entries[:2500], y_entries[:2500], 'darkgoldenrod', str(i)+"-rel_argsort_low2500"+timestamp_string)
		plot_as_specified(x_entries[2500:5000], y_entries[2500:5000], 'darkgoldenrod', str(i)+"-rel_argsort_mid2500"+timestamp_string)
		plot_as_specified(x_entries[5000:7000], y_entries[5000:7000], 'darkgoldenrod', str(i)+"-rel_argsort_high2000"+timestamp_string)
		
		# NOTE - May try dividing by count instead of by eig0 values. 
		#        Or maybe context count.
	
	os.chdir("..")
	
	
def eig0_diameter_ascending_plots(eig0_sorted, diameter_sorted, timestamp_string):
	n_words = len(eig0_sorted)
	#print("In eig0_diameter_ascending_plots, n_words =", n_words)
	dirname = "eig0_diameter_ascending_plots"
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	os.chdir(dirname)	# return required!
	
	# PLOT ORDERED DIAMETER VALUES AS ENUMERATION 	
	x_entries = np.arange(n_words)		# enumeration of vector coords understood to be in wordlist order
	y_entries = diameter_sorted[:]
	plot_as_specified(x_entries, y_entries, 'g', "diameter_sorted"+timestamp_string, scatter=True)
	
	# PLOT ORDERED Eig0 VALUES AS FUNCTION OF ORDERED DIAMETER VALUES
	x_entries = diameter_sorted[:]
	y_entries = eig0_sorted[:]
	plot_as_specified(x_entries, y_entries, 'c', "eig0_against_diameter_sorted"+timestamp_string, scatter=True)
	
	
	
	os.chdir("..")
	

def plot_as_specified(x, y, clr, title_string, scatter=False, ylimits=None):
	fig, ax = plt.subplots()
	plt.title(title_string)
	if not ylimits==None:
		plt.ylim(ylimits)
	if scatter==True:
		ax.scatter(x, y, marker='.', s=1, color=clr)   # Note: s is special parameter for scatter function (in points^^2 float)
	else:
		ax.plot(x, y, clr)
	fig.savefig(title_string + ".png")
	plt.close(fig)
	
	
	
def Chicago_get_laplacian(affinity_matrix):
	n_words = affinity_matrix.shape[0]
	diam_array = compute_diameter_array(n_words, affinity_matrix)		# Revise - omit n_words parameter
	incidence_graph = compute_incidence_graph(n_words, diam_array, affinity_matrix)
	laplacian_matrix = compute_laplacian(diam_array, incidence_graph)
	del incidence_graph
	return laplacian_matrix, np.sqrt(diam_array)
	
	
def spectral_clustering_sym(eigenvectors, random_seed=None):
	# For any matrix named as eigenvectors_xxx in this program, 
    #  each column is an eigenvector of some particular sort of laplacian
    #  each row represents a word as a tuple in that coordinate system
	
	# Step 1: Obtain eigenvectors of L_sym
	eigenvectors_sym = eigenvectors	  # At input, columns are L_sym unit-length eigenvectors.
	
	# Step 2: Get the word vectors for clustering 
	wordcoords_sym = skl_normalize(eigenvectors_sym, norm='l2', axis=1)		# row => unit-length
	
	# Step 3: apply clustering algorithm
	# Use the object-oriented version for access to cluster centers. Mostly we take the default parameter values.
	kmeans_clustering = sklearn.cluster.KMeans(n_clusters=eigenvectors.shape[1], random_state=random_seed).fit(wordcoords_sym)
	clusterlabels  = kmeans_clustering.labels_
	clustercenters = kmeans_clustering.cluster_centers_
	
	return wordcoords_sym, clusterlabels, clustercenters
	
def spectral_clustering_rw(eigenvectors, sqrt_diam, random_seed=None):
	# For any matrix named as eigenvectors_xxx in this program, 
    #  each column is an eigenvector of some particular sort of laplacian
    #  each row represents a word as a tuple in that coordinate system
	
	# Step 1: Obtain eigenvectors of L_rw
	eigenvectors_rw = eigenvectors / sqrt_diam[:, np.newaxis]			# At input, columns are L_sym unit-length eigenvectors.
	eigenvectors_rw = skl_normalize( eigenvectors_rw, norm='l2', axis=0 )   # Consider this
	
	# Step 2: Get the word vectors for clustering 
	wordcoords_rw = eigenvectors_rw		# algorithm does not call for row-based modification
	
	# Step 3: apply clustering algorithm
	# Use the object-oriented version for access to cluster centers. Mostly we take the default parameter values.
	kmeans_clustering = sklearn.cluster.KMeans(n_clusters=eigenvectors.shape[1], random_state=random_seed).fit(wordcoords_rw)
	clusterlabels  = kmeans_clustering.labels_
	clustercenters = kmeans_clustering.cluster_centers_
	
	return wordcoords_rw, clusterlabels, clustercenters
	

# Not in use in this form
def sk_lifted_spectral_clustering(affinity_matrix, num_eigenvectors, num_clusters, kmeans_random_state):
	# see note below about possibly using sparse affinity_matrix
	# latter 3 args could be kwargs, for clarity. 
	# Other kwargs for sklearn functions used within this function are omitted as not likely to be used.
	
	# N.B. In spectral_embedding definition, norm_laplacian = True  and eigen_solver='arpack' by default.
	#maps = spectral_embedding_aus_sklearn(affinity_matrix, n_components=num_eigenvectors, drop_first=False)
	#maps = spectral_embedding(affinity_matrix, n_components=num_eigenvectors, drop_first=False)
	#maps = skl_normalize(maps, norm='l2', axis=0)   ##### UNIT MAPS  FOR INVESTIGATING EFFECT   Oct.  9, 2018
	##################################################### DECIDED THAT INTENT IS UNIT VECTORS   Oct. 12, 2018	
	
	# Our affinity matrix  scm_csr  is sparse.  Returns laplacian as a numpy (dense) array. 
	# I'd rather give it dense affinity.  But try with sparse, since there was an issue needing np.asarrau
	laplacian, dd = csgraph_laplacian(affinity_matrix, normed=True, return_diag=True)   # returns L_sym
	laplacian_sparse = sparse.csr_matrix(laplacian)
	eigenvalues, eigenvectors = linalg.eigsh(laplacian_sparse, k=num_eigenvectors, which='SM')  #N.B. eigs complex; eigsh not
	
	# TEMPORARY - TO CHECK NEW CODE (extraction) AGAINST PREVIOUS BUT RECENT (i.e., sklearn 20.0)
	# APPROACH L_rw. Not necessarily temporary. sklearn and Chicago match on this.
	eigenvectors = eigenvectors / dd[:, np.newaxis]
	eigenvectors = skl_normalize(eigenvectors, norm='l2', axis=0)
	
	# APPROACH L_sym.
	# NOTE THAT THE EIGENVECTORS THEMSELVES ARE UNIT LENGTH AS RETURNED FROM linalg.eigsh. NEXT LINE IS FOR ROWS.
	#eigenvectors = skl_normalize(eigenvectors, norm='l2', axis=1)
	
	# STANDARDIZE DIRECTION apply this to the eigenvectors in whatever form is about to be submitted to clustering algorithm
	eigenvectors = (_deterministic_vector_sign_flip(eigenvectors.T)).T  # Probably not right; at this point they're not eigenvectors.
	
	# Use the object-oriented version for access to cluster centers.
	kmeans_clustering = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=kmeans_random_state).fit(eigenvectors)
	#kmeans_clustering = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=kmeans_random_state, n_init=1, max_iter=1).fit(maps)
	cluster_labels = kmeans_clustering.labels_
	cluster_centers = kmeans_clustering.cluster_centers_
	
	return eigenvectors, cluster_labels, cluster_centers    # CONDENSE THIS
	
	
	
def generate_eigenvector_spreadsheet(algorithm, eigenvectors, cluster_labels, cluster_centers, wordlist, diameter, output_dir, timestamp_string):
	# Now handled through xlsxwriter
	
	# PRELIMINARIES
	
	C = len(wordlist)
	N = eigenvectors.shape[1]         # dimension of wordvector space = number of eigenvectors
	
	eig_rel = np.ones_like(eigenvectors)
	for k in range(N):
		eig_rel[:,k] = eigenvectors[:,k] / eigenvectors[:,0]
	
	
	# USING xlsxwriter
	workbook = xlsxwriter.Workbook(output_dir + algorithm + ".eig_data" + timestamp_string + ".xlsx")
		
	bold = workbook.add_format({'bold': True})
	float_format = workbook.add_format({'num_format':'0.00000000'})
	merge_format = workbook.add_format({'align': 'center', 'bold': True})
	
	worksheet1 = workbook.add_worksheet('Wordlist order')
	worksheet2 = workbook.add_worksheet('Sorted by coord value')
	worksheet3 = workbook.add_worksheet('Sorted by cluster, then coord')
	worksheet4 = workbook.add_worksheet()  # this one is for cluster centers and anything else I record, maybe means as I calculate.
	
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
		
	for n, col in zip(range(1,N), range(N+6, 2*N+5)):
		worksheet1.write(0, col, "Eig" + str(n) + " / Eig0", bold)
	
	# Data
	for c in range(C):
		row = c + 1		
		worksheet1.write(row, 0, c)
		worksheet1.write(row, 1, wordlist[c])
		worksheet1.write(row, 2, cluster_labels[c])
		worksheet1.write(row, 3, diameter[c])
		
		#skip one column  
		for n, col in zip(range(N), range(5,N+5)):
			worksheet1.write(row, col, eigenvectors[c, n], float_format)
			
		#skip one column
		for n, col in zip(range(1,N), range(N+6, 2*N+5)):
			worksheet1.write(row, col, eig_rel[c, n], float_format)  
			# Recall from above that eig_rel[c,n] = eigenvectors[c, n]/eigenvectors[c,0]
			
	##############
	# WORKSHEET2 #
	##############
	## SORT EACH EIGENVECTOR FIRST BY COORD VALUE (INCREASING), THEN BY CLUSTER, USING LEXSORT
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix
	lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.lexsort((diameter, cluster_labels, eigenvectors[:,n]))
	
	for n in range(1,N):
		lex_sortation_rel[:,n] = np.lexsort((diameter, cluster_labels, eig_rel[:,n]))
	
	
	## SORT EACH EIGENVECTOR BY COORD VALUE (INCREASING), USING ARGSORT 
	#arg_sortation = np.argsort(eigenvectors, axis=0, kind='mergesort')    # C x N matrix
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
		
	for n in range(1,N):
		col = 1+ N*5 + (n-1)*5
		worksheet2.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		worksheet2.write(0, col+3, 'Eig'+str(n)+'/Eig0 w/labels', bold)
	
		
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
			worksheet2.write(row, col+2, eigenvectors[indx,n], float_format)
			worksheet2.write(row, col+3, cluster_labels[indx])
			
		for n in range(1,N):
			#indx = arg_sortation_rel[c,n]
			indx = lex_sortation_rel[c,n]
			col = 1 + N*5 + (n-1)*5
			worksheet2.write(row, col, indx)
			worksheet2.write(row, col+1, wordlist[indx])
			worksheet2.write(row, col+2, eig_rel[indx,n], float_format)
			worksheet2.write(row, col+3, cluster_labels[indx])
			
	
	##############
	# WORKSHEET3 #
	##############
	## SORT EACH EIGENVECTOR FIRST BY CLUSTER, THEN BY COORD VALUE (INCREASING), USING LEXSORT
	lex_sortation = np.ones((C,N), dtype=int)        # C x N matrix  
	lex_sortation_rel = np.ones((C,N), dtype=int)    # C x N matrix     disregard leftmost column
	
	for n in range(N):
		lex_sortation[:,n] = np.lexsort((diameter, eigenvectors[:,n], cluster_labels))
		
	for n in range(1,N):
		lex_sortation_rel[:,n] = np.lexsort((diameter, eig_rel[:,n], cluster_labels))
	
	
	# xlsxwriter instructions are similar to those for Worksheet2, above
	worksheet3.set_column(0, (2*N-1)*5-1, 11)   #set column width
	
	## Column headings
	for n in range(N):
		col = 1 + n*5
		worksheet3.merge_range(0, col, 0, col+2, "Eig"+str(n), merge_format)
		worksheet3.write(0, col+3, 'Clusters @Eig'+str(n), bold)
		
	for n in range(1,N):
		col = 1+ N*5 + (n-1)*5
		worksheet3.merge_range(0, col, 0, col+2, "Eig"+str(n)+" / Eig0", merge_format)
		worksheet3.write(0, col+3, 'Clusters @Eig'+str(n)+'/Eig0', bold)
	
	
	## Data
	for c in range(C):
		row = c + 1
		
		for n in range(N):    # for each eigenvector:
			indx = lex_sortation[c,n]
			col = 1 + n*5
			worksheet3.write(row, col, indx)
			worksheet3.write(row, col+1, wordlist[indx])
			worksheet3.write(row, col+2, eigenvectors[indx,n], float_format)
			worksheet3.write(row, col+3, cluster_labels[indx])
			
		for n in range(1,N):
			indx = lex_sortation_rel[c,n]
			col = 1 + N*5 + (n-1)*5
			worksheet3.write(row, col, indx)
			worksheet3.write(row, col+1, wordlist[indx])
			worksheet3.write(row, col+2, eig_rel[indx,n], float_format)
			worksheet3.write(row, col+3, cluster_labels[indx])

			
	##############
	# WORKSHEET4 #
	##############
	## RECORD MEAN OF EACH CLUSTER'S COORD VALUES ON EACH EIGENVECTOR
	cluster_ids, card_per_cluster = np.unique(cluster_labels, return_counts=True)
	num_clusters = len(cluster_ids)
	# BE SURE COUNTS MATCH CATEGORIES IN ORDER   COULD COUNT AT SAME TIME AS SUM TO BE SURE
	coord_sum_per_cluster_per_eigenvector  = np.zeros((num_clusters, N))
	coord_mean_per_cluster_per_eigenvector = np.zeros((num_clusters, N))
	
	for c in range(C):
		for n in range(N):
			
			coord_sum_per_cluster_per_eigenvector[cluster_labels[c],n] = \
				coord_sum_per_cluster_per_eigenvector[cluster_labels[c],n] + eigenvectors[c,n]
				
	for cluster_id in range(num_clusters):
		for n in range(N):
			coord_mean_per_cluster_per_eigenvector[cluster_id,n] = \
				coord_sum_per_cluster_per_eigenvector[cluster_id,n] / card_per_cluster[cluster_id]
				
	#for cluster_id in range(num_clusters):
	#	print("Cluster id =", cluster_id)
	#	for n in range(N):
	#		print(coord_mean_per_cluster_per_eigenvector[cluster_id,n])
			
	## XlsxWriter instructions
	worksheet4.set_column(0, N+1, 11)	#set column width
	worksheet4.write(0, 0, 'Coordinate mean per cluster per eigenvector', bold)
	
	#worksheet4.write(2, 0, 'A.  Computed from cluster output as produced by:   sklearn built-in spectral_clustering()   ' + timestamp_string, bold)
	worksheet4.write(2, 0, 'A.  Computed from cluster output as recorded on Worksheet3   ' + timestamp_string, bold)
	
	## Column headings
	for n, col in zip(range(N), range(1, N+1)):
		worksheet4.write(3, col, 'Eig'+str(n), bold)
		
	## Data 
	for cluster_id, row in zip(range(num_clusters), range(4, 4+num_clusters)):
		worksheet4.write(row, 0, 'Cluster'+str(cluster_id), bold)
		for n, col in zip(range(N), range(1, N+1)):
			worksheet4.write(row, col, coord_mean_per_cluster_per_eigenvector[cluster_id,n], float_format)
	
	#########
	
	row = num_clusters + 5
	worksheet4.write(row, 0, 'B.  Cluster centers returned by KMeans   ' + timestamp_string, bold)
	
	## Column headings
	for n, col in zip(range(N), range(1, N+1)):
		worksheet4.write(row+1, col, 'Eig'+str(n), bold)
		
	## Data 
	for cluster_id, row in zip(range(num_clusters), range(7+num_clusters, 7+2*num_clusters)):
		worksheet4.write(row, 0, 'Cluster'+str(cluster_id), bold)
		for n, col in zip(range(N), range(1, N+1)):
			worksheet4.write(row, col, cluster_centers[cluster_id,n], float_format)

	
	
	workbook.close()
	

# Not in use   October 2018
def eigenvector_data_for_excel(wordlist, eigenvectors, diameter, cluster_labels, dev_output_dirname, timestamp_string, rownorm=False):
	# AVOID EFFECT OF ',' WHEN .csv IS OPENED IN EXCEL. RESTORE ',' UPON RETURN.
	if wordlist.count(',') > 0:
		comma_index = wordlist.index(',')
		wordlist[comma_index] = 'comma_symbol'
		#print("comma_index =", comma_index)
		#print("The entry for comma is now", wordlist[comma_index])

	# OUTPUT TO FILE
	if rownorm == False:
		outfilename = dev_output_dirname + "/eigenvector_data_for_excel." + timestamp_string + ".csv"
	else:
		outfilename = dev_output_dirname + "/rownorm_eigenvector_data_for_excel." + timestamp_string + ".csv"
		
	outfile = open(outfilename, mode='w')
	#print(file=outfile)
	
	C = len(wordlist)
	N = eigenvectors.shape[1]         # dimension of wordvector space = number of eigenvectors
	
	# FIX THIS
	# SORT EACH EIGENVECTOR, carrying along with each element its index and associated word 
	#  eigenvectors[:,n], SORT (c, wordlist[c], eigenvectors[c,n])  Sort list or matrix??
	
	
	ae = np.argsort(eigenvectors, axis=0, kind='mergesort')    # 'ae' for 'argsorted_eigenvectors'
	#ae_rev = np.flipud(ae)					# prefer descending order to match y axis  #CHANGED TO ASCENDING Jan. 25, 2018
	
	e_rel = np.ones_like(eigenvectors)
	for k in range(N):
		e_rel[:,k] = eigenvectors[:,k] / eigenvectors[:,0]
	
	#for j in range(10):
	#	for k in range(N):
	#		print(", %.15f" % e_rel[j,k], end='')
	#	print( )

	ae_rel = np.argsort(e_rel, axis=0, kind='mergesort')
	
	
	print("Index, Wordlist, Cluster, Diameter, Count", end='', file=outfile)
	for n in range(N):
		print(",Eigenvector" + str(n), end='', file=outfile)
	print(',', end='', file=outfile)
	for n in range(1,N):
		print(",", str(n) + "-rel_0", end='', file=outfile)
	print(",,diameter", end='', file=outfile)
	for n in range(N):
		print(",,,,Eig" + str(n) + ",Eig" + str(n) + " Clusters", end='', file=outfile)
	for n in range(1,N):
		print(",,,,Eig" + str(n) + " / Eig0,Clusters", end='', file=outfile)
	#print(",,Labels", file=outfile)
	print(end ="\n", file=outfile)
	
	for c in range(C):
		print('%d, %s, %d, %d' % (c, wordlist[c], cluster_labels[c], diameter[c]), ',', end='', file=outfile)
		for n in range(N):
			print(', %.30f' % eigenvectors[c, n], end='', file=outfile)
		print(',', end='', file=outfile)
		
		for n in range(1,N):
			print(', %.30f' % (eigenvectors[c,n]/eigenvectors[c,0]), end='', file=outfile)
		print(',', end='', file=outfile)
		
		indx = ae[c,0]
		print(',%d,' % (diameter[indx]), end='', file=outfile)
		
		for n in range(N):
			#indx = ae_rev[c,n]	  #CHANGED TO ASCENDING Jan. 25, 2018
			indx = ae[c,n]
			print(',%d, %s, %.30f, %d,' % (indx, wordlist[indx], eigenvectors[indx,n], cluster_labels[indx]), end='', file=outfile) 
			
		for n in range(1,N):
			indx = ae_rel[c,n]
			print(',%d, %s, %.30f, %d,' % (indx, wordlist[indx], e_rel[indx,n], cluster_labels[indx]), end='', file=outfile)
		
		#print(",%d" % cluster_labels[c], end='', file=outfile)
		print(end='\n', file=outfile)
	
	outfile.close()   #NEW  June 30, 2018

	
	# EIGS ORDERED BY Eig1
	
	# ADDED June 28, 2018  in Odense    probably temporary
	outfilename2 = "eigs_ordered_by_Eig1" + timestamp_string + ".csv"
	outfile2 = open(outfilename2, mode='w')
	
	print("Index, Wordlist, Diameter, Count", end='', file=outfile2)
	for n in range(N):
		print(",Eigenvector" + str(n), end='', file=outfile2)
	print(',,,', end='', file=outfile2)
	for n in range(1,N):
		print(",Eig" + str(n), end='', file=outfile2)
	print(end="\n", file=outfile2)
	
	for c in range(C):
		print(c, ',%s,' % wordlist[c], diameter[c], ',', end='', file=outfile2)
		for n in range(0,N):
			print(', %.30f' % eigenvectors[c, n], end='', file=outfile2)   # NEW  June 30, 2018  
		print(',', end='', file=outfile2)
		
		if (N>1):
			indx = ae[c,1]
			print(',%d, %s' % (indx, wordlist[indx]), end='', file=outfile2)
			for n in range(1,N):
				print(', %.30f' % (eigenvectors[indx,n]), end='', file=outfile2)
			print(end='\n', file=outfile2)
		
	outfile2.close()
	wordlist[comma_index] = ','
	

# Not in use
def basic_data_for_excel(wordlist, eigenvectors, atoms, header, alg_label, timestamp): #IF USE, REMEMBER %.30f
	# OUTPUT TO FILE     
	outfilename = "data_for_excel." + alg_label + "." + timestamp.strftime("%Y_%m_%d.%H_%M") + ".csv"
	outfile = open(outfilename, mode='w')
	print(header, file=outfile)
	#print(file=outfile)
	
	C = len(wordlist)
	N = atoms.shape[0]         # dimension of wordvector space = number of eigenvectors
	M = atoms.shape[1]         # number of atoms
	
	comma_string = ",,,"
	for n in range(N):
		comma_string = comma_string+','

	for n in range(N):
		print(comma_string, end='', file=outfile)
		for m in range(M):
			print(',  %.15f' % atoms[n,m], end='', file=outfile)
		print(file=outfile)
	
	print("\nIndex, Wordlist,,", end='', file=outfile)
	for n in range(N):
		print("Eigenvector" + str(n) + ",", end='', file=outfile)
	for m in range(M):
		print(",Atom" + str(m), end='', file=outfile)
	print(file=outfile)
	
	atoms_long_form = eigenvectors @ atoms
	
	for c in range(C):
		#print(c, ',%s \n' % wordlist[c], end='', file=outfile)
		print(c, ',%s,' % wordlist[c], end='', file=outfile)
		for n in range(N):
			print(', %.15f' % eigenvectors[c, n], end='', file=outfile)
		print(',', end='', file=outfile)	
		for m in range(M):
			print(', %.15f' % atoms_long_form[c,m], end='', file=outfile)
		print(file=outfile)
		
		
		
	
	
	outfile.close()
	

#def decomposition_ica(data_array, wordlist, timestamp):
	#outfilename = "decomp_ica_info.sk_dl." + timestamp.strftime("%Y_%m_%d.%H_%M") + ".csv"
	#outfile = open(outfilename, mode='w')
	
#def sparse_repr(data_array, wordlist, timestamp):
	##sparse_repr_sk_dl(data_array, wordlist, timestamp)
	#decomposition_ica(data_array, wordlist, timestamp)
	
	





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

    #print(wordlist)

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
    
    
    
    laplacian, sqrt_diam = csgraph_laplacian(shared_context_matrix, normed=True, return_diag=True)   # returns L_sym
    #laplacian, sqrt_diam = Chicago_get_laplacian(shared_context_matrix)
    diameter = np.square(sqrt_diam)
    
    eigenvalues, eigenvectors = linalg.eigsh(laplacian, k=n_eigenvectors, which='SM')  #N.B. eigs complex; eigsh not
    eigenvectors = (_deterministic_vector_sign_flip(eigenvectors.T)).T                 #standardization convention; no other effect
    
    # Alternative algorithms   ref. Ulrike von Luxburg. A Tutorial on Spectral Clustering. 
    # These alternatives are organized as separate functions for clarity and ease of separate modification.
    wordcoords_sym, clusterlabels_sym, clustercenters_sym = spectral_clustering_sym(eigenvectors, random_seed=1)
    generate_eigenvector_spreadsheet('sym', wordcoords_sym, clusterlabels_sym, clustercenters_sym,
                                     wordlist, diameter, output_dir, timestamp_string)
    eigenvector_clusterwise_plots('sym', wordcoords_sym, clusterlabels_sym, diameter, output_dir, timestamp_string)
    
    wordcoords_rw, clusterlabels_rw, clustercenters_rw = spectral_clustering_rw(eigenvectors, sqrt_diam, random_seed=1)
    generate_eigenvector_spreadsheet('rw', wordcoords_rw, clusterlabels_rw, clustercenters_rw,
                                     wordlist, diameter, output_dir, timestamp_string)
    eigenvector_clusterwise_plots('rw', wordcoords_rw, clusterlabels_rw, diameter, output_dir, timestamp_string)                                 
    
    raise SystemExit    # October 26, 2018
    
    ### OMIT
    eigenvectors_sym = eigenvectors   # Does this preserve eigenvectors for Jackson's use later? Consider options.
    eigenvectors_rw  = skl_normalize( (eigenvectors_sym / sqrtdiam_array[:, np.newaxis]), norm='l2', axis=0 )
    
    
    # In eigenvectors matrix, 
    #  each column is a unit-length eigenvector of laplacian (L_sym or L_rw)
    #  each row represents a word as a tuple in that coordinate system 
    
    samples_to_cluster_sym = skl_normalize(eigenvectors_sym, norm='l2', axis=1)		# row => unit-length  
    samples_to_cluster_rw  = eigenvectors_rw
    
    
    #Keep going. Test as soon as possible.
    
    ### OMIT
    # May try with sparse affinity matrix  scm_csr, since there was an issue needing np.asarrau
    eigenvectors, sqrtdiam_array = sk_lifted_spectral_embedding(np.asarray(shared_context_matrix), n_eigenvectors)
    #eigenvectors, sqrtdiam_array = Chicago_spectral_embedding(np.asarray(shared_context_matrix), n_eigenvectors)
    
    ### OMIT
    diameter = compute_diameter_array(n_words, shared_context_matrix)		# moved up to here  Oct. 6, 2018
    
    sk_eigenvectors, sk_cluster_labels, sk_cluster_centers = sk_lifted_spectral_clustering(np.asarray(shared_context_matrix), 12, 12, 1)  # affinity_matrix, num_eigs, num_clusters, random seed for kmeans
    generate_eigenvector_spreadsheet('sklearn', wordlist, sk_eigenvectors, diameter, sk_cluster_labels, sk_cluster_centers, dev_output_dirname, timestamp_string, rownorm=False)
	
	
    # computing diameter
    #diameter = normalize(n_words, shared_context_matrix)   # diameter is an array   1000 x 1
    #diameter = compute_diameter_array(n_words, shared_context_matrix)   # diameter is an array   1000 x 1   Moved upward Oct. 6, 2018
    #print("\ndiameter:")
    #print(diameter)   # audrey  2016_12_13

    # computing incidence graph
    incidence_graph = compute_incidence_graph(n_words, diameter,
                                              shared_context_matrix)
    del shared_context_matrix

    # computing laplacian matrix
    laplacian_matrix = compute_laplacian(diameter, incidence_graph)
    # del diameter   # audrey  2016_12_13   keep it to pass to make_heatmap  Hmmm-not involved in heatmap  
    del incidence_graph
    
    # NOT SUCCESSFUL    Ran many hours; eventually terminated it.
    #Eigenvalue_list = sym_compute_eigenvalues(laplacian_matrix, n_eigenvectors)
    #print("Eigenvalue_list = ", Eigenvalue_list)
    
    # PRODUCE LOW_DIM DENSE WORDVECTORS
    # computing eigenvectors and eigenvalues
    eigenvalues, eigenvectors = compute_eigenvectors(laplacian_matrix, n_eigenvectors)    # AUDREY  added n_eigenvectors argument! 
    # Prevent ComplexWarning messages
    eigenvalues = eigenvalues.real			# AUDREY  Could be handled within compute_eigenvectors by using eigsh instead of eigs.
    eigenvectors = eigenvectors.real        # However, eigsh returns eigenvalues (and resulting eigenvectors) in ascending order, so would require sort(). 
    
    # MODIFY L_sym EIGENVECTORS TO GET L_rw EIGENVECTORS
    sqrt_diameter = np.sqrt(diameter)
    eigenvectors = eigenvectors / sqrt_diameter[:, np.newaxis]
    eigenvectors = skl_normalize(eigenvectors, norm='l2', axis=0)
    
    
    # TEMPORARY  APRIL 27, 2018
    #print("\n\nTEST THE INITIAL EIGENVALUE (because of concern over largest vs. smallest)")
    #print("product of laplacian_matrix and 0th column of eigenvectors, initial entries")
    #Check0thColumn = laplacian_matrix @ eigenvectors[:,0]
    #print(Check0thColumn[0:11])
        
    #print("\nCONFIRM THAT  sqrt(diameter)/norm   MATCHES Eig0 ON SPREADSHEET  (unit vector for eigenvalue 0)")
    #sqrt_diam = np.sqrt(diameter)
    #print("sqrt of diameter, first 10 entries:", sqrt_diam[:10])    
    #print("norm of sqrt of diameter =", np.linalg.norm(sqrt_diam)) 
    
    del laplacian_matrix
    
    
    ## SECOND STANDARDIZATION CODE   replaced by _deterministic_vector_sign_flip 2018_10_22 (below)
    #print("\nSTANDARDIZE EIGENVECTORS wrt SIGN")
    #print("eigenvalues:", eigenvalues)
    #sum_vec  = np.sum(eigenvectors, axis=0)
    #print("sum_vec (initially):", sum_vec)
    #for i in range(len(sum_vec)):
    #	if sum_vec[i] < 0:
    #		eigenvectors[:,i] = -1 * eigenvectors[:,i]
    #sum_vec  = np.sum(eigenvectors, axis=0)
    #print("sum_vec (after standardizing):", sum_vec)
	#
    #test_ind = wordlist.index('the')
    #test_vec = eigenvectors[test_ind, :]
    ##print("eigenvalues:", eigenvalues)		# SHOW onscreen
    #print("test_vec (resulting coords for 'the'):", test_vec)		# standardize to  (positive, negative) coordinates for 'the'
    #print()
    
    # FIRST STANDARDIZATION CODE
    #if (test_vec[0] < 0):
    #	eigenvectors[:,0] = -1 * eigenvectors[:,0]
    #if (test_vec[1] > 0):
    #	eigenvectors[:,1] = -1 * eigenvectors[:,1]
    #if (len(test_vec)>2):
    #	if (test_vec[2] < 0):
    #		eigenvectors[:,2] = -1 * eigenvectors[:,2]
    
    #print("dot product of Eig0 and Eig1 =", np.dot(eigenvectors[:,0], eigenvectors[:,1]))
    
    
    # take first N columns of eigenvector matrix
    # coordinates = eigenvectors[:, : n_eigenvectors]		# AUDREY  eigenvectors matrix has n_eigenvectors columns by construction
    # print("\ncoordinates")   # audrey  2016_12_05
    # print(coordinates)       # audrey  2016_12_05
    
    # USED THIS TO CHECK WHEN linalg.eig PRODUCES 6 EIGENVECTORS (i.e., default behavior) AND coordinates IS SUPPOSEDLY TRUNCATED TO 11. 
    # ANSWER IS THAT THE TWO ARRAYS ARE THE SAME; BOTH ARE 1000 X 6
    #if (eigenvectors == coordinates).all():    #if np.array_equal(eigenvectors, coordinates):
    #	print("YES! eigenvectors and coordinates are same matrix", file=outfile)
    #outfile.close()
    
    
    # MAY 24, 2018   audrey   ROWNORM (for L_sym)                    # MINIMAL CODE CHANGES FOR FIRST TEST
    rownorm_eigenvectors = skl_normalize(eigenvectors, norm='l2') 	 # makes each ROW unit length # should I use csr??  Or not use skl??
    ### eigenvectors = rownorm_eigenvectors       NO !!              # FOR NOW -- AT LEAST FIRST TEST. SPARSE??
    #print("\nEigenvector matrix row_normalized:\n")
    #print(rownorm_eigenvectors[0:5, :])
    #print("Row0 norm of input is ", np.linalg.norm(eigenvectors[0, :]))
    #print("Row1 norm of input is ", np.linalg.norm(eigenvectors[1, :]))
    
    
    # STANDARDIZE DIRECTION apply this to the eigenvectors in whatever form is about to be submitted to clustering algorithm
    eigenvectors = (_deterministic_vector_sign_flip(eigenvectors.T)).T
    rownorm_eigenvectors = (_deterministic_vector_sign_flip(rownorm_eigenvectors.T)).T
    
    # SEPTEMBER 26, 2018  audrey  Use k-means to cluster rows    (for Chicago spreadsheet)
    kmeans_clustering = sklearn.cluster.KMeans(n_clusters=12, random_state=1).fit(eigenvectors)
    #kmeans_clustering = sklearn.cluster.KMeans(n_clusters=12, random_state=1, n_init=1, max_iter=1).fit(eigenvectors)  # SIMPLEST
    cluster_labels = kmeans_clustering.labels_
    cluster_centers = kmeans_clustering.cluster_centers_
    #_, cluster_labels, _ = k_means(rownorm_eigenvectors, n_clusters=6, random_state=1)
    rownorm_kmeans_clustering = sklearn.cluster.KMeans(n_clusters=12, random_state=1).fit(rownorm_eigenvectors)
    #rownorm_kmeans_clustering = sklearn.cluster.KMeans(n_clusters=12, random_state=1, n_init=1, max_iter=1).fit(rownorm_eigenvectors)    # SIMPLEST
    rownorm_cluster_labels = rownorm_kmeans_clustering.labels_
    rownorm_cluster_centers = rownorm_kmeans_clustering.cluster_centers_
    for i in range(15):
    	print(rownorm_cluster_labels[i])
    
    
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
        
    #timestamp = datetime.datetime.now()
    #timestamp_string = timestamp.strftime(".%Y_%m_%d.%H_%M")
    
    # make_heatmap(wordlist, words_to_neighbors, words_to_neighbor_distances, timestamp)  #SKIP THIS WHEN NOT NEEDED 
    #make_wordvector_plots(eigenvalues, eigenvectors, wordlist, timestamp)   #SKIPPING  --  August 2018
    
    ## RESOLVE
    ##sparse_repr(eigenvectors, wordlist, timestamp)   #WHAT SHAPE SHOULD THIS BE?  WHAT LENGTH? NEED TRANSLATION FROM INDICES?
    #atoms, codes, header, alg_label = learn_sparse_repr(eigenvectors)
    #study_decomposition(wordlist, eigenvectors, atoms, codes, header, alg_label, timestamp)		
    ##investigate_atoms(wordlist, eigenvectors, atoms)
    #basic_data_for_excel(wordlist, eigenvectors, atoms, header, alg_label, timestamp)
    
    #eigenvector_data_for_excel(wordlist, eigenvectors, diameter, cluster_labels, dev_output_dirname, timestamp_string, rownorm=False)	# May want these two also for atoms  
    generate_eigenvector_spreadsheet('Chicago', wordlist, eigenvectors, diameter, cluster_labels, cluster_centers, dev_output_dirname, timestamp_string, rownorm=False)
    eigenvector_plots(wordlist, eigenvectors, timestamp_string)
    eig0_diameter_ascending_plots(sorted(eigenvectors[:,0]), sorted(diameter), timestamp_string)
    generate_eigenvector_spreadsheet('Chicago', wordlist, rownorm_eigenvectors, diameter, rownorm_cluster_labels, rownorm_cluster_centers, dev_output_dirname, timestamp_string, rownorm=True)
    #eigenvector_data_for_excel(wordlist, rownorm_eigenvectors, diameter, cluster_labels, cluster_centers, dev_output_dirname, timestamp_string, rownorm=True)
    
    #outfilename = 'count.csv'
    #outfile = open(outfilename, mode='w')
    #for pr in word_freq_pairs:
    #	print(pr[1], file=outfile)
    #outfile.close()
    
    #print(words_to_contexts['the'])
    #print(words_to_contexts['and'])
    #print(words_to_contexts['that'])
    
    return words_to_neighbors, words_to_contexts, contexts_to_words
    
    
