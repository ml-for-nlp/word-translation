"""Word translation

Usage:
  plsr_regression.py --ncomps=<n> --nns=<n> [-v | --verbose]
  plsr_regression.py (-h | --help)
  plsr_regression.py --version

Options:
  --ncomps=<n>   Number of principal components
  --nns=<n>      Number of nearest neighbours for the evaluation
  -h --help      Show this screen.
  --version      Show version.
  -v --verbose   Show verbose output.

"""

from docopt import docopt
import numpy as np
import utils
from sklearn.cross_decomposition import PLSRegression


def mk_training_matrices(pairs, en_dimension, cat_dimension, english_space, catalan_space):
    en_mat = np.zeros((len(pairs),en_dimension)) 
    cat_mat = np.zeros((len(pairs),cat_dimension))
    c = 0
    for p in pairs:
        en_word,cat_word = p.split()
        en_mat[c] = english_space[en_word]   
        cat_mat[c] = catalan_space[cat_word]   
        c+=1
    return en_mat,cat_mat


def PLSR(mat_english,mat_catalan,ncomps):
    plsr = PLSRegression(n_components=ncomps)
    plsr.fit(mat_english,mat_catalan)
    return plsr 


if __name__ == '__main__':
    args = docopt(__doc__, version='PLSR regression for word translation 1.1')
    ncomps = int(args["--ncomps"])
    nns = int(args["--nns"])
    verbose = False
    if args["--verbose"]:
        verbose = True

    '''Read semantic spaces'''
    english_space = utils.readDM("data/english.subset.dm")
    catalan_space = utils.readDM("data/catalan.subset.dm")

    '''Read all word pairs'''
    all_pairs = []
    f = open("data/pairs.txt")
    for l in f:
        l = l.rstrip('\n')
        all_pairs.append(l)
    f.close()

    '''Make training/test fold'''
    training_pairs = all_pairs[:120]
    test_pairs = all_pairs[121:]

    '''Make training/test matrices and get PLSR model'''
    en_mat, cat_mat = mk_training_matrices(training_pairs, 400, 300, english_space, catalan_space)
    plsr = PLSR(en_mat,cat_mat,ncomps)

    ''' Predict with PLSR'''
    score = 0
    for p in test_pairs:
        en, cat = p.split()
        predicted_vector = plsr.predict(english_space[en].reshape(1,-1))[0]
        #print(predicted_vector[:20])
        nearest_neighbours = utils.neighbours(catalan_space,predicted_vector,nns)
        if cat in nearest_neighbours:
            score+=1
            if verbose:
                print(en,cat,nearest_neighbours,"1")
        else:
            if verbose:
                print(en,cat,nearest_neighbours,"0")

    print("Precision PLSR:",score/len(test_pairs))
