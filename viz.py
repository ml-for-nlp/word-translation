import numpy as np
import utils
import sys

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

utils.run_PCA(english_space,[p.split()[0] for p in all_pairs],"english_space.png")
utils.run_PCA(catalan_space,[p.split()[1] for p in all_pairs],"catalan_space.png")

if len(sys.argv) == 4:
    lang = sys.argv[1]
    word = sys.argv[2]
    num_neighbours = int(sys.argv[3])
    if lang == "en":
        english_neighbours = utils.neighbours(english_space,english_space[word],num_neighbours)
        utils.run_PCA(english_space,[word]+english_neighbours,"english_neighbours.png")
    if lang == "cat":
        catalan_neighbours = utils.neighbours(catalan_space,catalan_space[word],num_neighbours)
        utils.run_PCA(catalan_space,[word]+catalan_neighbours,"catalan_neighbours.png")
