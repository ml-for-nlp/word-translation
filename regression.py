import numpy as np
import utils


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


def linalg(mat_english,mat_catalan):
    w = np.linalg.lstsq(mat_english,mat_catalan)[0] # obtaining the parameters
    print(mat_english.shape,mat_catalan.shape,w.shape)
    return w





'''Read semantic spaces'''
english_space = utils.readDM("data/english.subset.dm")
catalan_space = utils.readDM("data/catalan.subset.dm")
utils.run_PCA(english_space,english_space.keys(),"english_space.png")
utils.run_PCA(catalan_space,catalan_space.keys(),"catalan_space.png")

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

'''Make training/test matrices'''
en_mat, cat_mat = mk_training_matrices(training_pairs, 400, 300, english_space, catalan_space)
params = linalg(en_mat,cat_mat)

'''Test'''

'''Sanity check -- is the regression matrix retrieving the training vectors?'''
#print(training_pairs[0])
#en, cat = training_pairs[0].split()
#predict = np.dot(params.T,english_space[en])
#print(predict[:20])
#print(catalan_space[cat][:20])

'''Loop through test pairs and evaluate translations'''
score = 0
for p in test_pairs:
    en, cat = p.split()
    predicted_vector = np.dot(params.T,english_space[en])
    #print(predicted_vector)
    nearest_neighbours = utils.neighbours(catalan_space,predicted_vector,5)
    if cat in nearest_neighbours:
        score+=1
        print(en,cat,nearest_neighbours,"1")
    else:
        print(en,cat,nearest_neighbours,"0")

print("Precision:",score/len(test_pairs))

