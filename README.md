# Mapping semantic spaces for translation

In this tutorial, you will map from a small English semantic space to a Catalan semantic space. You may need a Catalan dictionary for the following exercises. Here's a good one: *https://www.diccionaris.cat/*.


### Pen and paper exercise

The following are two toy semantic spaces, one for English, one for Catalan. Rows represent vectors, columns represent dimensions.

<table>
<tr><td></td><td>nature</td><td>argue</td></tr>
<tr><td>horse</td><td>0.3</td><td>0.0</td></tr>
<tr><td>dog</td><td>0.3</td><td>0.0</td></tr>
<tr><td>house</td><td>0.2</td><td>0.1</td></tr>
<tr><td>parliament</td><td>0.0</td><td>0.7</td></tr>
<tr><td>politics</td><td>0.1</td><td>0.9</td></tr>
<tr><td>right</td><td>0.1</td><td>0.6</td></tr>
<tr><td>wrong</td><td>0.1</td><td>0.7</td></tr>
</table>

<table>
<tr><td></td><td>lluitar</td><td>arbre</td></tr>
<tr><td>cavall</td><td>0.1</td><td>0.3</td></tr>
<tr><td>gos</td><td>0.1</td><td>0.2</td></tr>
<tr><td>casa</td><td>0.0</td><td>0.2</td></tr>
<tr><td>parlament</td><td>0.5</td><td>0.0</td></tr>
<tr><td>política</td><td>0.6</td><td>0.0</td></tr>
<tr><td>correcte</td><td>0.4</td><td>0.0</td></tr>
<tr><td>equivocat</td><td>0.5</td><td>0.0</td></tr>
</table>

Now, you get a new vector in English, say:

<table>
<tr><td></td><td>nature</td><td>argue</td></tr>
<tr><td>green</td><td>0.6</td><td>0.1</td></tr>
</table>

Which of those two Catalan words do you think is the translation of *green* according to your semantic spaces? Why? NB: you don't have to know any calculus to solve this by hand.

<table>
<tr><td></td><td>lluitar</td><td>arbre</td></tr>
<tr><td>verd</td><td>0.1</td><td>0.5</td></tr>
<tr><td>vermell</td><td>0.2</td><td>0.1</td></tr>
</table>


### Running the visualisation code

Running the following will create pictures of the English and Catalan spaces in your directory.

    python3 viz.py


You can also visualise the neighbourhood of a specific word by doing e.g.

    python3 viz.py en bird 20

(This will give you a graph of the 20 nearest neighbours of *bird* in the English space.)

For Catalan, you can similarly do:

    python3 viz.py cat ocell 20


### Preliminaries to running the code

Familiarise yourself with the content of the data/ directory. The *pairs.txt* file contains gold standard translations from English to Catalan. *english.subset.dm* and *catalan.subset.dm* are subsets of an English and a Catalan semantic space corresponding to the words occurring in *pairs.txt*.

There are 166 pairs and we will be splitting the data into 120 pairs for training and 46 for testing. You can look at the test pairs by doing 

    tail -46 data/pairs.txt

Just looking at the data and the associated visualisation (before running anything), can you tell where the model might do well and where it might fail?


### Running the regression code

Running the code will split the data into training and test set, calculate the regression matrix on the training data and evaluate it on the test set:

    python3 regression.py

The output first gives the predictions for each pair. For instance, it could be:

    bird ocell ['arbre', 'peix', 'ocell', 'gos', 'animal'] 1

Here, *bird* should have been translated with *ocell*. The 5 nearest neighbours of the predicted vector are *arbre, peix, ocell, gos* and *animal*, meaning the gold translation can be found in those close neighbours.

The last line gives the precision @ *k*, where *k* is the number of nearest neighbours considered for evaluation.

What can you say about the system's errors? Do they confirm your hypotheses?


### All too easy?

There is a small Italian semantic space in the data/ folder, with 1000 frequent Italian words. You can try and build the regression for a new train/test set for Italian!






