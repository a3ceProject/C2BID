from functools import reduce

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rrcf import RCTree  # https://klabum.github.io/rrcf/


def train_tree(ixs, X):
    '''
        Add sampled trees to forest
    :param ixs: set of samples to be considered
    :type ixs:  list
    :param X: dataset
    :type X: numpy
    :return: tree
    :rtype: list
    '''
    return [RCTree(X[ix], index_labels=ix)
            for ix in ixs]

class Rrcf():
    '''
    adaptado de  https://klabum.github.io/rrcf/
    Applies Robust Random Cut Forest Algorithm aogorithm for outliers detection
    '''
    def __init__(self,num_trees=100,tree_size=128,perc=99, n_jobs=1):
        '''

        :param num_trees: trees number
        :type num_trees:int
        :param tree_size: tree size
        :type tree_size: int
        :param perc: percentage of non outliers
        :type perc: float
        :param n_jobs: parallel runing 
        :type n_jobs: int
        '''
        self.num_trees=num_trees
        self.tree_size=tree_size
        self.n_jobs=n_jobs
        self.perc=perc
        self.forest=[]

    def fit_predict(self, X=None):
        '''
        Does the training and prediction of the model
        :param X: dataset
        :type: DataFrame
        :return: classification as outlier (-1) or not outlier (2)
        :rtype: list[int]
        '''
        X = X.to_numpy()
        n=X.shape[0]
        sample_size_range = (n // self.tree_size, self.tree_size)
        if n // self.tree_size==0:
            sample_size_range=(1,self.tree_size)
        IXS=[]
        while len(IXS) < self.num_trees:
            # Select random subsets of points uniforml
                # Select random subsets of points uniformly from point set
                IXS.append(np.random.choice(n, size=sample_size_range,
                                       replace=False))

        trees=Parallel(n_jobs=self.n_jobs)(
            delayed(train_tree)(ixs,X) for ixs in IXS)
        trees=reduce(lambda a, b: a + b,trees)

        self.forest.extend(trees)

        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(n), )
        index = np.zeros(n)

        for tree in self.forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index


        avg_cod = avg_codisp
        mask = np.percentile(avg_cod, self.perc)
        avg_cod[avg_cod < mask] = 2
        avg_cod[avg_cod > mask] = -1
        return avg_cod.tolist()
