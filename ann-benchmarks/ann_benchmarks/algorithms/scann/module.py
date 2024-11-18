import numpy as np
import scann
import threading
from ..base.module import BaseANN

class Scann(BaseANN):
    def __init__(self, n_leaves, avq_threshold, dims_per_block, dist):
        self.name = "scann n_leaves={} avq_threshold={:.02f} dims_per_block={}".format(
            n_leaves, avq_threshold, dims_per_block
        )
        self.n_leaves = n_leaves
        self.avq_threshold = avq_threshold
        self.dims_per_block = dims_per_block
        self.dist = dist
        self.batch_size = 256
        self.num_threads = 320

    def fit(self, X):
        if self.dist == "dot_product":
            spherical = True
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        else:
            spherical = False

        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, self.dist)
            .tree(self.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True)
            .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)
            .reorder(1)
            .build()
        )

    def set_query_arguments(self, query_args):
        if len(query_args) == 5:
            self.leaves_to_search, self.reorder, self.thd, self.refined, self.batch_size = query_args
        elif len(query_args) == 6:
            self.leaves_to_search, self.reorder, self.thd, self.refined, self.batch_size, self.num_threads = query_args
        else :
            self.leaves_to_search, self.reorder, self.thd, self.refined = query_args

    def query(self, v, n):
        # print("Current thread ID:", threading.current_thread().ident)
        return self.searcher.search(v, n, self.reorder, self.leaves_to_search)[0]

    def batch_query(self, v, n):
        if self.dist == "dot_product":
            v[np.linalg.norm(v, axis=1) == 0] = 1.0 / np.sqrt(v.shape[1])
            v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
        self.searcher.search_additional_params(self.thd, self.refined, self.leaves_to_search)
        if (self.num_threads != 320) and (self.num_threads >= 1):
            self.searcher.set_num_threads(self.num_threads-1)
        self.res = self.searcher.search_batched_parallel(v, n, self.reorder, self.leaves_to_search, self.batch_size)[0]
    