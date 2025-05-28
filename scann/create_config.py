 
def _factory_decorator(key):
  """Wraps a function that produces a portion of the ScaNN config proto."""
 
  def func_taker(f):
    """Captures arguments to function and saves them to params for later."""
 
    def inner(self, *args, **kwargs):
      if key in self.params:
        raise Exception(f"{key} has already been configured")
      kwargs.update(zip(f.__code__.co_varnames[1:], args))
      self.params[key] = kwargs
      return self
 
    inner.proto_maker = f
    return inner
 
  return func_taker
 
 
 
class ScannConfiger(object):
  """Builder class."""
 
  def __init__(self, ndims, num_neighbors, distance_measure):
    self.params = {}
    self.training_threads = 0
    self.builder_lambda = None
    self.ndims = ndims
    self.num_neighbors = num_neighbors
    self.distance_measure = distance_measure
 
  def set_n_training_threads(self, threads):
    self.training_threads = threads
    return self
 
  def set_builder_lambda(self, builder_lambda):
    """Sets builder_lambda, which creates a ScaNN searcher upon calling build().
 
    Args:
      builder_lambda: a function that takes a dataset, ScaNN config text proto,
        number of training threads, and optional kwargs as arguments, and
        returns a ScaNN searcher.
    Returns:
      The builder object itself, as expected from the builder pattern.
    """
    self.builder_lambda = builder_lambda
    return self

  @_factory_decorator("tree")
  def tree(
      self,
      num_leaves,
      num_leaves_to_search,
      training_sample_size=100000,
      min_partition_size=50,
      training_iterations=12,
      spherical=False,
      quantize_centroids=False,
      random_init=True,
      soar_lambda=None,
      overretrieve_factor=None,
      # the following are set automatically
      distance_measure=None,
  ):
    """Configure partitioning. If not called, no partitioning is performed."""
    soar_stanza = ""
    if soar_lambda is not None:
      if self.distance_measure != "dot_product":
        raise ValueError("SOAR requires dot product distance.")
      overretrieve_factor_stanza = (
          f"overretrieve_factor: {overretrieve_factor}"
          if overretrieve_factor is not None else "")
      soar_stanza = f"""database_spilling {{
        spilling_type: TWO_CENTER_ORTHOGONALITY_AMPLIFIED
        orthogonality_amplification_lambda: {soar_lambda}
        {overretrieve_factor_stanza}
      }}"""

    return f"""
      partitioning {{
        num_children: {num_leaves}
        min_cluster_size: {min_partition_size}
        max_clustering_iterations: {training_iterations}
        single_machine_center_initialization: {
            "RANDOM_INITIALIZATION" if random_init else "DEFAULT_KMEANS_PLUS_PLUS"
        }
        partitioning_distance {{
          distance_measure: "SquaredL2Distance"
        }}
        query_spilling {{
          spilling_type: FIXED_NUMBER_OF_CENTERS
          max_spill_centers: {num_leaves_to_search}
        }}
        expected_sample_size: {training_sample_size}
        query_tokenization_distance_override {distance_measure}
        partitioning_type: {"SPHERICAL" if spherical else "GENERIC"}
        query_tokenization_type: {"FIXED_POINT_INT8" if quantize_centroids else "FLOAT"}
        {soar_stanza}
      }}
    """

  @_factory_decorator("score_ah")
  def score_ah(
      self,
      dimensions_per_block,
      anisotropic_quantization_threshold=float("nan"),
      training_sample_size=100000,
      min_cluster_size=100,
      hash_type="lut16",
      training_iterations=10,
      # the following are set automatically
      residual_quantization=None,
      n_dims=None):
    """Configure asymmetric hashing. Must call this or score_brute_force."""
    del min_cluster_size  # Deprecated field.
    hash_types = ["lut16", "lut256"]
    if hash_type == hash_types[0]:
      clusters_per_block = 16
      lookup_type = "INT8_LUT16"
    elif hash_type == hash_types[1]:
      clusters_per_block = 256
      lookup_type = "INT8"
    else:
      raise ValueError(f"hash_type must be one of {hash_types}")
    full_blocks, partial_block_dims = divmod(n_dims, dimensions_per_block)
    if partial_block_dims == 0:
      proj_config = f"""
        projection_type: CHUNK
        num_blocks: {full_blocks}
        num_dims_per_block: {dimensions_per_block}
      """
    else:
      proj_config = f"""
        projection_type: VARIABLE_CHUNK
        variable_blocks {{
          num_blocks: {full_blocks}
          num_dims_per_block: {dimensions_per_block}
        }}
        variable_blocks {{
          num_blocks: 1
          num_dims_per_block: {partial_block_dims}
        }}
      """
    # global top-N requires LUT16, int16 accumulators, and residual quantization
    global_topn = (
        hash_type == hash_types[0] and
        (full_blocks + (partial_block_dims > 0)) <= 256 and
        residual_quantization)
    return f"""
      hash {{
        asymmetric_hash {{
          lookup_type: {lookup_type}
          use_residual_quantization: {residual_quantization}
          use_global_topn: {global_topn}
          quantization_distance {{
            distance_measure: "SquaredL2Distance"
          }}
          num_clusters_per_block: {clusters_per_block}
          projection {{
            input_dim: {n_dims}
            {proj_config}
          }}
          noise_shaping_threshold: {anisotropic_quantization_threshold}
          expected_sample_size: {training_sample_size}
          max_clustering_iterations: {training_iterations}
        }}
      }} """
 
  @_factory_decorator("score_bf")
  def score_brute_force(self, quantize=False):
    return f"""
      brute_force {{
        fixed_point {{
          enabled: {quantize}
        }}
      }}
    """
 
  @_factory_decorator("reorder")
  def reorder(self, reordering_num_neighbors, quantize=False):
    return f"""
      exact_reordering {{
        approx_num_neighbors: {reordering_num_neighbors}
        fixed_point {{
          enabled: {quantize}
        }}
      }}
    """
 
  def create_config(self):
    """Returns a text ScaNN config matching the specification in self.params."""
    allowed_measures = {
        "dot_product": '{distance_measure: "DotProductDistance"}',
        "squared_l2": '{distance_measure: "SquaredL2Distance"}',
    }
    distance_measure = allowed_measures.get(self.distance_measure)
    if distance_measure is None:
      raise ValueError(
          f"distance_measure must be one of {list(allowed_measures.keys())}")
    config = f"""
      num_neighbors: {self.num_neighbors}
      distance_measure {distance_measure}
    """
 
    tree_params = self.params.get("tree")
    if tree_params is not None:
      tree_params["distance_measure"] = distance_measure
      config += self.tree.proto_maker(self, **tree_params)
 
    ah = self.params.get("score_ah")
    bf = self.params.get("score_bf")
    if ah is not None and bf is None:
      ah["residual_quantization"] = tree_params is not None and self.distance_measure == "dot_product"
      ah["n_dims"] = self.ndims
      config += self.score_ah.proto_maker(self, **ah)
    elif bf is not None and ah is None:
      config += self.score_brute_force.proto_maker(self, **bf)
    else:
      raise Exception(
          "Exactly one of score_ah or score_brute_force must be used")
 
    reorder_params = self.params.get("reorder")
    if reorder_params is not None:
      config += self.reorder.proto_maker(self, **reorder_params)
    return config
 
import sys
if __name__ == "__main__":
    n_leaves = int(sys.argv[1])
    train_size = int(sys.argv[2])
    spherical = True
    dims_per_block = int(sys.argv[4]) # 2
    avq_threshold = float(sys.argv[5]) # 0.2
    dim = int(sys.argv[6]) # 100
    topK = int(sys.argv[7]) # 10
    soar_lambda = float(sys.argv[8]) # 10
    overretrieve_factor = float(sys.argv[9]) # 10
    if (soar_lambda == -1):
      soar_lambda = None
    if (overretrieve_factor == -1):
      overretrieve_factor = None
    metric_str = "dot_product"
    if (sys.argv[3] == "squared_l2"):
        spherical = False
        metric_str = "squared_l2"
    config = ScannConfiger(dim, topK, metric_str) \
    .tree(n_leaves, 1, training_sample_size=train_size, spherical=spherical, quantize_centroids=True, soar_lambda=soar_lambda, overretrieve_factor=overretrieve_factor) \
    .score_ah(dims_per_block, anisotropic_quantization_threshold=avq_threshold) \
    .reorder(1).create_config()
    print(config)