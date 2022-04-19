.. customization:

Custom Models
=============

The previous sections described how to used some of the pre-defined models in the
``morphoclass`` package. These are the models that have proven to have a very good
performance at least in some cases, and therefore we provide a high-level interface
for their training and evaluation.

The ``morphoclass`` package provides additional pre-defined models, as well as many
tools for the creation of new graph based models that operate on morphological data.
Unlike for the previous models, there is no high-level interface, and many steps,
including the instantiation, training loop, and evaluation, need to be done in a
manual fashion. The goal of this section is to provide an entry point for the users
who wish to experiment with new models.

Layers
------
Layers are the building blocks of deep-learning models. The ``morphoclass`` package
provides a number of pre-define layers that can be readily used and can be found
in the ``morphoclass.layers`` module.

Graph Convolution Layers
........................
The following graph-convolution layers are pre-defined:

- ``ChebConv`` -- the classic graph convolutional layer based on Chebyshev polynomials
  that was introduced in `arXiv:1606.09375 <https://arxiv.org/abs/1606.09375>`__. The
  implementation is based on that in the ``torch_geometric`` package. See the API
  documentation for more details.
- ``ChebConvSeparable`` -- a version of ``ChebConv`` with two additions:

  - Depth-separable convolutions. Known for example from the
    `MobileNet <https://arxiv.org/abs/1704.04861>`__ they split the usual convolution
    into a space-wise and a depth-wise convolution.
  - More control in the specification of the orders of Chebyshev polynomials. The
    ``ChebConv`` use polynomials up to the fixed order ``K``, while here we allow
    to provide a list of polynomial orders, which allows to skip some of them,
    e.g. ``[1, 3, 5]``.

  Both additions aim to reduce the number of parameters by making the convolutions "more sparse".

- ``BidirectionalBlock`` -- the layer used in ``ManNet`` that was described in the :ref:`gnn`
  section. It only makes sense on directed graphs (where the adjacency matrix is not symmetric)
  and performs two ChebConvs in parallel, one of them with the reversed edge orientation. The
  result of both convolutions is then concatenated channel-wise.
- ``BidirectionalResBlock`` -- a residual block with the structure similar to those used in
  `ResNets <https://arxiv.org/abs/1512.03385>`__. It is a stack of two bidirectional block layers
  (see previous bullet point) with a skip connection.

PersLay Layer
.............
The PersLay layer introduced in `arXiv:1904.09378 <https://arxiv.org/abs/1904.09378>`__
is implemented in the ``morphoclass.layers.PersLay`` class. Please refer to the
official article, as well as to the API documentation for further details.

Internally PersLay performs several transformation steps, one of which is
a point transformation applied to the points of the input graph. The point
transformation can be implemented in different ways, and the current PersLay
implementation supports two such transformations which are implemented
in the following classes:

- ``morphoclass.layers.GaussianPointTransformer``
- ``morphoclass.layers.PointwisePointTransformer``

Pooling Layers
..............

One common characteristic of graph neural nets is the presence of
pooling layers. These are necessary to reduce the input graphs with
arbitrary numbers of nodes to an output that is a tensor of constant
dimension for any kind of input.

More precisely, a typical graph classification or regression net will
consit of three stages:

1. Node feature extractor
2. Pooling
3. Classifier / regressor

The feature extractor may consist of arbitrary convolutions that assign feature
vectors to all nodes in the graph. The pooling stage combines all node
feature vectors into one vector that represents the features of the whole graph.
Finally, the last stage is uses this graph embedding vector to produce
a classification or regression output, and is typically implemented as a
combination of fully connected layers.

The simplest way of pooling is bei either averaging or summing the
node feature vectors. These are readily implemented in the PyTorch-Geometric
package in the following functions:

- ``torch_geometric.nn.global_mean_pool``
- ``torch_geometric.nn.global_add_pool``

The morphoclass package provides two additional pooling layers:

- ``morphoclass.layers.AttentionGlobalPool``
- ``morphoclass.layers.TreeLSTMPool``

The TreeLSTM layer is an experimental implementation of the architecture introduced in
`arXiv:1503.00075 <https://arxiv.org/abs/1503.00075>`__. Its current implementation is
relatively slow and therefore not recommended. An example of an alternative implementation
can be found in the
`DGL library documentation <https://docs.dgl.ai/tutorials/models/2_small_graph/3_tree-lstm.html>`__.

The attention layer, in contrast, is a viable alternative to the sum and average pooling and
in some case may yield better results. In contrast to these layers it computes a weighted
sum of the node features where the weights depend on learnable parameters. Note that the
``ManNet`` model presented in the :ref:`gnn` section used the average pooling by default, but
can be forced to use the attention pool instead by passing the parameter ``pool_name="att"``
in the constructor.

Other Layers
............

There are two layers that don't fall in any of the categories above:

- ``morphoclass.layers.Cat``
- ``morphoclass.layers.RunningStd``

The former is very important and allows to concatenate activations of two different layers.
For example, if the outputs of two different layers have shapes ``(n_nodes, n_features_1)``
and ``(n_nodes, n_features_2)``, then the combined output will have the shape
``(n_nodes, n_features_1 + n_features_2)``. This allows to construct models with a non-linear
flow where the data takes two different paths that are re-combined at a later point. An
example of such construction are the bi-directional graph convolution layers discussed above.

The second layer, ``RunningStd``, can be used to normalize node features of the input data
by their standard deviation. The standard deviation is computed at training time on the fly
from all the data that passes through this layer. At inference time the inputs are normalized
by the standard deviation that was obtained at training time.

Models
------
The morphoclass package contains a number of pre-defined models. Some of them were already
introduced in the previous sections. Here we give an overview of all remaining models.
All models can be found in the ``morphoclass.models`` module.

There are a number of models that are related to the ``ManNet`` model presented in the
:ref:`gnn` section. The models ``MultiAdjNet`` and ``BidirectionalNet`` are precursors
of the ``ManNet`` model with fewer customization possibilities than the latter. In fact,
the word `man` in ``ManNet`` was initially an abbreviation for ``MultiAdjNet``.

Furthermore, one finds the ``ManEmbedder`` and ``ManNetR`` models. The former is the feature
extraction part of the ``ManNet`` and can be used as a building block for constructing other models.
The latter is the same as the ``ManNet`` but without the final softmax layer. This makes it
a regression-type model.

Next there are a family of residual-type graph nets: ``ManResNet1``, ``ManResNet2``, and
``ManResNet3``. The are all modification of the ``ManNet`` model and make use of the
residual bidirectional layers discussed previously. They differ by the number of layers,
the former being the most shallow one, and the latter the deepest.

Finally, the last graph-based models is called ``HBNet`` (for hierarchical bidirectional net).
It is yet another modification of the ``BidirectionalNet`` model that takes into account
the hierarchical structure of the morphological classes, e.g. the classes TPC-A, TPC-B, and UPC
can be first split into TPC and UPC, and the TPC can then be further refined into TPC-A and TPC-B.
The ``HBNet`` has the same feature extractor as the ``BidirectionalNet``, but utilizes a
hierarchical version of the classifier that tracks the hierarchy in the class labels.

Apart from the graph-convolution based models from above and the non-graph models shown in
sections :ref:`cnn` and :ref:`perslay` there are also two composite models that combine the
feature extractors based on graph-convolutions, regular convolutions, and PersLay layers.

The first one is called ``ConcateCNNet`` (with the trainer class ``ConcateCNNetTrainer``) and
combines the embedders from the GNN adn the CNN models. The resulting node features are combined
and processed by a common classification layer.

The second compound model and trainer are called ``ConcateNet`` and ``ConcateNetTrainer``, and are
an analogous combination of the GNN with the CorianderNet.

Example
-------
