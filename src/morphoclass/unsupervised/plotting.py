# Copyright © 2022-2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plotting utilities for unsupervised m-type classification tools."""
from __future__ import annotations

import matplotlib.cm as cm
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn import cluster
from sklearn import decomposition
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score


def plot_embedding_pca(
    embeddings,
    ax=None,
    pca=None,
    labels=None,
    title=None,
    annotate_indices=False,
    annotation_fontsize=7,
):
    """Plot two principal components of given embedding.

    This is an altered version which shows annotations.

    Parameters
    ----------
    embeddings : np.ndarray
        The embedding to be plotted. The shape should be (n_samples, d_embedding).
    ax :  matplotlib.axes.Axes, optional
        A matplotlib axis object.
    pca : sklearn.decomposition.PCA, optional
        An fitted instance of the PCA class from scikit-learn. If not provided
        then a new instance will be created and fitted.
    labels : list_like, optional
        Labels for all samples. Should be of length n_samples.
    title : str
        Title for the plot.
    annotate_indices : bool
        If true then each point in the plot will be annotated with its
        index in the provided embeddings.
    annotation_fontsize : int
        The font size for the annotation. Has no effect if `annotate_indices`
        is False.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The PCA of the given embeddings.
    """
    if ax is None:
        fig = Figure()
        ax = fig.subplots()

    color = None
    if labels is not None:
        colormap = cm.tab10

        # Initialise colors
        unique_labels = sorted(set(labels))
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        n_classes = len(label_to_num)
        numeric_labels = np.array([label_to_num[label] for label in labels])
        color = colormap(numeric_labels / n_classes)

        # Create Legend
        handles = []
        names = []
        for label, numeric_label in sorted(label_to_num.items()):
            handle = Line2D(
                xdata=[0],
                ydata=[0],
                marker="o",
                linestyle="",
                color=colormap(numeric_label / n_classes),
            )
            handles.append(handle)
            names.append(label)
        legend = ax.legend(handles, names, title="Classes", framealpha=0.5)
        ax.add_artist(legend)

    # Plot PCA components
    if pca is None:
        pca = decomposition.PCA(n_components=2)
        pca.fit(embeddings)
    components0, components1 = pca.transform(embeddings).T
    ev0, ev1 = np.round(pca.explained_variance_ratio_ * 100).astype(int)
    ax_title = f"explained variance:\n{ev0}% + {ev1}% = {ev0 + ev1}%"
    if title is not None:
        ax_title = f"{title}\n{ax_title}"
    ax.set_title(ax_title)
    ax.scatter(components0, components1, color=color)
    if annotate_indices:
        for i, (c0, c1) in enumerate(zip(components0, components1)):
            ax.annotate(i, (c0, c1), fontsize=annotation_fontsize)
    ax.set_xlabel("PCA${}_0$")
    ax.set_ylabel("PCA${}_1$")

    return pca


def make_silhouette_plot(embeddings, n_clusters, random_state=0):
    """Create a silhouette plot given an embedding.

    This code up to small amendments is copied from
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Parameters
    ----------
    embeddings : np.ndarray
        The embedding to be plotted. The shape should be (n_samples, d_embedding).
    n_clusters : int
        The number of clusters to create.
    random_state : int
        The random state to use for KMeans (first plot) and PCA (second plot).
    """
    # Create a subplot with 1 row and 2 columns
    fig = Figure(figsize=(12, 5), constrained_layout=True)
    ax1, ax2 = fig.subplots(1, 2)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(embeddings) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = clusterer.fit_predict(embeddings)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(embeddings, cluster_labels)

    y_lower = 10
    colormap = cm.tab10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colormap(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(f"Silhouette Plot $n={n_clusters}$")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the y-axis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = colormap(cluster_labels.astype(float) / n_clusters)
    pca = decomposition.PCA(random_state=random_state)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    centers = pca.transform(clusterer.cluster_centers_)
    ax2.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        marker=".",
        s=50,
        lw=0,
        alpha=1,
        c=colors,
        edgecolor="k",
    )

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("PCA Visualisation")
    ax2.set_xlabel(r"$\operatorname{PCA}_0$")
    ax2.set_ylabel(r"$\operatorname{PCA}_1$")

    fig.show()
