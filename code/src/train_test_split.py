#!/usr/bin/env python

import inspect

import pandas as pd
import numpy as np
import mendeleev
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import HDBSCAN, if not available, provide alternative
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan")


def group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data by groups so train and test share no cluster IDs."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in gss.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return X_train, X_test, y_train, y_test
    raise RuntimeError("GroupShuffleSplit produced no splits")


def group_split_with_val(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    val_random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train / validation / test with no cluster overlap between splits."""
    if val_random_state is None:
        val_random_state = random_state + 1

    X_train_pool, X_test, y_train_pool, y_test = group_split(
        X, y, groups, test_size=test_size, random_state=random_state
    )
    groups_train = groups.loc[X_train_pool.index]
    X_train, X_val, y_train, y_val = group_split(
        X_train_pool,
        y_train_pool,
        groups_train,
        test_size=val_size,
        random_state=val_random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def group_split_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Return integer index lists for group-based train/test splits."""
    X_train, X_test, _, _ = group_split(X, y, groups, test_size=test_size, random_state=random_state)
    return list(X_train.index), list(X_test.index)


def group_split_with_val_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    val_random_state: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return integer index lists for group-based train/val/test splits."""
    X_train, X_val, X_test, _, _, _ = group_split_with_val(
        X,
        y,
        groups,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        val_random_state=val_random_state,
    )
    return list(X_train.index), list(X_val.index), list(X_test.index)


def group_split_with_random_val(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    val_random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train / validation / test.

    Stage 1 — group-based: train+val share no cluster IDs with test.
    Stage 2 — random:      train and val are split randomly (may share clusters).
    """
    if val_random_state is None:
        val_random_state = random_state + 1

    # Stage 1: isolate test set by cluster groups
    X_train_pool, X_test, y_train_pool, y_test = group_split(
        X, y, groups, test_size=test_size, random_state=random_state
    )

    # Stage 2: random split of the pool into train / validation
    val_fraction_of_pool = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=val_fraction_of_pool,
        random_state=val_random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def group_split_with_random_val_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    val_random_state: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return integer index lists for group-based train/val/test splits.

    Stage 1 — group-based: train+val share no cluster IDs with test.
    Stage 2 — random:      train and val are split randomly (may share clusters).
    """
    X_train, X_val, X_test, _, _, _ = group_split_with_random_val(
        X,
        y,
        groups,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        val_random_state=val_random_state,
    )
    return list(X_train.index), list(X_val.index), list(X_test.index)


def random_kfold_indices(
    X: pd.DataFrame,
    n_splits: int = 10,
    random_state: int = 42,
    shuffle: bool = True,
) -> list[tuple[list[int], list[int]]]:
    """Return train/test index lists for random K-fold CV."""
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    folds: list[tuple[list[int], list[int]]] = []
    index = X.index
    for train_pos, test_pos in kfold.split(X):
        folds.append((list(index[train_pos]), list(index[test_pos])))
    return folds


def group_kfold_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 10,
    random_state: int = 42,
    shuffle: bool = False,
) -> list[tuple[list[int], list[int]]]:
    """Return train/test index lists for grouped K-fold CV.

    All records with the same group label are assigned to the same fold.
    """
    group_kfold_params = {"n_splits": n_splits}
    group_kfold_signature = inspect.signature(GroupKFold)
    if "shuffle" in group_kfold_signature.parameters:
        group_kfold_params["shuffle"] = shuffle
        group_kfold_params["random_state"] = random_state if shuffle else None
    elif shuffle:
        raise ValueError(
            "This scikit-learn version does not support GroupKFold(shuffle=True). "
            "Use shuffle=False or upgrade scikit-learn."
        )
    group_kfold = GroupKFold(**group_kfold_params)
    folds: list[tuple[list[int], list[int]]] = []
    index = X.index
    for train_pos, test_pos in group_kfold.split(X, y, groups=groups):
        folds.append((list(index[train_pos]), list(index[test_pos])))
    return folds


def valid_elements(elements: list[str]) -> list[str]:
    """
    Get the valid elements from a list of elements.
    """
    return [element for element in elements if is_valid_element(element)]

def is_valid_element(element: str) -> bool:
    """
    Check if an element is valid.
    """
    try:
        mendeleev.element(element)
        return True
    except:
        return False

def k_means_clustering(df_composition: pd.DataFrame, 
                      n_clusters: int = 5, 
                      random_state: int = 42,
                      standardize: bool = False) -> pd.DataFrame:
    """
    Perform k-means clustering on the composition dataframe.
    
    Parameters:
    -----------
    df_composition : pd.DataFrame
        DataFrame with element compositions (columns are element names, values are fractions)
    n_clusters : int, default=5
        Number of clusters to form
    random_state : int, default=42
        Random state for reproducibility
    standardize : bool, default=False
        Whether to standardize the features before clustering
        
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added 'cluster' column
    """
    # Make a copy to avoid modifying the original
    df_result = df_composition.copy()
    
    # Prepare data
    X = df_composition.values
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    df_result['cluster'] = clusters
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"K-means clustering completed with {n_clusters} clusters")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    return df_result

def DBSCAN_clustering(df_composition: pd.DataFrame, 
                     eps: float = 0.5, 
                     min_samples: int = 5,
                     standardize: bool = False) -> pd.DataFrame:
    """
    Perform DBSCAN clustering on the composition dataframe.
    
    Parameters:
    -----------
    df_composition : pd.DataFrame
        DataFrame with element compositions (columns are element names, values are fractions)
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be considered as a core point
    standardize : bool, default=True
        Whether to standardize the features before clustering
        
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added 'cluster' column (-1 indicates noise/outliers)
    """
    # Make a copy to avoid modifying the original
    df_result = df_composition.copy()
    
    # Prepare data
    X = df_composition.values
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    df_result['cluster'] = clusters
    
    # Calculate statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"DBSCAN clustering completed")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Calculate silhouette score (only for non-noise points)
    if n_clusters > 1 and n_noise < len(clusters):
        non_noise_mask = clusters != -1
        if np.sum(non_noise_mask) > 1:
            silhouette_avg = silhouette_score(X_scaled[non_noise_mask], clusters[non_noise_mask])
            print(f"Silhouette Score (excluding noise): {silhouette_avg:.3f}")
    
    return df_result

def HDBSCAN_clustering(df_composition: pd.DataFrame, 
                      min_cluster_size: int = 5,
                      min_samples: Optional[int] = None,
                      standardize: bool = False) -> pd.DataFrame:
    """
    Perform HDBSCAN clustering on the composition dataframe.
    
    Parameters:
    -----------
    df_composition : pd.DataFrame
        DataFrame with element compositions (columns are element names, values are fractions)
    min_cluster_size : int, default=5
        Minimum size of clusters
    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered a core point
    standardize : bool, default=True
        Whether to standardize the features before clustering
        
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added 'cluster' column (-1 indicates noise/outliers)
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN is not available. Install with: pip install hdbscan")
    
    # Make a copy to avoid modifying the original
    df_result = df_composition.copy()
    
    # Prepare data
    X = df_composition.values
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = clusterer.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    df_result['cluster'] = clusters
    
    # Calculate statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"HDBSCAN clustering completed")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Calculate silhouette score (only for non-noise points)
    if n_clusters > 1 and n_noise < len(clusters):
        non_noise_mask = clusters != -1
        if np.sum(non_noise_mask) > 1:
            silhouette_avg = silhouette_score(X_scaled[non_noise_mask], clusters[non_noise_mask])
            print(f"Silhouette Score (excluding noise): {silhouette_avg:.3f}")
    
    return df_result

def find_optimal_k(df_composition: pd.DataFrame, 
                  max_k: int = 30, 
                  standardize: bool = False) -> Tuple[int, float]:
    """
    Find the optimal number of clusters using the elbow method and silhouette analysis.
    
    Parameters:
    -----------
    df_composition : pd.DataFrame
        DataFrame with element compositions
    max_k : int, default=10
        Maximum number of clusters to test
    standardize : bool, default=True
        Whether to standardize the features before clustering
        
    Returns:
    --------
    Tuple[int, float]
        Optimal number of clusters and corresponding silhouette score
    """
    X = df_composition.values
    
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Calculate inertia and silhouette scores for different k values
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, clusters))
    
    # Find optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    optimal_score = max(silhouette_scores)
    
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best silhouette score: {optimal_score:.3f}")
    
    return optimal_k, optimal_score

def visualize_clusters(df_with_clusters: pd.DataFrame, 
                      method: str = "PCA",
                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize clustering results using dimensionality reduction.
    
    Parameters:
    -----------
    df_with_clusters : pd.DataFrame
        DataFrame with cluster labels
    method : str, default="PCA"
        Dimensionality reduction method ("PCA" or "t-SNE")
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    """
    # Separate features and cluster labels
    feature_cols = [col for col in df_with_clusters.columns if col != 'cluster']
    X = df_with_clusters[feature_cols].values
    clusters = df_with_clusters['cluster'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if method == "PCA":
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_scaled)
        xlabel = f"PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)"
        ylabel = f"PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)"
    elif method == "t-SNE":
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            xlabel = "t-SNE 1"
            ylabel = "t-SNE 2"
        except ImportError:
            print("t-SNE not available, using PCA instead")
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)
            xlabel = f"PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)"
            ylabel = f"PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)"
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
    
    # Create visualization
    plt.figure(figsize=figsize)
    
    # Get unique clusters and colors
    unique_clusters = np.unique(clusters)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster in enumerate(unique_clusters):
        if cluster == -1:  # Noise points
            plt.scatter(X_reduced[clusters == cluster, 0], 
                       X_reduced[clusters == cluster, 1],
                       c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            plt.scatter(X_reduced[clusters == cluster, 0], 
                       X_reduced[clusters == cluster, 1],
                       c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {cluster}')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Clustering Results ({method})')
    #plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_cluster_compositions(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the average composition of each cluster.
    
    Parameters:
    -----------
    df_with_clusters : pd.DataFrame
        DataFrame with cluster labels
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with average compositions for each cluster
    """
    feature_cols = [col for col in df_with_clusters.columns if col != 'cluster']
    
    # Calculate mean composition for each cluster
    cluster_means = df_with_clusters.groupby('cluster')[feature_cols].mean()
    
    # Calculate cluster sizes
    cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
    cluster_means['cluster_size'] = cluster_sizes
    
    print("Cluster Analysis:")
    print("=" * 50)
    for cluster in cluster_means.index:
        if cluster == -1:
            print(f"\nNoise Points (Cluster {cluster}): {cluster_sizes[cluster]} samples")
        else:
            print(f"\nCluster {cluster}: {cluster_sizes[cluster]} samples")
            # Show top 5 elements by average composition
            top_elements = cluster_means.loc[cluster, feature_cols].nlargest(5)
            print("Top 5 elements by average composition:")
            for element, composition in top_elements.items():
                if composition > 0.01:  # Only show elements with >1% composition
                    print(f"  {element}: {composition:.3f}")
    
    return cluster_means


def get_composition(df:pd.Series)-> str:
    """
    Get the composition string from the series.
    """
    tmp_composition = []
    for k,v in df.items():
        if v > 1e-5:
            tmp_composition.append((k,v*100))
    tmp_composition.sort(key=lambda x: x[1], reverse=True)
    return "".join([f"{k}{v:.2f}" for k,v in tmp_composition])

def get_composition_string(df:pd.DataFrame)-> pd.DataFrame:
    """
    Get the composition string from the dataframe.
    """
    #from pymatgen.core.composition import Composition
    return df.apply(lambda x: get_composition(x), axis=1).to_frame(name="composition")

def test_get_composition_string():
    """
    Test the get_composition_string function.
    """
    df = pd.read_excel("density_mixing_rule_vs_experimental_density_relative_error.xlsx")
    df_composition = df[valid_elements(df.columns)]
    df_composition_string = get_composition_string(df_composition)
    df_composition_string.to_excel("composition_string.xlsx", index=False)

def test_groups():
# Example usage
    # Load data
    df = pd.read_excel("density_mixing_rule_vs_experimental_density_relative_error.xlsx")

    density = df["p_g_per_cm3"]
    
    # Get valid elements
    valid_elements_list = valid_elements(df.columns)
    df_composition = df[valid_elements_list]
    
    print(f"Loaded data with {len(df_composition)} samples and {len(valid_elements_list)} elements")
    print(f"Elements: {valid_elements_list}")
    
    # Find optimal number of clusters
    optimal_k, best_score = find_optimal_k(df_composition, max_k=60)
    
    # Perform clustering with optimal k
    df_kmeans = k_means_clustering(df_composition, n_clusters=optimal_k,standardize=False)
    df_kmeans["density"] = density
    df_dbscan = DBSCAN_clustering(df_composition, eps=0.5, min_samples=5,standardize=False)
    df_dbscan["density"] = density
    #df_hdbscan = HDBSCAN_clustering(df_composition, min_cluster_size=2,min_samples=1,standardize=False)
    df_hdbscan = HDBSCAN_clustering(df_composition, min_cluster_size=2,standardize=False)
    df_hdbscan["density"] = density
    
    # Visualize results
    print("K-means clustering results:")
    visualize_clusters(df_kmeans, method="t-SNE")
    print("DBSCAN clustering results:")
    visualize_clusters(df_dbscan, method="t-SNE")
    print("HDBSCAN clustering results:")
    visualize_clusters(df_hdbscan, method="t-SNE")
    
    # Analyze cluster compositions
    #print("K-means cluster analysis:")
    #cluster_analysis = analyze_cluster_compositions(df_kmeans)
    #print("DBSCAN cluster analysis:")
    #cluster_analysis = analyze_cluster_compositions(df_dbscan)
    print("HDBSCAN cluster analysis:")
    cluster_analysis = analyze_cluster_compositions(df_hdbscan)

    # Save results
    df_composition_string = pd.read_excel("composition_string.xlsx")
    df_kmeans["composition"] = df_composition_string["composition"]
    df_dbscan["composition"] = df_composition_string["composition"]
    df_hdbscan["composition"] = df_composition_string["composition"]
    df_kmeans["density"] = density
    df_dbscan["density"] = density
    df_hdbscan["density"] = density
    df_kmeans.sort_values(by="cluster", inplace=True)
    df_dbscan.sort_values(by="cluster", inplace=True)
    df_hdbscan.sort_values(by="cluster", inplace=True)
    df_kmeans.to_excel("kmeans_clustering_results_with_composition.xlsx", index=False)
    df_dbscan.to_excel("dbscan_clustering_results_with_composition.xlsx", index=False)
    df_hdbscan.to_excel("hdbscan_clustering_results_with_composition.xlsx", index=False)
    
if __name__ == "__main__":
    test_groups()
    #test_get_composition_string()
