---
layout: custom_post
title: "Rust: Clustering"
date: 2025-02-19 9:00:00 +0100
categories: clustering
#permalink: /seq2seq
#categories: encoder-decoder keras
#permalink: CoRec
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Introduction

Clustering is an unsupervised learning task, that aims at grouping together unlabeled data using some kind of metric.
We are going to implement two approaches to clustering, KMeans and DBSCAN, using linfa, a Rust ML library that follows similar principles to scikit-learn. We are using the well-known Iris Dataset without considering the targets. 

KMeans is a clustering algorithm that minimizes the within-cluster-sum-of-squares (WCSS), or inertia

$$
\sum_{i=0}^n min_{m_j} (|| x_i - m_j ||)^2
$$

where $$m_j$$ is the mean of the cluster j, also called **centroid**. The number of clusters K is predetermined and given to the algorithm. This objective function tries to separate samples in groups of equal variance. However, it has its drawbacks:  inertia is not normalized, and as such performs poorly on high-dimensional spaces, where distances quickly explode, and it also assumes that the data distribution is isotropic and convex, thus behaving poorly on elongated clusters, or clusters with irregular shapes.

In practice, KMeans starts by defining centroids, as many as the number of clusters. Then, the algorithm loops between two steps: assigning each sample point to its nearest centroid, and recomputing each centroid as the mean of all the points assigned to it. The difference between new and old centroids is computed and the loop continues until this difference is less than a predetermined threshold.

!["Clusters here"](../../../../assets/voronoi.png)

DBSCAN views clustering as the task of separating areas of high density from areas of low density, that is finding a set of *core* samples, belonging to high density areas, as opposed to *non-core*, or noisy, samples. In order to govern this process, two hyperparameters have to be defined: *eps* and *min_samples*, the former defines the maximum distance between samples to be considered core, thereby controlling the local neighborhood of the points, while the latter defines the minimum number of neighbor samples of a sample that have to exist for it to be taken as core. Increasing the number of minimum samples is useful for very noisy and large data sets.

!["Clusters here"](../../../../assets/dbscan.png)

# Solution

The whole code example is publically available [here](https://github.com/Plutone11011/plutone11011.github.io).
First off, we have to load the iris dataset. Typically, since for KMeans we have to choose the number of clusters, we would use a method like **elbow method** to choose the optimal number of clusters


```rust
    let ds = load_iris_dataset();
    let K: Range<usize> = 2..10usize;
    // let's reduce to two dimensions
    let mut wcss : Vec<f64> = vec![];
    let n_points = ds.records.len_of(Axis(0));
    for k in K.clone() {
        
        let (clusters, centroids) = kmeans(&ds, k);
        //println!("Clusters for k = {}, {:?}", k, clusters);
        //println!("Centroids shape {:?}", centroids.shape());
        
        let mut squared_distances = Array1::zeros(n_points);
        // for each cluster k, we compute the within cluster sum of squares WCSS, which is defined 
        // as the sum of squared distances between each point and its cluster centroid
        for i in 0..n_points {
            let point = ds.records.row(i);
            let centroid = centroids.row(clusters[i]); // Get the corresponding centroid
            let distance = point.to_owned() - centroid.to_owned(); // Compute the difference
            squared_distances[i] = distance.dot(&distance); // Compute squared distance
        }
        wcss.push(squared_distances.sum());
        // let _ = draw_clusters(clusters, &feature_names);
    }
    println!("WCSS calculated as inertia {:?}", wcss);
    let _ = draw_wcss(&K, wcss);
```

With plotters, we can show the wcss computed for each number of clusters.

!["Elbow method"](../../../../assets/elbow_method.jpg)

Three clusters is where there is the steepest change of inertia, and we also know that the Iris Dataset contains samples of three different classes.

```rust
fn kmeans(ds: &Dataset<f64, usize, Ix1>, n_clusters: usize) -> (Array1<usize>, Array2<f64>) {
    
    let rng = thread_rng();        
    // Let's configure and run our K-means algorithm
    // We use the builder pattern to specify the hyperparameters
    // `n_clusters` is the only mandatory parameter.
    // If you don't specify the others (e.g. `n_runs`, `tolerance`, `max_n_iterations`)
    // default values will be used.
    let model = KMeans::params_with_rng(n_clusters, rng.clone())
        .tolerance(1e-2)
        .fit(ds)
        .expect("KMeans fitted");


    let clusters= model.predict(ds);
    let centroids = model.centroids().to_owned();
    (clusters, centroids)
    // let _ = draw_clusters(clusters, feature_names);
}

```

We end up with the following clusters
!["KMeans"](../../../../assets/clusters_iris_kmeans.jpg)


Let's compare results with DBSCAN. We choose a tolerance *eps* of 0.5 and number of *min_samples* equal to 5.


```rust
fn dbscan(ds: &Dataset<f64, usize, Ix1>, min_points: usize, tol: f64) -> Array1<Option<usize>> {
    
           

    let clusters = Dbscan::params(min_points)
        .tolerance(tol)
        .transform(&ds.records)
        .unwrap();


    // let clusters= model.predict(ds);
    // let centroids = model.centroids().to_owned();
    clusters
    // let _ = draw_clusters(clusters, feature_names);
}
```

These are the clusters produced by DBSCAN
!["DBSCAN"](../../../../assets/clusters_iris_dbscan.jpg)

Notice how, in both cases, we use a dimensionality reduction technique, PCA, to plot the otherwise 4-dimensional Iris dataset in two dimensions.
DBSCAN finds some points to be noisy, non-core, and two clusters, instead of the three that we gave to KMeans.

# Conclusion
Clustering is a core unsupervised learning task, KMeans and DBSCAN are two well-known methods in classic machine learning. There are other ways to cluster data,
like using autoencoders or embeddings, however, for a smaller dataset like Iris, simpler methods are sufficient. 