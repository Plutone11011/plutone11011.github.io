use std::{collections::{vec_deque, VecDeque}, error::Error};
use linfa::prelude::*;
use ndarray::{prelude::*, Data};
use rand::prelude::*;
use plotters::prelude::*;
use linfa_clustering::KMeans;
/// Loads the Iris dataset from a CSV file and returns a linfa Dataset
pub fn load_iris_dataset() -> Dataset<f64, usize, Ix1>{


    let ds: Dataset<f64, usize, Ix1> = linfa_datasets::iris();

    println!("Feature names {:?}", ds.feature_names());
    println!("Dataset records shape: {:?}", ds.records.shape());
    println!("Dataset targets shape: {:?}", ds.targets.shape());
    ds
}

pub fn draw_clusters(clusters_ds: Dataset<f64, usize, Ix1>, feature_names: &Vec<String>) -> Result<(), Box<dyn Error>>{
    
    let mut c1: Vec<(f64,f64)> = Vec::new();
    let mut c2: Vec<(f64,f64)> = Vec::new();
    
    // split dataset in two clusters
    for (feature, cluster) in clusters_ds.sample_iter() {
        let point = (feature[0], feature[1]);
        if cluster.first().unwrap() == &0_usize {
            c1.push(point);
        } else {
            c2.push(point);
        }
    }
    
    let drawing_area_width = 1000;
    let drawing_area_height = 1000;
    let file_name = format!("clusters_iris_{}.jpg", feature_names.join("_"));
    println!("File name {}", file_name);
    let root_area = BitMapBackend::new(&file_name, (drawing_area_width, drawing_area_height)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .margin_bottom(50)
        .margin_left(80)
        .caption("Clusters", ("sans-serif", 40))
        .build_cartesian_2d(-5f64..10f64, -5f64..10f64)
        .unwrap();

    ctx.configure_mesh()
        // .x_labels(0)
        // .y_labels(0)
        // .label_style(("sans-serif", 20))
        .draw().unwrap();

    ctx.draw_series(
        c1.iter().map(|point| TriangleMarker::new(*point, 5, &BLUE)),
    )
    .unwrap();

    ctx.draw_series(c2.iter().map(|point| Circle::new(*point, 5, &RED)))
        .unwrap();

    root_area.draw_text(&feature_names[0],  &("sans-serif", 20).into_text_style(&root_area), (500, 970))?;
    root_area.draw_text(&feature_names[1],  &("sans-serif", 20).into_text_style(&root_area), (10, 500))?;
    // ctx.draw_series(PointSeries::<_,_,Circle<_,_>,_>::new(c1, c1_size, &BLUE))
    //     .unwrap();

    // ctx.draw_series(PointSeries::<_,_,Circle<_,_>,_>::new(c2, c2_size, &RED))
    //     .unwrap();

    Ok(())
}


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


fn main() {
    println!("Hello, world!");

    let ds = load_iris_dataset();
    let K = 2..8usize;
    // let's reduce to two dimensions
    let mut wcss : Vec<f64> = vec![];
    let n_points = ds.records.len_of(Axis(0));
    for k in K {
        
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
    // let's reduce to two dimensions
    // let feature_names = features[1..3].to_vec();
    // let ds_slice: Array2D = ds.records.slice(s![..,1..3]);
    // let clusters = kmeans_on_features_subset(&feature_names, ds_slice);
    // let _ = draw_clusters(clusters, &feature_names);

    // let feature_names = features[2..4].to_vec();
    // let ds_slice: Array2D = ds.records.slice(s![..,2..4]);
    // let clusters = kmeans_on_features_subset(&feature_names, ds_slice);
    // let _ = draw_clusters(clusters, &feature_names);

    // let feature_names = features[0..3].iter().step_by(2).cloned().collect();
    // let ds_slice: Array2D = ds.records.slice(s![..,0..3;2]);
    // let clusters = kmeans_on_features_subset(&feature_names, ds_slice);
    // let _ = draw_clusters(clusters, &feature_names);
}
