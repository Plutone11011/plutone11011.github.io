use std::error::Error;
use std::collections::HashMap;
use std::ops::Range;
use linfa::prelude::*;
use ndarray::prelude::*;
use rand::prelude::*;
use plotters::{prelude::*, style::full_palette::PURPLE};
use linfa_reduction::Pca;
use linfa_clustering::{KMeans, Dbscan};
/// Loads the Iris dataset from a CSV file and returns a linfa Dataset
pub fn load_iris_dataset() -> Dataset<f64, usize, Ix1>{


    let ds: Dataset<f64, usize, Ix1> = linfa_datasets::iris();

    println!("Feature names {:?}", ds.feature_names());
    println!("Dataset records shape: {:?}", ds.records.shape());
    println!("Dataset targets shape: {:?}", ds.targets.shape());
    ds
}

pub fn draw_clusters(ds: &Dataset<f64, f64, Ix2>, clusters: &ArrayBase<ndarray::OwnedRepr<usize>, Ix1>) -> Result<(), Box<dyn Error>>{


    let mut clusters_points: HashMap<usize, Vec<(f64,f64)>> = HashMap::new();

    // split dataset in two clusters
    for (i, (feature, _)) in ds.sample_iter().enumerate() {
        // assume dataset has been reduced to two dimensions with PCA or other similar method
        let point = (feature[0], feature[1]);
        clusters_points.entry(clusters[i]).and_modify(|points_in_cluster| points_in_cluster.push(point)).or_insert(vec![]);
    }
    
    println!("Clusters to points {:?}", clusters_points);

    let drawing_area_width = 1000;
    let drawing_area_height = 1000;
    let file_name = "clusters_iris_kmeans.jpg";
    let root_area = BitMapBackend::new(&file_name, (drawing_area_width, drawing_area_height)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .margin_bottom(50)
        .margin_left(80)
        .margin_right(10)
        .caption("Clusters", ("sans-serif", 40))
        .build_cartesian_2d(-5f64..10f64, -5f64..10f64)
        .unwrap();

    ctx.configure_mesh()
        // .x_labels(0)
        // .y_labels(0)
        // .label_style(("sans-serif", 20))
        .draw()?;

    // assuming 6 clusters with elbow 
    let colors = [&PURPLE, &GREEN, &RED, &BLACK, &YELLOW, &BLUE];
    for (i, (_,points)) in clusters_points.iter().enumerate(){
        ctx.draw_series(
            points.iter().map(|point| Circle::new(*point, 5, colors[i])),
        )
        .unwrap();
    }

    // root_area.draw_text(&feature_names[0],  &("sans-serif", 20).into_text_style(&root_area), (500, 970))?;
    // root_area.draw_text(&feature_names[1],  &("sans-serif", 20).into_text_style(&root_area), (10, 500))?;
    

    Ok(())
}

pub fn draw_wcss(n_clusters: &Range<usize>, wcss: Vec<f64>) -> Result<(), Box<dyn Error>> {

    let drawing_area_width = 1000;
    let drawing_area_height = 1000;
    let file_name = "elbow_method.jpg";
    println!("File name {}", file_name);
    let root_area = BitMapBackend::new(&file_name, (drawing_area_width, drawing_area_height)).into_drawing_area();
    
    root_area.fill(&WHITE)?;
    
    let max_clusters = n_clusters.clone().max().unwrap_or_default();
    let max_wcss = wcss.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&300.0);
    let mut chart = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .margin_bottom(50)
        .margin_left(80)
        .margin_right(10)
        .caption("Elbow method", ("sans-serif", 40))
        .build_cartesian_2d(0..max_clusters+2, 0f64..max_wcss+1.)
        .unwrap();

    chart.configure_mesh().draw()?;
    let series_data: Vec<(usize, f64)> = n_clusters.clone().zip(wcss.iter()).map(|(n, &w)| (n, w)).collect();
    chart.draw_series(LineSeries::new(series_data, &RED))?
        .label("WCSS");


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

fn dbscan(ds: &Dataset<f64, usize, Ix1>, min_points: usize) -> Array1<Option<usize>> {
    
           

    let clusters = Dbscan::params(min_points)
        .tolerance(0.5)
        .transform(&ds.records)
        .unwrap();


    // let clusters= model.predict(ds);
    // let centroids = model.centroids().to_owned();
    clusters
    // let _ = draw_clusters(clusters, feature_names);
}



fn main() {
    println!("Hello, world!");

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

    let (k_means_clusters, _) = kmeans(&ds, 6);
    println!("KMeans clusters {:?}", k_means_clusters);
    
    
    // dbscan
    let dbscan_clusters = dbscan(&ds, 2);
    println!("Dbscan clusters {:?}", dbscan_clusters);
    let embedding = Pca::params(2)
        .fit(&ds).unwrap();
    let reduced_ds = embedding.predict(ds);
    println!("PCAd dataset {:?}", reduced_ds.records.shape());
    let _ = draw_clusters(&reduced_ds, &k_means_clusters);
}
