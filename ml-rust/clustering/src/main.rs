use std::error::Error;
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


fn kmeans_on_features_subset(feature_names: &Vec<String>, ds: Array2D) {
    let ds_small = DatasetBase::from(ds)
        .with_feature_names(feature_names.clone())
        .to_owned();
    // Our random number generator, seeded for reproducibility
    
    let mut rng = thread_rng();
    // `expected_centroids` has shape `(n_centroids, n_features)`
    // i.e. three points in the 2-dimensional plane
    let expected_centroids = array![[0., 1.], [-0.1, 5.6]];
    // Let's generate a synthetic dataset: three blobs of observations
    // (100 points each) centered around our `expected_centroids`
    let n_clusters = expected_centroids.len_of(Axis(0));

    // Standard K-means
    
        
    // Let's configure and run our K-means algorithm
    // We use the builder pattern to specify the hyperparameters
    // `n_clusters` is the only mandatory parameter.
    // If you don't specify the others (e.g. `n_runs`, `tolerance`, `max_n_iterations`)
    // default values will be used.
    let model = KMeans::params_with_rng(n_clusters, rng.clone())
        .tolerance(1e-2)
        .fit(&ds_small)
        .expect("KMeans fitted");


    let clusters = model.predict(ds_small);
    let _ = draw_clusters(clusters, feature_names);
}

type Array2D<'a> = ArrayBase<ndarray::ViewRepr<&'a f64>, Dim<[usize; 2]>> ;

fn main() {
    println!("Hello, world!");

    let ds = load_iris_dataset();
    let features = ds.feature_names();
    
    // let's reduce to two dimensions
    let feature_names = features[0..2].to_vec();
    let ds_slice: Array2D = ds.records.slice(s![..,0..2]);
    kmeans_on_features_subset(&feature_names, ds_slice);

    // let's reduce to two dimensions
    let feature_names = features[1..3].to_vec();
    let ds_slice: Array2D = ds.records.slice(s![..,1..3]);
    kmeans_on_features_subset(&feature_names, ds_slice);
    

    let feature_names = features[2..4].to_vec();
    let ds_slice: Array2D = ds.records.slice(s![..,2..4]);
    kmeans_on_features_subset(&feature_names, ds_slice);
    

    let feature_names = features[0..3].iter().step_by(2).cloned().collect();
    let ds_slice: Array2D = ds.records.slice(s![..,0..3;2]);
    kmeans_on_features_subset(&feature_names, ds_slice);
}
