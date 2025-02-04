
use linfa::prelude::*;
use std::error::Error;
use ndarray::prelude::*;
use csv::ReaderBuilder;

#[derive(Debug)]
struct IrisData {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class_plant: String
}

#[derive(Debug)]
enum IrisType{
    Iris_Setosa,
    Iris_Versicolour,
    Iris_Virginica
}

/// Loads the Iris dataset from a CSV file and returns a linfa Dataset
pub fn load_iris_dataset(path: &str) -> Result<Dataset<f64, String>, Box<dyn Error>> {

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<String> = Vec::new();


    for result in reader.records() {
        let record = result?;
        

        let iris = IrisData {
            sepal_length: record[0].parse::<f64>()?,
            sepal_width: record[1].parse::<f64>()?,
            petal_length: record[2].parse::<f64>()?,
            petal_width: record[3].parse::<f64>()?,
            class_plant: record[4].to_string(),
        };


        features.push(vec![
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width,
        ]);
        targets.push(iris.class_plant);
    }

    // array 2 is a two-dimensional array
    let feature_array = Array2::from_shape_vec(
        (features.len(), 4),
        features.into_iter().flatten().collect(),
    )?;

    let target_array = Array2::from_shape_vec(
        (targets.len(), 1),
        targets.clone(),
    )?;

    // Create and return the Dataset
    Ok(Dataset::new(feature_array, target_array).with_feature_names(vec!["sepal_length", "sepal_width", "petal_length", "petal_width"]))
}

// Example usage:
fn main() -> Result<(), Box<dyn Error>> {
    let dataset = load_iris_dataset("data/iris.data")?;
    println!("Dataset shape: {:?}", dataset.records.shape());
    println!("Dataset first rows: {:?}", dataset.records.slice(s![0..4, ..]));
    println!("Dataset first targets: {:?}", dataset.targets.slice(s![0..4, ..]));
    Ok(())
}