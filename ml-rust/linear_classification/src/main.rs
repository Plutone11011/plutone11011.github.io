
use linfa::prelude::*;
use std::error::Error;
use ndarray::prelude::*;
use csv::ReaderBuilder;
use linfa_logistic::MultiLogisticRegression;

#[derive(Debug)]
struct IrisData {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class_plant: String
}


/// Loads the Iris dataset from a CSV file and returns a linfa Dataset
pub fn load_iris_dataset(split_ratio: f32) -> (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>){

    // let mut reader = ReaderBuilder::new()
    //     .has_headers(false)
    //     .from_path(path)?;

    let (train, test): (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>) = linfa_datasets::iris().split_with_ratio(split_ratio);
    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train.nsamples()
    );
    println!("Dataset records shape: {:?}", train.records.shape());
    println!("Dataset targets shape: {:?}", train.targets.shape());
    println!("Dataset first targets: {:?}", train.targets.slice(s![0..10]));
    (train,test)
}


fn main() -> Result<(), Box<dyn Error>> {
    let (train, test) = load_iris_dataset(0.9);

    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train.nsamples()
    );

    // fit a Logistic regression model with 150 max iterations
    let model = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(&train)
        .unwrap();

    println!(
        "Predict class of #{} testing points",
        test.nsamples()
    );
    // predict and map targets
    let pred = model.predict(&test);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&test).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}