
use linfa::prelude::*;
use ndarray::prelude::*;
use ndarray::Array1;
use linfa_logistic::{MultiLogisticRegression, MultiFittedLogisticRegression};
use linfa::metrics::ConfusionMatrix;
use rand::prelude::*;



/// Loads the Iris dataset from a CSV file and returns a linfa Dataset
pub fn load_iris_dataset(split_ratio: f32) -> (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>){

    // let mut reader = ReaderBuilder::new()
    //     .has_headers(false)
    //     .from_path(path)?;
    let mut rng = thread_rng();
    
    let (train, test): (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>) = linfa_datasets::iris().shuffle(&mut rng)
        .with_feature_names(vec!["sepal length", "sepal width", "petal length", "petal width"])
        .split_with_ratio(split_ratio);
    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train.nsamples()
    );
    println!("Feature names {:?}", train.feature_names());
    println!("Dataset records shape: {:?}", train.records.shape());
    println!("Dataset targets shape: {:?}", train.targets.shape());
    (train,test)
}


fn fit_logistic_regressor(train_set: &Dataset<f64, usize, Ix1>) -> MultiFittedLogisticRegression<f64, usize> {
    

    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train_set.nsamples()
    );

    // fit a Logistic regression model with 150 max iterations
    let model = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(train_set)
        .unwrap();

    model

}

fn predict_class(test_set: &Dataset<f64, usize, Ix1>, model: MultiFittedLogisticRegression<f64, usize>) 
    -> (Array1<usize>, ConfusionMatrix<usize>) {
    println!(
        "Predict class of #{} testing points",
        test_set.nsamples()
    );

    let pred = model.predict(test_set);
    let cm = pred.confusion_matrix(test_set).unwrap();
    (pred, cm)
}

fn main(){
    let (train_set, test_set) = load_iris_dataset(0.9);
    let corr_matrix = train_set.pearson_correlation();
    println!("Pearson correlation matrix of training features");
    println!("{}", corr_matrix);
    let model = fit_logistic_regressor(&train_set);

    let (prediction, cm) = predict_class(&test_set, model);

    let n_samples_test = test_set.nsamples();
    println!("Predictions: {:?}", prediction.slice(s![0..n_samples_test]));
    println!("Ground truth: {:?}", test_set.targets.slice(s![0..n_samples_test])); 

    println!("Confusion matrix {:?}", cm);
    
}