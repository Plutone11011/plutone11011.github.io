use std::error::Error;

use linfa::prelude::*;
use ndarray::prelude::*;
use ndarray::Array1;
use linfa_logistic::{MultiLogisticRegression, MultiFittedLogisticRegression};
use linfa::metrics::ConfusionMatrix;
use rand::prelude::*;
use plotters::prelude::*;


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

fn symm_corr_matrix(corr_matrix: &PearsonCorrelation<f64>, train_set: &Dataset<f64, usize, Ix1>) -> Array2<f32>{
    let feature_names = train_set.feature_names();
    let corr_coefficients = corr_matrix.get_coeffs();

    let n_features = feature_names.len();
    let mut matrix_of_coeff = Array2::<f32>::zeros((n_features, n_features));
    let mut k = 0;
    for i in 0..n_features {
        for j in i+1..n_features {
            if i == j{
                matrix_of_coeff[[i, j]] = 1.0;
            }
            else {
                matrix_of_coeff[[i, j]] = corr_coefficients[k] as f32;
                matrix_of_coeff[[j, i]] = corr_coefficients[k] as f32;
                k += 1;
            }
            
        }
    }

    matrix_of_coeff



    // Ok(())
}


fn draw_corr_matrix(sym_cor_matrix: &Array2<f32>) -> Result<(), Box<dyn Error>>{
    let drawing_area_width = 800;
    let drawing_area_height = 800;
    let root = BitMapBackend::new("corr_matrix.jpg", (drawing_area_width, drawing_area_height)).into_drawing_area();
    root.fill(&WHITE)?;

    let corr_matrix_dim = sym_cor_matrix.shape()[0];
    // create chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Cov Matrix", ("sans-serif", 60))
        .margin(10)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0i32..corr_matrix_dim as i32, corr_matrix_dim as i32..0i32)?;
    // definde chart attributes
    chart
        .configure_mesh()
        .x_labels(corr_matrix_dim)
        .y_labels(corr_matrix_dim)
        .x_label_offset(30)
        .y_label_offset(25)
        .label_style(("sans-serif", 20))
        .draw()?;
    
    let range = 0usize..corr_matrix_dim;
    // Get the plotting area dimensions
    let plotting_area = chart.plotting_area();
    let (width, height) = plotting_area.dim_in_pixel();
    // Calculate cell size
    let cell_width = width / corr_matrix_dim as u32;
    let cell_height = height / corr_matrix_dim as u32;
    println!("{} {}", cell_height, cell_width);
    chart.draw_series(
        range
            .clone()
            .flat_map(|row| {
                range.clone().map(move |column| {
                    (
                        row as i32,
                        column as i32,
                        sym_cor_matrix.get((row, column)).unwrap(),
                    )
                })
            })
            .map(|(x, y, v)| {
                // Map correlation values (-1 to 1) to RGB colors
                // Negative correlations -> blue
                // Zero correlation -> white
                // Positive correlations -> red
                let rgb = if *v < 0.0 {
                    // Negative values: scale from white (0,0) to blue (-1.0)
                    let intensity = (-*v).min(1.0);
                    RGBColor(
                        (255.0 * (1.0 - intensity)) as u8,
                        (255.0 * (1.0 - intensity)) as u8,
                        255,
                    )
                } else {
                    // Positive values: scale from white (0,0) to red (1.0)
                    let intensity = (*v).min(1.0);
                    RGBColor(
                        255,
                        (255.0 * (1.0 - intensity)) as u8,
                        (255.0 * (1.0 - intensity)) as u8,
                    )
                };

                EmptyElement::at((x, y))
                    + Rectangle::new(
                        [(0, 0), (cell_width as i32, cell_height as i32)],
                        rgb.filled(),
                    )
                    + Text::new(
                        format!("{:.3}", *v),
                        (10, 22),
                        ("sans-serif", 16.0).into_font(),
                    )
            }),
    )?;
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", "corr_matrix.jpg");

    Ok(())
}

fn main(){
    let (train_set, test_set) = load_iris_dataset(0.9);
    let corr_matrix = train_set.pearson_correlation();
    println!("Pearson correlation matrix of training features");
    println!("{}", corr_matrix);
    let _ = draw_corr_matrix(&symm_corr_matrix(&corr_matrix, &train_set));
    let model = fit_logistic_regressor(&train_set);

    let (prediction, cm) = predict_class(&test_set, model);

    let n_samples_test = test_set.nsamples();
    println!("Predictions: {:?}", prediction.slice(s![0..n_samples_test]));
    println!("Ground truth: {:?}", test_set.targets.slice(s![0..n_samples_test])); 

    println!("Confusion matrix {:?}", cm);
    
}