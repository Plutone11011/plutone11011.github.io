---
layout: custom_post
title: "Rust: logistic regression"
date: 2025-02-09 10:09:00 +0100
categories: logregression
#permalink: /seq2seq
#categories: encoder-decoder keras
#permalink: CoRec
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Introduction

Logistic regression is a linear method for classification that models probabilistic outcomes with a sigmoid function, which maps the input to a $$[0,1]$$ range

$$
f(x) = \frac{1}{1 + e^-x}
$$

!["Sigmoid here"](../../../../assets/sigmoid.png)
Mathematically, the model predicts the conditional probability for the i-th sample $$p(y_i = 1 | X_i)$$, by minimizing the cost function

$$
-y_ilog(p(X_i)) - (1 - y_i)log(1 - p(X_i))
$$


Typically, logistic regression is applied to binary classification, while variations of this method, like multinomial logistic regression, or One-vs-Rest logistic regression, can be used for multiclass problems. 

Here, we are going to implement logistic regression using linfa, a Rust ML library that follows similar principles to scikit-learn. We are using the well-known Iris Dataset for this, which is configured as a multiclass problem
where the task is to determine whether a certain flower is one of Iris setosa, Iris versicolor, or Iris virginica. 

# Solution

The whole code example is publically available [here](https://github.com/Plutone11011/plutone11011.github.io).
First off, we have to load the iris dataset. For the purposes of this example, we will partition the dataset into a train and test split, after shuffling, in order to avoid an uneven spread of classes between train and test samples.


```rust
pub fn load_iris_dataset(split_ratio: f32) -> (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>){

    let mut rng = thread_rng();

    let feature_names = linfa_datasets::iris().feature_names();
    let (train, test): (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>) = linfa_datasets::iris().shuffle(&mut rng)
        .with_feature_names(feature_names)
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
```

Before proceeding with fitting a logistic regressor on this data, we want to examine the relative importance of each 4 features of the dataset. To achieve this, we
compute a correlation matrix, which contains the correlation coefficients of each pair of features. Correlation coefficients are values between $$[-1,1]$$, with 1 being
perfect positive correlation (or increasing linear relationship) between features, while -1 indicates a perfect inverse linear relationship.

Linfa offers a way to compute the pearson correlation coefficients, however it returns a 1-dimensional ndarray, so we then populate a symmetric matrix to better show it with the *plotters* library

```rust
let corr_matrix = train_set.pearson_correlation();
let feature_names = train_set.feature_names();
let corr_coefficients = corr_matrix.get_coeffs();

let n_features = feature_names.len();
let mut matrix_of_coeff = Array2::<f32>::zeros((n_features, n_features));
let mut k = 0;
for i in 0..n_features {
    for j in i+1..n_features {
           
        matrix_of_coeff[[i, j]] = corr_coefficients[k] as f32;
        matrix_of_coeff[[j, i]] = corr_coefficients[k] as f32;
        k += 1;
            
    }
}

matrix_of_coeff
```

The correlation matrix was drawn with plotters, the code is available in the repo linked [here](https://github.com/Plutone11011/plutone11011.github.io)

!["Correlation matrix here"](../../../../assets/corr_matrix.jpg)

This correlation matrix shows that, for example, that petal length is highly positively correlated with sepal length and petal width, while on the contrary it has a smaller negative correlation with sepal width.
This information is useful for feature selection: whenever we have too many features, such that it could be more efficient to prune a few of them, we might want to eliminate the features that are positively correlated with one another, in order to avoid redundancy and keep the model more simple.


Finally, we can create the model and fit. This is a simple problem, so it does not require specific attention to certain hyperparameters.


```rust
let model = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(train_set)
        .unwrap();


```

Then, we use the fitted model to predict the classes of the test set.

```rust
let pred = model.predict(test_set);
let cm = pred.confusion_matrix(test_set).unwrap();

Confusion matrix
classes    | 1          | 2          | 0
1          | 2          | 1          | 0
2          | 0          | 3          | 0
0          | 0          | 0          | 9
```

Confusion matrix shows on the rows the target classes, while on the columns the predicted classes. Thus, on the diagonal we have the number of correct predictions, while the other elements show wrong predictions.

# Conclusion
Logistic regression is a basic method for classification, which can be useful for tabular data like the Iris Dataset, as we can see from the results, where only a single sample was misclassified.

