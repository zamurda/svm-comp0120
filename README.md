# svm-comp0120

This is an implementation of the Support Vector Classifier, which uses the Sequential Minimal Optimisation (Platt, 1998) algorithm to optimise the dual-SVM formulation.
The implementation is very similar to Scikit-Learn, and is designed to be intuitive, use only numpy, and mimic SVMLight while offering slightly stronger theoretical convergence.

## Installation
The project can easily be installed using `poetry`

`poetry install`

To be able to run the experiment(s) and see for yourself how the model works, install the optional dependencies

`poetry install --with experiments`

## Usage

Instantiate an SVM class as `svm.SVM`

### Methods
- Call `fit()` on your data
- Call `predict()` on new data
- Call `decision_function()` to compute the decision for a given sample

### Attributes
Attribute naming conventions are very similar to sklearn:

`svm.SVM.support_vectors`
- A 2D array of support vector 

`svm.SVM.support_labels`
- An array of labels for each support vector

`svm.SVM.support_`
- An array of indices containing support vector locations in data matrix

`svm.SVM.dual_variables`
- The Lagrange multipliers for each sample in the training data
