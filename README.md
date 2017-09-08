# Pulsar ML
## Paper submitted to MNRAS

There are four Machine Learning models discussed here:

1. Artificial Neural Network(Multi Layer Perceptron)
2. Adaboost
3. Gradient Boosting Classifier(GBC)
4. eXtreme Gradient BOOSTing(XGBoost)

These four models are trained in four iPyNB files present in the repo.

Each of the ipynb files contains separate sections. In each of them, only the first section concerns with training. Remaining all present confusion matrices, feature importances(in case of tree based methods), False Positive Rate analysis, area analysis and metric scores. 

I've written my some helper functions to plot the confusion matrices, feature importances and computing areas. I relied heavily on the following Python Modules: 

1. NumPy(bread and butter of scientific Python)
2. Scikit-learn(one stop Python machine learning)
3. XGBoost(For Python implementation)
4. SKNN(scikit-learn neural network)
5. imbalance(SMOTE)
6. SciPy(for numerical integration)

### Installation and a small workaround
Installation for NumPy, SciPy, xgboost, imbalance and scikit-learn is straightforward, either using `pip` or `conda`. Installation of sknn is tricky since there is a version incompatibility. With `theano=0.8.2` only `scikit-learn neural-network=0.7` is compatible, so it is advised to install sknn=0.7 whereas 0.8 is the latest version. This is due to an update in `theano` which affects the `MultiLayerPerceptron` class of `sknn`.

### Dataset

The entire dataset is uploaded in the `all_in_one.csv` file. This is the result of generating features from Pulsar Candidate file as outputted by PulsarHunter. I've used `ExtractFeatures.py` script to extract features from `PHCX` files. 

### Running

Ensure that all the python files and the dataset file(`all_in_one.csv`) are in the same directory as the `ipynb` files. Open the `ipynb` file. 


