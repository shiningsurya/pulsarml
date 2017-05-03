# Pulsar ML
## Paper submiited to MNRAS

There are four Machine Learning models discussed here:

1. Adaboost
2. Gradient Boosting Classifier(GBC)
3. Neural Network
4. eXtreme Gradient BOOSTing(XGBoost)

These four models are trained in four iPyNB files present in the repo.

Each of the ipynb files contains separate sections. In each of them, only the first section concerns with training. Remaining all present confusion matrices, feature importances(in case of tree based methods), False Positive Rate analysis, area analysis and metric scores. 

I've written my some helper functions to plot the confusion matrices, feature importances and computing areas. I've relied heavily on the following Python Modules: 

1. NumPy(bread and butter of scientific Python)
2. Scikit-learn(one stop Python machine learning)
3. XGBoost(For Python implementation)
4. SKNN(scikit-learn neural network)
5. imbalance(SMOTE)
6. SciPy(for numerical integration)


### Dataset

The entire dataset is uploaded in the `all_in_one.csv` file. This is the result of generating features from Pulsar Candidate file as outputted by PulsarHunter.

### Running

Ensure that all the python files and the dataset file(`all_in_one.csv`) are in the same directory as the `ipynb` files. Open the `ipynb` file. 


