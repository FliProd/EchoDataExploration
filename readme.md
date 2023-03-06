# Echo Report and Clinical Variable Analysis
This repository contains code which analyses Echo report Data as well as Clinical Variables. 

## Data
The echocardiography report dataset and the clinical variable datasets can be found on Leomed as .csv.
- Echocardiography reports: cluster/work/shfn/data/unified/NORMAL/preprocessed_normal_cohort_echo_reports.csv
- Clinical Variables: /cluster/work/shfn/data/unified/NORMAL/clinical_var/all_clinical_vars_normal_cohort_LR.csv

## Preprocessing
The folder Analysis/preprocessing contains a notebook to process the unprocessed echocardiography reports .csv file from above. All other notebooks assume the content to be in this format. The clinical variables .csv does not have to be preprocessed in any way.


## Interactive
The folder Interactive contains a streamlit web application for exploring correlation between echocardiography variables as well as clinical variables. Note that there are some performance issues and a simple hack is to just comment out sections currently not needed in Interactive/main.py.

## Analysis
Contains 5 foldes (apart from preprocessing) each handling one analysis.

### Correlation
The folder contains a notebook to generate the different correlation analyses as well as some precomputed correlation tables as csv.

### Prediction
The folder prediction_experiment contains a notebook to generate the different predictions. There are also a lot of .csv encoded with information in the variable name:
- obv_deps means the obvious dependendencies are kept in the input
- no_obv_deps means the obvious dependencies are removed from the input
- subsampled means we subsample the majority class while the absence of it means we upsample the minority class
- LR is for Lasso Regression and RF for Random Forest regression
- poly means polynomial feature transformation was applied
- echo_reports means only echo report variables were predicted
- clinical_variables means clinical variables were predicted
The _predictabilty_ files contain detailed information about the configuraations performance for all variables while the compaarison compares the different models (RF/LR and poly/no poly) unde the configuration. the folders predictions, roc_plots and confusion_matrices contain more info for each model and variable encoded similarly as above


### Clustering Experiment
The folder clustering_experiment contains multiple clustering experiments.
- pyproclus: contains the code for the proclus experiment in experiment.py
- clique: contains the code for running the clique experiment on leomed, simply execute: sbatch run.sh
- clustering_naive: contains code that naively tries to find good subspaces by just randomly trying or with a greedy strategy
- dimension_reduced_clustering contains code that sequentially performs dimension reduction and clustering

### Clinical Variables
Contains two notebooks:
- all_clin_vars_analysis.ipynb: contains code that joins the echo report and clinical variable dataset as well as a correlation analysis of the clinical variables
- all_clin_vars_initial_analysis.ipynb: contains code that investigates the relation between clinical variables and echo report patients

### Validating Ranges
Contains a notebook that can be used to validate the ranges in the echo report against an external source.


