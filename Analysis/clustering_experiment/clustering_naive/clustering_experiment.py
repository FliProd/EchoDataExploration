import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import json
import multiprocessing as mp
import array

from sklearn.cluster import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_mutual_info_score, calinski_harabasz_score 
from sklearn.preprocessing import normalize


from experiment import score_clustering_algo
"""
Utility functions
"""

# serializes array of sklearn objecs
def serialize(array):
    serialized = []
    for object in array:
        serialized.append(str(object))
    return serialized


def __random_subset(size, max_val, column_names, random_state):
    subset = []
    while len(subset) != size:
        np.random.seed(random_state)
        new_variable = np.random.randint(low=0,high=max_val)
        if new_variable not in subset:
            subset.append(new_variable)
    return [column_names[i] for i in subset]


def __total_correlation(subset, correlation_table):
    total_correlation = 0
    for i in range(len(subset)):
        for j in range(i+1,len(subset)):
            total_correlation += correlation_table[subset[i]][subset[j]]
    return total_correlation


def generate_random_subsets(subset_size, num_subsets, correlation_table, total_correlation_cutoff, random_state):
    subsets = []
    while len(subsets) < num_subsets:
        new_subset = __random_subset(subset_size, correlation_table.shape[0], correlation_table.columns, random_state)
        if __total_correlation(new_subset, correlation_table) < total_correlation_cutoff:
            subsets.append(new_subset)
    return subsets


def generate_all_subsets(subset_size, correlation_table, total_correlation_cutoff, upperbound):
    subsets = []
    for i in range(0, correlation_table.shape[0]):
        for j in range(i+1, correlation_table.shape[0]):
            for u in range(j+1, correlation_table.shape[0]):
                for v in range(u+1, correlation_table.shape[0]):
                    new_subset = [i,j,u,v]
                    new_subset = [correlation_table.columns[i] for i in new_subset]
                    if upperbound and __total_correlation(new_subset, correlation_table) < total_correlation_cutoff:
                        subsets.append(new_subset)
                    elif __total_correlation(new_subset, correlation_table) > total_correlation_cutoff:
                        subsets.append(new_subset)
    return subsets




"""
Preprocessing
"""


preprocessed_normal_cohort_csv_path = "../../../preprocessed_normal_cohort_echo_reports.csv"
df = pd.read_csv(preprocessed_normal_cohort_csv_path)

df_strings = df.select_dtypes('object')

df_filtered = df.drop(columns=['Pulmonalklappe', 'Ebene'])
df_strings = df_filtered.select_dtypes('object')
df_numerical = df.select_dtypes('number')

normal_map = lambda s: 0 if s == "nicht normal" else 1
extended_normal_map = lambda s: 0 if s == "Ersetzt" else 1 if s == "nicht normal" else 2
gut_map = lambda s: 0 if s == "genügend" else 1

value_mappers = {
    "Echoqualitaet": gut_map,
    "Mitralklappe": normal_map,
    "Aortenklappe": normal_map,
    "Aorta": extended_normal_map,
    "Trikuspidalklappe": normal_map,
    "VenaCavaInferior": normal_map,
    "RechterVorhof": normal_map,
    "RechterVentrikel": normal_map,
    "LinkerVorhof": normal_map,
    "LinkerVentrikel": normal_map,
    "RegionaleWandbewegungen": normal_map,
    "LinkerVorhof": normal_map
}

string_columns = df_strings.columns
df_mapped = df_filtered.copy()
df_mapped[string_columns] = df_strings.apply(lambda series: series.map(value_mappers[series.name]))


"""
Correlation tables for subset selection
"""
correlation_table = df_mapped.corr()
numerical_correlation_table = df_numerical.corr()




"""
Actual Experiment
"""

if __name__ == '__main__':

    num_clusters = 3
    random_state = 42
    df = df_numerical


    variable_subset_size = 4
    num_variable_subsets = 1
    total_correlation_cutoff = 2.5
    upperbound = False


    variable_subsets = generate_all_subsets(
        subset_size=variable_subset_size,
        total_correlation_cutoff=total_correlation_cutoff,
        correlation_table=df.corr(),
        upperbound=upperbound # generate subsets with correlation higher than the cutoff
    )

    condition = "lower" if upperbound else "higher"
    print("{} subsets with total correlation {} than {}".format(len(variable_subsets), condition, total_correlation_cutoff))

    algorithms = [
        KMeans(n_clusters=num_clusters, random_state=random_state),
        AffinityPropagation(max_iter=300, random_state=random_state),
        AgglomerativeClustering(n_clusters=num_clusters),
        Birch(threshold=0.05),
        #DBSCAN(),
        #FeatureAgglomeration(n_clusters=num_clusters),
        MiniBatchKMeans(n_clusters=num_clusters, random_state=random_state),
        MeanShift(),
        OPTICS(),
        SpectralClustering(n_clusters=num_clusters, random_state=random_state),
    ]

    imputation_methods = [
        'iterative',
    #    'KNN',
    #    'mode',
        'mean',
    #    'median'
    ]

    measures = [
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score
    ]

    config = {
        'subset_size'
        'variable_subsets': variable_subsets,
        'df': 'numerical',
        'num_clusters':  num_clusters,
        'random_state': random_state,
        'algorithms': serialize(algorithms),
        'imputation_methods': imputation_methods,
        'measures': serialize(measures)
    }

    num_algos = len(algorithms)
    num_subsets = len(variable_subsets)
    num_imputation_methods = len(imputation_methods)
    scores = np.zeros((num_algos,
                        num_subsets,
                        num_imputation_methods, 
                        len(measures)))
    models = []
    



    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=2))
    pool = mp.Pool(processes=n_cpus)

    # apply function in parallel
    results = [pool.apply_async(score_clustering_algo, 
                                args=(algo, variable_subset, imputation_method, measures, df,)) 
                for imputation_idx, imputation_method in enumerate(imputation_methods)
                for subset_idx, variable_subset in enumerate(variable_subsets) 
                for algo_idx, algo in enumerate(algorithms)]

    for algo_idx in range(num_algos):
        models.append([])
        for subset_idx in range(num_subsets):
            models[algo_idx].append([])
            for imputation_idx in range(num_imputation_methods):
                scores[algo_idx, subset_idx, imputation_idx], model = results[algo_idx*(num_subsets*num_imputation_methods) + subset_idx*num_imputation_methods + imputation_idx].get()
                models[algo_idx][subset_idx].append(model)

    results_path = "{}/results".format(os.getcwd())
            
    old_results = [int(dir) for dir in os.listdir(results_path)]
    current_result_folder_name = len(old_results) + 1
    current_result_folder_path = "{}/{}/".format(results_path, current_result_folder_name)
    os.mkdir(current_result_folder_path)

    # store processed df in folder
    with open("{}/{}".format(current_result_folder_path, 'df.pickle'), 'wb') as outfile:
        pickle.dump(df, outfile)

    # store results in folder
    with open("{}/{}".format(current_result_folder_path, 'scores.pickle'), 'wb') as outfile:
        pickle.dump(scores, outfile)

    # store models in folder
    print(models)
    with open("{}/{}".format(current_result_folder_path, 'models.pickle'), 'wb') as outfile:
        pickle.dump(models, outfile)

    # store configuration in folder
    with open("{}/{}".format(current_result_folder_path, 'configuration.json'), 'w') as file:
        json.dump(config, file, indent=4)
    
