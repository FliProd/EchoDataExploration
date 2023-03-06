import pandas as pd
import pickle

from pyclustering.cluster.clique import clique
 

preprocessed_normal_cohort_csv_path = "/cluster/work/shfn/data/unified/NORMAL/preprocessed_normal_cohort_echo_reports.csv"
preprocessed_normal_cohort_csv_path = "../../../../preprocessed_normal_cohort_echo_reports.csv"
df = pd.read_csv(preprocessed_normal_cohort_csv_path)
df = df.drop(['Unnamed: 0', 'Pulmonalklappe', 'Ebene'], axis=1)

#impute nan values with mode
for column in df.columns:
    series = df[column]
    if series.dtype == 'object':
        df[column] = series.fillna(series.mode()[0])
    else:
        df[column] = series.fillna(series.median())

# drop the single aorta replced patient
df = df.drop(df[df['Aorta'] == 'Ersetzt'].index)


# map strings to categorical
for column in df.columns:
    series = df[column]
    if series.dtype == 'object':
        df[column] = pd.Categorical(series)
        df[column] = df[column].cat.codes
        df[column] = df[column].astype('category')


# convert to list of lists
data = df.values.tolist()

clique_instance = clique(data, density_threshold=0.01, amount_intervals=1)
 
# Run cluster analysis.
clique_instance.process()
 
# Get allocated clusters.
clusters = clique_instance.get_clusters()

cluster_encoding = clique_instance.get_cluster_encoding()

# pickle restuls and cluster_instance
with open('results.pickle', 'wb') as f:
    pickle.dump({"intance":clique_instance, "cluster_assignments":clusters, "cluster_encodings":cluster_encoding}, f)
