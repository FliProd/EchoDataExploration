import pandas as pd


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
        "LinkerVorhof": normal_map,
}


def map_string_columns(df):
   
    df_string = df.select_dtypes(['object'])
    string_columns = list(value_mappers.keys())

    df_mapped = df.copy()
    df_mapped[string_columns] = df[string_columns].apply(lambda series: series.map(value_mappers[series.name]))
    return df_mapped




def filter_df(df, filter_var, filter_val, mapped_strings=True):
    string_columns = value_mappers.keys()
    
    if filter_var in string_columns:
        if mapped_strings:
            filter_val = value_mappers[filter_var](filter_val)
        filter_indices = df[filter_var] == filter_val
    else:
        filter_indices = df[filter_var].between(filter_val[0], filter_val[1])
    
    return df[filter_indices]


def read_data(path, only_numerical, echo_report):
    df = pd.read_csv(path, index_col=0)

    if echo_report:
        df = df.drop(columns=['Pulmonalklappe', 'Ebene'])
    else:
        df = df.drop(columns=['Pat_ResearchID', 'Pulmonalklappe', 'Ebene'])
    
    if only_numerical:
        df = df.select_dtypes('number')

    return df


def correlation(df, only_numerical,  filter_var=None, filter_val=None):
    if only_numerical:
        df = df.select_dtypes('number')
    
    if not filter_var is None:
        df = filter_df(df, filter_var, filter_val)
    
    correlation_table = case_correlation(df).round(2)

    return correlation_table


def case_correlation(df):
    columns = df.columns

    # create quadratic matrix with columns as index and columns as columns
    correlation_table = pd.DataFrame(index=columns, columns=columns)

    for var_a in columns:
        for var_b in columns:
            if df[var_a].dtype == 'object' or df[var_b].dtype == 'object':
                correlation_table[var_a][var_b] = df[var_a].corr(df[var_b], method='spearman')
            else:
                correlation_table[var_a][var_b] = df[var_a].corr(df[var_b], method='pearson')
    
    return correlation_table.astype("float")
 

def correlation_differences(df, num_bins, correlation_change_threshold, numerical_only, min_patients):
    if numerical_only:
        df = df.select_dtypes('number')
    
    variables = df.columns
    string_variables = df.select_dtypes('object')

    differences = []

    for variable in variables:
        
        series = df[variable]

        if variable in string_variables:
            bins = series.unique()
        else:
            _, bins = pd.qcut(series, num_bins, retbins=True, duplicates='drop')
            bins = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        

        correlation_tables = []

        for bin in bins:
            correlation_tables.append(correlation(df, only_numerical=False, filter_var=variable, filter_val=bin))
        
        for i in range(len(correlation_tables)):
            for j in range(i, len(correlation_tables)):
                correlation_change = correlation_tables[i] - correlation_tables[j]

                correlation_change = correlation_change.rename_axis(index='idx', columns='col')
                above_threshold_indices = correlation_change.stack().reset_index(name='value').query('abs(value) >= ' + str(correlation_change_threshold))

                for _, row in above_threshold_indices.iterrows():                    
                    num_patients_base = filter_df(df, filter_var=variable, filter_val=bins[i], mapped_strings=False).shape[0]
                    num_patients_subtractor = filter_df(df, filter_var=variable, filter_val=bins[j], mapped_strings=False).shape[0]
                    if num_patients_base > min_patients and num_patients_subtractor > min_patients:

                        differences.append({
                            'restricted_variable': variable,
                            'restriction_base': bins[i],
                            'patiens_in_base_restriction': num_patients_base,
                            'restriction_subtractor': bins[j],
                            'patiens_in_subtractor_restriction': num_patients_subtractor,
                            'variable_a': row['idx'],
                            'variable_b': row['col'],
                            'base_correlation': correlation_tables[i][row['idx']][row['col']],
                            'subtractor_correlation': correlation_tables[j][row['idx']][row['col']],
                            'difference':  row['value'],
                            'absolute difference': abs(row['value'])
                        })

    return pd.DataFrame.from_records(differences)
        
