import streamlit as st
import pandas as pd


from utility import filter_df, correlation, correlation_differences




def only_numerical(key):
    return st.radio("Show", [False, True], format_func=lambda x: "Only numerical columns" if x else "All columns", key=key)

def heatmap(df):
    st.dataframe(df.style.format("{:.2}").background_gradient(cmap="bwr", vmin=-1, vmax=1), width=1200, height=600)


def correlation_heatmap(df, filter_string=False):
    full_correlation_table = correlation(df, only_numerical=filter_string)
    heatmap(full_correlation_table)

def interesting_correlation_differences(differences):
    st.write("Its always: Correlation from df restricted to first bin minus second bin")
    st.dataframe(differences.astype(str))



def filtered_correlation(df, filter_string=False, key="filtered_correlation"):
    st.subheader('Filter out rows for which the selected columns value isnt in the selected row.')

    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox('Variable to restrict:', df.columns, key=key + 'restriction_selector')
    with col2:
        bin = restrictor(df, variable, key=key + 'restrictor')

    filtered_correlation_table = correlation(df, only_numerical=filter_string, filter_var=variable, filter_val=bin)
    heatmap(filtered_correlation_table)

def filtered_correlation_comparison(df, filter_string=False, key="filtered_correlation_comparison"):
    col1, col2, col3 = st.columns(3)

    with col1:
        variable = st.selectbox('Variable to restrict:', df.columns, key=key + 'comparison_selector')
    with col2:
        bin_a = restrictor(df, variable, key=key + 'restrictor_a')
    with col3:
        bin_b = restrictor(df, variable, key=key + 'restrictor_b')
    

    filtered_correlation_table_a = correlation(df, only_numerical=filter_string, filter_var=variable, filter_val=bin_a)
    filtered_correlation_table_b = correlation(df, only_numerical=filter_string, filter_var=variable, filter_val=bin_b)

    st.write('Difference of the absolute correlation of the two filtered datasets: correleation range 1 - correlation range 2')
    st.write('A negative number means the correlation is higher in the second range than in the first range.')
    heatmap(filtered_correlation_table_a - filtered_correlation_table_b)


def filtered_correlation_comparison_automatic(df, key="filtered_correlation_comparison_automatic"):
    st.write("Give the number of bins and the correlation_change_threshold. \
        Then each variable gets partitioned into the number of equal sized bins (1/num_bins elements per bin) and for each restriction \
        the data gets filtered and the correlation table is computed. \
        For every variable and each combination of their restricted correlation tables \
        the differences are calculated and all differences above correlation_change_threshold is displayed in the table below.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_bins = st.number_input('num bins', min_value=2, max_value=10, step=1, key=key + "number_input1")

    with col2:
        min_patients = st.number_input('minimum #patients in bin', min_value=10, step=1, key=key + "number_input2")

    with col3:
        correlation_change_threshold = st.slider('correlation_change_thresholds', 0.0, 2.0, 1.0, key=key + "slider1")

    

    with col4:
        st.write('Only show differences in correlation where the restricted variable is numerical.')
        only_numerical_restrictions = st.checkbox("numerical only", key=key + "checkbox")

    differences = correlation_differences(df, num_bins, correlation_change_threshold, only_numerical_restrictions, min_patients)
    interesting_correlation_differences(differences)




# gives some options to restrict a column to
def restrictor(df, variable, key):
    if df[variable].dtype == 'object':
        variable_options = df[variable].unique()
        variable_options = [x for x in variable_options if str(x) != 'nan']

        bin = st.radio("value", variable_options, key=key)
    else:
        variable_min = df[variable].min()
        variable_max = df[variable].max()

        bin = st.slider('Select a range of values',variable_min, variable_max, (0.333*variable_max, 0.666*variable_max), key=key)
    
    st.write('{} patients in selection.'.format(filter_df(df, variable, bin, False).shape[0]))
    return bin



