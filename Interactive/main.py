import streamlit as st

import numpy as np

from utility import read_data, correlation
from modules import heatmap, only_numerical, correlation_heatmap, filtered_correlation, filtered_correlation_comparison, filtered_correlation_comparison_automatic

# annoyingly many warnings
import warnings
warnings.filterwarnings("ignore")

CLINICAL_REPORTS_PATH = "../normal_cohort_echo_reports_and_clinical_variables.csv"
ECHO_REPORTS_PATH = "../preprocessed_normal_cohort_echo_reports.csv"
CONCRETE_CORRELATION_CUTOFF = 0.1


# Top level configurations
st.set_page_config(layout="wide")


echo_reports = read_data(path=ECHO_REPORTS_PATH, only_numerical=False, echo_report=True)
clinical_reports = read_data(path=CLINICAL_REPORTS_PATH, only_numerical=False, echo_report=False)


st.title('Heart Echo Reports - Data Analysis')

st.title('Echo Reports')
st.write('Echo report contains categorical variables like normal/not normal. The problem is that those columns are extremely skewed with very few not normal values which makes correlation results unreliable. Select below if you want to include those string columns.')
filter_string_columns = only_numerical(key="filter1")

# Correlation without any modifications
st.header('Correlation')
st.subheader('Correlation between all columns')
correlation_heatmap(echo_reports, filter_string=False)


# Filtered Correlation
st.title('Filtered Correlation')
filtered_correlation(echo_reports, filter_string=filter_string_columns)


# Correlation comparison
st.title("Filtered Correlation comparison")
st.write("Select two ranges")
filtered_correlation_comparison(echo_reports, filter_string=filter_string_columns)

st.title("Correlation Comparison Automatic")
filtered_correlation_comparison_automatic(echo_reports, key="filtered_correlation_comparison_atuomatic_echo")







st.title('Echo Reports with Clinical Variables')
st.write("For most patients that got an echo report we also have some clinical variable measurements. Which measurements were taken differ from patient to patient, it is however possible to join the clinical variables to the echo report data using a timeframe. Below a timeframe of 1 hour was used")
# Correlation without any modifications
st.header('Correlation')
st.subheader('Correlation between all columns')
correlation_heatmap(clinical_reports, filter_string=False)


# Filtered Correlation
st.title('Filtered Correlation with Clinical Variables')
filtered_correlation(clinical_reports, filter_string=filter_string_columns, key="filtered_correlation_clinical")


# Correlation comparison
st.title("Filtered Correlation comparison with Clinical Variables")
st.write("Select two ranges")
filtered_correlation_comparison(clinical_reports, filter_string=filter_string_columns, key="filtered_correlation_comparison_clinical")


st.title("Correlation Comparison Automatic")
filtered_correlation_comparison_automatic(clinical_reports, key="filtered_correlation_comparison_atuomatic_clinical")