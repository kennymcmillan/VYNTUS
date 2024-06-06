import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pyperclip

# Function to transform the dataset
def transform_dataset(df):
    df = df.iloc[1:]  # Remove the first row which contains units
    df.columns = df.columns.str.replace("'", "").str.replace("\\", "").str.replace(" ", "_").str.replace("/", "_")
    df = df[df['Time'] != 'min']
    df = df.drop(columns=['O2pulse', 'BR_FEV%'])

    columns_to_replace_zeros = ['VE', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
    df[columns_to_replace_zeros] = df[columns_to_replace_zeros].replace(0, np.nan)

    columns_to_convert = ['HR', 'VE', 'VO2', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    def convert_time_to_seconds(time_str):
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except:
            return np.nan

    df['Time_sec'] = df['Time'].apply(convert_time_to_seconds)

    for column in columns_to_convert:
        df[f'{column}_30s_avg'] = df[column].rolling(window=30, min_periods=1, center=True).mean()

    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    cutoff = 0.033
    fs = 1

    for column in columns_to_convert:
        valid_data = df.dropna(subset=[column, 'Time_sec'])
        if len(valid_data) > 3 * 4:
            df[f'{column}_butter'] = np.nan
            df.loc[valid_data.index, f'{column}_butter'] = butter_lowpass_filter(valid_data[column], cutoff, fs)

    return df

# Function to plot the filtered data
def plot_filtered_data(df, metric):
    fig, ax = plt.subplots()
    ax.plot(df['Time_sec'], df[metric], label=f"Original {metric}", color='gray', alpha=0.5)
    ax.plot(df['Time_sec'], df[f'{metric}_30s_avg'], label=f"30s Moving Average {metric}", color='blue')
    if f'{metric}_butter' in df.columns:
        ax.plot(df['Time_sec'], df[f'{metric}_butter'], label=f"Butterworth Filtered {metric}", color='red')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(metric)
    ax.set_title(f"Original and Smoothed {metric} over Time")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def calculate_5s_average(df):
    columns_to_average = ['HR', 'VE', 'VO2', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
    df_5s_avg = df.copy()
    
    for column in columns_to_average:
        df_5s_avg[column] = df[column].rolling(window=5, min_periods=1).mean()
        
    return df_5s_avg

st.set_page_config(layout="wide")  # Set the layout to wide

# Add CSS to reduce left margin and top padding
st.markdown(
    """
    <style>
        /* Reduce padding and margin at the top of the sidebar */
        .css-1d391kg { 
            padding-top: 0rem;
            margin-top: 0rem;
        }
        
        .appview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        .sidebar .sidebar-content {
            width: 100px;
        }

        .spacer {
            margin-bottom: 2rem;
        }

        .small-input-box .stNumberInput div div input {
            padding: 2px !important;
            width: 70px !important;
            height: 25px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for file upload and download
with st.sidebar:
    st.header("Upload and Download")

    # Input for Name
    name = st.text_input("Enter Name:")

    # Date picker for Date of Test
    test_date = st.date_input("Date of Test:")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Transform the dataset (Assuming transform_dataset is defined elsewhere)
        transformed_df = transform_dataset(df)

        # Add Name and Test Date columns
        transformed_df.insert(0, 'Name', name)
        transformed_df.insert(1, 'TestDate', test_date)

        # Store transformed data in session state
        st.session_state.transformed_df = transformed_df

        # Format the file name using Name and Test Date
        formatted_date = test_date.strftime("%Y-%m-%d")
        file_name = f"{name}_{formatted_date}_transformed.csv"

        # Download the transformed dataset
        st.download_button(
            label="Download Transformed CSV",
            data=transformed_df.to_csv(index=False).encode('utf-8'),
            file_name=file_name,
            mime='text/csv'
        )

    st.markdown("### Submaximal Stage Levels")

    # Initialize session state for stop minutes and seconds if it doesn't exist
    if "stop_minutes" not in st.session_state:
        st.session_state.stop_minutes = [0] * 10
    if "stop_seconds" not in st.session_state:
        st.session_state.stop_seconds = [0] * 10

    stop_minutes = st.session_state.stop_minutes
    stop_seconds = st.session_state.stop_seconds

    rerun = False
    for i in range(10):
        col1, col2, col3 = st.columns([1, 3, 3])
        with col1:
            st.write(f"S {i + 1}")
        with col2:
            new_min = st.number_input(f"Stop Minute {i + 1}", min_value=0, max_value=60, value=stop_minutes[i], step=1, key=f"stop_min_{i}", format="%d")
        with col3:
            new_sec = st.number_input(f"Stop Seconds {i + 1}", min_value=0, max_value=59, value=stop_seconds[i], step=1, key=f"stop_sec_{i}", format="%d")

        if new_min != stop_minutes[i] or new_sec != stop_seconds[i]:
            st.session_state.stop_minutes[i] = new_min
            st.session_state.stop_seconds[i] = new_sec
            rerun = True

    if rerun:
        st.experimental_rerun()

def create_stage_times_df(metric, time_secs_list, transformed_df):
    def get_avg_for_interval(metric, start_time, end_time):
        interval_df = transformed_df[(transformed_df['Time_sec'] >= end_time) & (transformed_df['Time_sec'] <= start_time)]
        if not interval_df.empty:
            return interval_df[f'{metric}_30s_avg'].mean()
        else:
            return np.nan

    stage_times = []
    for i, time_sec in enumerate(time_secs_list):
        closest_time = transformed_df.iloc[(transformed_df['Time_sec'] - time_sec).abs().argmin()]['Time_sec']
        metric_value = transformed_df.iloc[(transformed_df['Time_sec'] - time_sec).abs().argmin()][f'{metric}_30s_avg']
        stage_info = {
            "Stage": i + 1,
            "Time_secs": time_sec,
            metric: f"{metric_value:.1f}" if not np.isnan(metric_value) else np.nan,
            "-5": f"{get_avg_for_interval(metric, time_sec, time_sec - 5):.1f}" if not np.isnan(get_avg_for_interval(metric, time_sec, time_sec - 5)) else np.nan,
            "-10": f"{get_avg_for_interval(metric, time_sec - 5, time_sec - 10):.1f}" if not np.isnan(get_avg_for_interval(metric, time_sec - 5, time_sec - 10)) else np.nan,
            "-15": f"{get_avg_for_interval(metric, time_sec - 10, time_sec - 15):.1f}" if not np.isnan(get_avg_for_interval(metric, time_sec - 10, time_sec - 15)) else np.nan,
            "-20": f"{get_avg_for_interval(metric, time_sec - 15, time_sec - 20):.1f}" if not np.isnan(get_avg_for_interval(metric, time_sec - 15, time_sec - 20)) else np.nan,
            "-25": f"{get_avg_for_interval(metric, time_sec - 20, time_sec - 25):.1f}" if not np.isnan(get_avg_for_interval(metric, time_sec - 20, time_sec - 25)) else np.nan,
            "-30": f"{get_avg_for_interval(metric, time_sec - 25, time_sec - 30):.1f}" if not np.isnan(get_avg_for_interval(metric, time_sec - 25, time_sec - 30)) else np.nan
        }
        avg_metric = np.nanmean([get_avg_for_interval(metric, time_sec, time_sec - 5), get_avg_for_interval(metric, time_sec - 5, time_sec - 10), get_avg_for_interval(metric, time_sec - 10, time_sec - 15), get_avg_for_interval(metric, time_sec - 15, time_sec - 20), get_avg_for_interval(metric, time_sec - 20, time_sec - 25), get_avg_for_interval(metric, time_sec - 25, time_sec - 30)])
        stage_info["Average"] = f"{avg_metric:.1f}" if not np.isnan(avg_metric) else np.nan
        stage_times.append(stage_info)
    return pd.DataFrame(stage_times)

if uploaded_file is not None:
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = "Show Transformed Data"

    selected_tab = st.selectbox(" ", ["Show Transformed Data", "Transformed Graph", "Analysis", "Summary for Smartabase"], index=0 if st.session_state.selected_tab == "Show Transformed Data" else 1 if st.session_state.selected_tab == "Transformed Graph" else 2 if st.session_state.selected_tab == "Analysis" else 3)
    if selected_tab != st.session_state.selected_tab:
        st.session_state.selected_tab = selected_tab
        st.experimental_rerun()

    if st.session_state.selected_tab == "Show Transformed Data":

        if "transformed_df" in st.session_state:
            st.dataframe(st.session_state.transformed_df)

    elif st.session_state.selected_tab == "Transformed Graph":

        if "transformed_df" in st.session_state:

            metrics_to_plot = ['VO2', 'VE', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
            metric = st.selectbox("Select a metric to plot", metrics_to_plot)
            plot_filtered_data(st.session_state.transformed_df, metric)

    elif st.session_state.selected_tab == "Analysis":

        transformed_df = st.session_state.transformed_df
        time_secs_list = [
            st.session_state.stop_minutes[i] * 60 + st.session_state.stop_seconds[i]
            for i in range(10)
            if st.session_state.stop_minutes[i] > 0 or st.session_state.stop_seconds[i] > 0
        ]

        metrics = ['VE', 'VO2', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
        stage_dfs = {}
        for metric in metrics:
            st.markdown(f"##### {metric}")
            stage_times_df = create_stage_times_df(metric, time_secs_list, transformed_df)
            st.table(stage_times_df)
            stage_dfs[metric] = stage_times_df

    elif st.session_state.selected_tab == "Summary for Smartabase":

        if "transformed_df" in st.session_state:
            transformed_df = st.session_state.transformed_df

            time_secs_list = [
                st.session_state.stop_minutes[i] * 60 + st.session_state.stop_seconds[i]
                for i in range(10)
                if st.session_state.stop_minutes[i] > 0 or st.session_state.stop_seconds[i] > 0
            ]

            metrics = ['VE', 'VO2', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
            summary_data = []
            for i, time_sec in enumerate(time_secs_list):
                row = {"Stage": i + 1, "Time": time_sec}
                for metric in metrics:
                    stage_times_df = create_stage_times_df(metric, time_secs_list, transformed_df)
                    row[metric] = stage_times_df.loc[stage_times_df['Stage'] == i + 1, 'Average'].values[0]
                summary_data.append(row)

            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)

            # Button to copy the table data to clipboard
            if st.button("Copy Table to Clipboard"):
                summary_csv = summary_df.to_csv(index=False, sep='\t')
                pyperclip.copy(summary_csv)
                st.success("Table copied to clipboard")
