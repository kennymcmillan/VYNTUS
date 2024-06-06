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