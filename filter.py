import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('FerrahZennidine.XLS.csv')


# Remove the first row which contains units
df = df.iloc[1:]

# Clean up the column names
df.columns = df.columns.str.replace("'", "").str.replace("\\", "").str.replace(" ", "_").str.replace("/", "_")
df = df[df['Time'] != 'min']
df = df.drop(columns=['O2pulse', 'BR_FEV%'])
df['VO2_kg'] = df['VO2_kg'].replace(0, np.nan)

# Check the data types of the columns
print("Data types before conversion:")
print(df.dtypes)

# Convert relevant columns to numeric types, forcing errors to NaN
columns_to_convert = ['HR', 'VE', 'VO2', 'VCO2', 'RER', 'VO2_kg', 'PETO2', 'PETCO2']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Convert Time column from mm:ss to seconds
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except:
        return np.nan

df['Time_sec'] = df['Time'].apply(convert_time_to_seconds)

# Check the data types of the columns after conversion
print("Data types after conversion:")
print(df.dtypes)

# Apply a 30-second moving average
df['VO2_30s_avg'] = df['VO2'].rolling(window=30, min_periods=1, center=True).mean()


# Apply a Butterworth digital filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Parameters for Butterworth filter
cutoff = 0.033  # Adjusting cutoff frequency for better smoothing
fs = 1  # Sampling frequency in Hz (1 sample per second)

# Drop NA values from VO2 and Time_sec for Butterworth filtering
valid_data = df.dropna(subset=['VO2', 'Time_sec'])

# Apply the Butterworth filter
valid_data['VO2_butter'] = butter_lowpass_filter(valid_data['VO2'], cutoff, fs)

df = df.merge(valid_data[['Time_sec', 'VO2_butter']], on='Time_sec', how='left')

# Plot smoothed VO2 against Time_sec after applying moving average and Butterworth filter
plt.figure(figsize=(10, 6))
plt.plot(df['Time_sec'], df['VO2'], label="Original VO2 (mL/min)", color='gray', alpha=0.5)
plt.plot(df['Time_sec'], df['VO2_30s_avg'], label="30s Moving Average VO2 (mL/min)", color='blue')
plt.plot(df['Time_sec'], df['VO2_butter'], label="Butterworth Filtered VO2 (mL/min)", color='red')
plt.xlabel('Time (seconds)')
plt.ylabel("VO2 (mL/min)")
plt.title("Original and Smoothed VO2 over Time")
plt.legend()
plt.grid(True)
plt.show()
