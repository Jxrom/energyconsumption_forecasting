import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.dates import MonthLocator, DateFormatter
import plotly.figure_factory as ff
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

filepath = r'final_data.csv'

energycon_df = pd.read_csv(filepath)
energycon_df['Datetime'] = pd.to_datetime(energycon_df['Datetime'])  # Convert the date column to datetime
energycon_df.set_index('Datetime', inplace=True)
energycon_df.sort_values(by=['Datetime'], inplace=True, ascending=True)

# Checking if there are differences in the zero values of PFT and PFT
zero_mwt = energycon_df.loc[energycon_df.loc[:,'MWT'] == 0].index
zero_pft = energycon_df.loc[energycon_df.loc[:,'PFT'] == 0].index

def create_dataset(dataset, look_back=1):
    dataX = []  # Initializing as a list
    dataY = []  # Initializing as a list

    for i in range(len(dataset) - look_back):
        dataX.append(dataset.iloc[i:(i + look_back)].values)
        dataY.append(dataset.iloc[i + look_back])

    # Convert the lists to DataFrames
    dataX = pd.DataFrame(np.array(dataX).reshape(-1, look_back))
    dataY = pd.Series(dataY)

    return dataX, dataY



# Imputing zero values
for index in zero_mwt:
    energycon_df.loc[index,'MWT'] = np.random.normal(
        energycon_df.loc[:, 'MWT'].mean(),
        energycon_df.loc[:, 'MWT'].std()
        )
energycon_df.loc[index,'PFT'] = energycon_df.loc[:, 'PFT'].mean()

MWT_data = energycon_df
MWT_data.index.freq = 'H'

st.title("⚡Energy Consumption Forecasting")

# Load your data and perform necessary preprocessing here

st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to", ("Exploratory Data Analysis", "Models", "Model Summary"))

if page == "Exploratory Data Analysis":
    st.write("#### Raw Data")
    st.dataframe(energycon_df)
    
    st.write("### Exploratory Data Analysis")
    
    # Plotting 'MWT' Time Series
    fig, ax = plt.subplots()
    ax.plot(MWT_data.index, MWT_data['MWT'], color='dodgerblue')  # Change color here
    ax.set_xlabel('Month')
    ax.set_ylabel('MWT')
    ax.set_title('MWT Time Series Plot')
    ax.grid(True)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    
    # Displaying the Matplotlib figure directly in Streamlit
    st.pyplot(fig)
    
    # Plotting 'PFT' Time Series in a separate figure
    fig_pft, ax_pft = plt.subplots()
    ax_pft.plot(MWT_data.index, MWT_data['PFT'], color='dodgerblue')  # Change color here
    ax_pft.set_xlabel('Month')
    ax_pft.set_ylabel('PFT')
    ax_pft.set_title('PFT Time Series Plot')
    ax_pft.grid(True)
    ax_pft.xaxis.set_major_locator(MonthLocator())
    ax_pft.xaxis.set_major_formatter(DateFormatter("%b"))
    
    # Displaying the Matplotlib figure for 'PFT' directly in Streamlit
    st.pyplot(fig_pft)
    
    # For Plotly figure of 'MWT'
    plotly_fig_mwt = go.Figure()
    plotly_fig_mwt.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MWT'], mode='lines', name='MWT', line=dict(color='dodgerblue')))
    plotly_fig_mwt.update_layout(xaxis_title='Month', yaxis_title='MWT', title='MWT Time Series Plot')
    st.plotly_chart(plotly_fig_mwt, use_container_width=True)
    
    # For Plotly figure of 'PFT'
    plotly_fig_pft = go.Figure()
    plotly_fig_pft.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['PFT'], mode='lines', name='PFT', line=dict(color='dodgerblue')))
    plotly_fig_pft.update_layout(xaxis_title='Month', yaxis_title='PFT', title='PFT Time Series Plot')
    st.plotly_chart(plotly_fig_pft, use_container_width=True)

    # Interactive 'MWT' distribution using Plotly
    plotly_fig_dist = go.Figure(data=[go.Histogram(x=MWT_data['MWT'], nbinsx=20, marker=dict(color='dodgerblue'))])
    plotly_fig_dist.update_layout(
        title='MWT Distribution',
        xaxis_title='MWT Values',
        yaxis_title='Frequency'
    )

    st.plotly_chart(plotly_fig_dist, use_container_width=True)

    # Decomposition of 'MWT' data
    decomposition = seasonal_decompose(MWT_data['MWT'], model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Create separate Plotly figures for decomposed components

    # Trend plot
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=MWT_data.index, y=trend, mode='lines', name='Trend', line=dict(color='dodgerblue')))
    fig_trend.update_layout(title='Trend of MWT', xaxis_title='Month', yaxis_title='Trend')

    # Seasonal plot
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=MWT_data.index, y=seasonal, mode='lines', name='Seasonal', line=dict(color='dodgerblue')))
    fig_seasonal.update_layout(title='Seasonal Component of MWT', xaxis_title='Month', yaxis_title='Seasonal')

    # Residual plot
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=MWT_data.index, y=residual, mode='lines', name='Residual', line=dict(color='dodgerblue')))
    fig_residual.update_layout(title='Residual Component of MWT', xaxis_title='Month', yaxis_title='Residual')

    # Display the separate decomposition plots using st.plotly_chart()
    st.plotly_chart(fig_trend, use_container_width=True)
    st.plotly_chart(fig_seasonal, use_container_width=True)
    st.plotly_chart(fig_residual, use_container_width=True)

    # ACF and PACF using statsmodels
    lags = 100  # Number of lags for ACF and PACF

    # Calculate ACF and PACF
    acf_values = acf(MWT_data['MWT'], nlags=lags)
    pacf_values = pacf(MWT_data['MWT'], nlags=lags)

    # ACF Plot
    fig_acf = go.Figure(data=[go.Bar(x=list(range(lags + 1)), y=acf_values, marker=dict(color='dodgerblue'))])
    fig_acf.update_layout(title='Autocorrelation Function (ACF)', xaxis_title='Lag', yaxis_title='ACF')

    # PACF Plot
    fig_pacf = go.Figure(data=[go.Bar(x=list(range(lags + 1)), y=pacf_values, marker=dict(color='dodgerblue'))])
    fig_pacf.update_layout(title='Partial Autocorrelation Function (PACF)', xaxis_title='Lag', yaxis_title='PACF')

    # Display ACF and PACF using st.plotly_chart()
    st.plotly_chart(fig_acf, use_container_width=True)
    st.plotly_chart(fig_pacf, use_container_width=True)

    # Rolling Statistics using Plotly
    rolling_mean = MWT_data['MWT'].rolling(10).mean()
    rolling_std = MWT_data['MWT'].rolling(10).std()

    fig_rolling = go.Figure()

    fig_rolling.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MWT'], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
    fig_rolling.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='limegreen')))
    fig_rolling.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name='Rolling STD', line=dict(color='indianred')))

    fig_rolling.update_layout(
        title='Rolling Statistics',
        xaxis_title='Date',
        yaxis_title='MWT',
        xaxis=dict(tickformat="%b")  # Show all months on x-axis
    )

    # Display Rolling Statistics using st.plotly_chart()
    st.plotly_chart(fig_rolling, use_container_width=True)

    original_df_perhour = MWT_data
    resampled_df_perday = MWT_data.resample('D').sum()
    resampled_df_weekly = MWT_data.resample('7D').sum()
    resampled_df_monthly = MWT_data.resample('M').sum()

    # Plotly figures for different resampled time intervals
    fig_original = go.Figure()
    fig_original.add_trace(go.Scatter(x=original_df_perhour.index, y=original_df_perhour['MWT'], mode='lines', name='Original Time Series Data', line=dict(color='dodgerblue')))
    fig_original.update_layout(title='MWT Hourly', xaxis_title='Date', yaxis_title='MWT')

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=resampled_df_perday.index, y=resampled_df_perday['MWT'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_daily.update_layout(title='MWT Daily', xaxis_title='Date', yaxis_title='MWT')

    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(x=resampled_df_weekly.index, y=resampled_df_weekly['MWT'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_weekly.update_layout(title='MWT Weekly', xaxis_title='Date', yaxis_title='MWT')

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(x=resampled_df_monthly.index, y=resampled_df_monthly['MWT'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_monthly.update_layout(title='MWT Monthly', xaxis_title='Date', yaxis_title='MWT')

    # Display Plotly figures using st.plotly_chart()
    st.plotly_chart(fig_original, use_container_width=True)
    st.plotly_chart(fig_daily, use_container_width=True)
    st.plotly_chart(fig_weekly, use_container_width=True)
    st.plotly_chart(fig_monthly, use_container_width=True)

elif page == "Models":
    from sklearn.preprocessing import MinMaxScaler

    # Feature scaling
    scaler = MinMaxScaler()
    MWT_data['MinMax'] = scaler.fit_transform(MWT_data['MWT'].values.reshape(-1, 1))

    # Plotly figure for 'MinMax' Time Series
    fig_minmax = go.Figure()
    fig_minmax.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MinMax'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_minmax.update_layout(
        title='Normalized MWT Time Series Plot',
        xaxis_title='Date',
        yaxis_title='MWT'
    )

    # Display Plotly figure using st.plotly_chart()
    st.plotly_chart(fig_minmax, use_container_width=True)

    # Split the data into training and testing DataFrames
    data = MWT_data['MinMax']

    train_size = int(len(data) * 0.8)  # 80% of the data for training

    train = data.iloc[:train_size]  # Select the first 80% for training
    test = data.iloc[train_size:]   # Select the remaining 20% for testing

    # Plotly figure for Train-Test Split
    fig_train_test = go.Figure()
    fig_train_test.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data', line=dict(color='dodgerblue')))
    fig_train_test.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Testing Data', line=dict(color='indianred')))
    fig_train_test.update_layout(
        title='80/20 Train-Test Split',
        xaxis_title='Time',
        yaxis_title='Value'
    )
    
    # Display Plotly figure using st.plotly_chart()
    st.plotly_chart(fig_train_test, use_container_width=True)
    look_back = 24  # Number of previous hours to use for prediction
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    model_section = st.selectbox("Choose a Model", ["MLP", "LSTM", "GRU"])

    if model_section == "MLP":
        # Load your Keras model
        model = keras.models.load_model('best_model_mlp.h5')
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = trainPredict.flatten()
        testPredict = testPredict.flatten()

        # Plotting in Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=train.index[look_back:], y=trainY, mode='lines', name='Training Data', line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=test.index[look_back:], y=testY, mode='lines', name='Testing Data', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index[look_back:len(trainPredict) + look_back], y=trainPredict, mode='lines', name='Training Predictions', line=dict(color='indianred')))
        fig.add_trace(go.Scatter(x=data.index[len(trainPredict) + 2 * look_back:], y=testPredict, mode='lines', name='Testing Predictions', line=dict(color='yellow')))

        fig.update_layout(
            title='Actual vs Predicted',
            xaxis_title='Months',
            yaxis_title='MWT',
            xaxis=dict(tickformat="%b"),  # Show all months on x-axis
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[:744], y=data[:744], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[:744], y=trainPredict[:744], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Month Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[:24], y=data[:24], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[:24], y=trainPredict[:24], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Day Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)
    elif model_section == "LSTM":
        # Load your Keras model
        model = keras.models.load_model('best_model_lstm.h5')
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = trainPredict.flatten()
        testPredict = testPredict.flatten()

        # Plotting in Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=train.index[look_back:], y=trainY, mode='lines', name='Training Data', line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=test.index[look_back:], y=testY, mode='lines', name='Testing Data', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index[look_back:len(trainPredict) + look_back], y=trainPredict, mode='lines', name='Training Predictions', line=dict(color='indianred')))
        fig.add_trace(go.Scatter(x=data.index[len(trainPredict) + 2 * look_back:], y=testPredict, mode='lines', name='Testing Predictions', line=dict(color='yellow')))

        fig.update_layout(
            title='Actual vs Predicted',
            xaxis_title='Months',
            yaxis_title='MWT',
            xaxis=dict(tickformat="%b"),  # Show all months on x-axis
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[:744], y=data[:744], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[:744], y=trainPredict[:744], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Month Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[:24], y=data[:24], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[:24], y=trainPredict[:24], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Day Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)
    elif model_section == "GRU":
        # Load your Keras model
        model = keras.models.load_model('best_model_gru.h5')
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = trainPredict.flatten()
        testPredict = testPredict.flatten()

        # Plotting in Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=train.index[look_back:], y=trainY, mode='lines', name='Training Data', line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=test.index[look_back:], y=testY, mode='lines', name='Testing Data', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index[look_back:len(trainPredict) + look_back], y=trainPredict, mode='lines', name='Training Predictions', line=dict(color='indianred')))
        fig.add_trace(go.Scatter(x=data.index[len(trainPredict) + 2 * look_back:], y=testPredict, mode='lines', name='Testing Predictions', line=dict(color='yellow')))

        fig.update_layout(
            title='Actual vs Predicted',
            xaxis_title='Months',
            yaxis_title='MWT',
            xaxis=dict(tickformat="%b"),  # Show all months on x-axis
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[:744], y=data[:744], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[:744], y=trainPredict[:744], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Month Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[:24], y=data[:24], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[:24], y=trainPredict[:24], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Day Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

elif page == "Model Summary":
    # Create a dictionary with the model names and their respective metrics
    data = {
        'MLP': {
            'MAE': 0.038255095193906415,
            'RMSE': 0.06185594955817233,
            'MAPE': 9.907460706123754
        },
        'LSTM': {
            'MAE': 0.03755048672155376,
            'RMSE': 0.056345582334668395,
            'MAPE': 38.02152078277709
        },
        'GRU': {
            'MAE': 0.03743318837196568,
            'RMSE': 0.058640491568784016,
            'MAPE': 37.64786391197181
        }
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Transpose the DataFrame to have models as rows and metrics as columns
    df = df.transpose()

    # Display the DataFrame using Streamlit
    st.write("### Model Performance Metrics")
    st.write(df)

