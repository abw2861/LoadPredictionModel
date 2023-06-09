import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# function to get and process weather data
@st.cache_data
def get_weather_data():
    #import weather data Houston, CST/CDT, 2017-01-01 to 2022-12-30
    weather_data = pd.read_csv('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/HOU.csv',
                            usecols=['valid','tmpf','dwpf','relh','p01i','sknt','feel'])

    # rename columns    
    weather_data.rename(columns={'valid': 'time',
                                'tmpf' : 'temp',
                                'dwpf' : 'dew_point',
                                'relh' : 'humidity', 
                                'p01i' : 'precip',
                                'sknt' : 'wind_speed',
                                'feel' :'feels_like' }, 
                                inplace=True)

    # replace M (missing data) with Nan
    weather_data.replace(to_replace='M', value=np.nan, inplace=True)

    # replace T (trace precip) with 0.00
    weather_data['precip'].replace(to_replace='T', value=0.00, inplace=True)

    # remove all rows containing Nan
    weather_data.dropna(axis=0, how='any', inplace=True)

    # convert values to float datatypes
    weather_data = weather_data.astype({'temp':'float','dew_point':'float',
                                        'humidity':'float','precip':'float',
                                        'wind_speed':'float','feels_like':'float' })

    # convert time to datetime datatype
    weather_data['time'] = pd.to_datetime(weather_data['time'])
    
    # resample weather dataframe by hour to match load dataset
    weather_df = weather_data.resample('1H', on='time').mean()

    # remove all rows containing Nan after resample
    weather_df.dropna(axis=0, how='any', inplace=True)

    return weather_df

# function to get and process load data
@st.cache_data
def get_load_data():
    # import load archive data
    load_2017 = pd.read_excel('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/native_Load_2017.xlsx')
    load_2017.rename(columns={'Hour Ending' : 'HourEnding'}, inplace=True)
    load_2018 = pd.read_excel('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/Native_Load_2018.xlsx')
    load_2019 = pd.read_excel('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/Native_Load_2019.xlsx')
    load_2020 = pd.read_excel('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/Native_Load_2020.xlsx')
    load_2021 = pd.read_excel('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/Native_Load_2021.xlsx')
    load_2021.rename(columns={'Hour Ending' : 'HourEnding'}, inplace=True)
    load_2022 = pd.read_excel('https://github.com/abw2861/LoadPredictionModel/blob/main/datasets/Native_Load_2022.xlsx')
    load_2022.rename(columns={'Hour Ending': 'HourEnding'}, inplace=True)

    dataframes = [load_2017,load_2018, load_2019, load_2020, load_2021, load_2022]

    # concat files, include only load usage for Coastal area
    load_test_data = pd.concat(dataframes, ignore_index=True)
    load_test_data = load_test_data[['HourEnding', 'COAST']]

    # change column names
    load_test_data.rename(columns={'HourEnding' : 'time', 'COAST' : 'load'}, inplace=True)

    # replace 2400 with 0:00 and convert to datetime type
    load_test_data['time'] = load_test_data['time'].replace('24:00', '00:00', regex=True)

    # change time column to datetime type
    load_test_data['time'] = pd.to_datetime(load_test_data['time'])

    # resample
    load_df = load_test_data.resample('1H', on='time').mean()

    # drop null values
    load_df.dropna(axis=0, how='any', inplace=True)
    
    load_df = add_time_ft(load_df)

    return load_df

# function to create time features
def add_time_ft(df):
    # add seasons
    df['is_summer'] = np.where((df.index.month >= 5) & (df.index.month <= 9),1,0)       # summer months 6-8
    df['is_winter'] = np.where((df.index.month == 12) | (df.index.month <= 2), 1, 0)    # winter months 12-2
    df['is_spring'] = np.where((df.index.month >= 3) & (df.index.month <= 5), 1, 0)     # spring months 3-5
    df['is_autumn'] = np.where((df.index.month >= 9) & (df.index.month <= 11), 1, 0)    # autumn months 9-11

    # add weekends
    df['is_weekend'] = np.where((df.index.weekday >= 5), 1, 0) 

    # add workhour
    df['is_workhour'] = np.where((df.is_weekend == 0) & (df.index.hour >= 8) & (df.index.hour <= 17),1,0)

    return df

# function to create line graphs
def create_load_line_graphs(load_df, year):
    # line plots load by year
    start_month = f'{year}-01'
    end_month = f'{year}-12'

    # resampled by month
    load_year = load_df[start_month : end_month].resample('M').mean()

    # create line plots
    fig = px.line(y=load_year['load'],labels={'x':'Month', 'y':'Energy Consumption (MW)'},title=f'Monthly Energy Consumption in {year}')

    # center title
    fig.update_layout(title={'y':0.9,'x':0.5,'xanchor':'center','yanchor':'top'}, width=600, height=500)

    # x axis tick labels
    fig.update_xaxes(tickvals=(np.arange(12)),ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    st.plotly_chart(fig)

# function to merge datasets
def merge_datasets(df1, df2, column):
    full_df = pd.merge(left=df1, right=df2, on=[column])
    return full_df

# function to create correlation matrix
def create_corr_matrix(df, columns):
    # correlation matrix for target & weather features
    corr_matrix = px.imshow(df[columns].corr(), 
                            color_continuous_scale='mint', title=' ')

    corr_matrix.update_layout(title={'y':0.9,'x':0.5,'xanchor':'center','yanchor':'top'}, width=600, height=500) # center title

    st.plotly_chart(corr_matrix)

# function to create boxplot
def create_boxplot(df):
    # box plot for summer months
    is_summer = df[df['is_summer'] == 1]
    is_not_summer = df[df['is_summer'] == 0]

    box_plot = go.Figure()
    box_plot.add_trace(go.Box(y=is_summer['load'], name="Summer Months"))
    box_plot.add_trace(go.Box(y=is_not_summer['load'], name="All Other Months"))

    box_plot.update_layout(yaxis_title='Load (MW)', title='Box Plot Distribution of Summer Months & Non-Summer Months',width=800, height=400)

    st.plotly_chart(box_plot)

# page config
st.set_page_config(page_title='Analysis', page_icon='📈')

# streamlit containers
title_container = st.container()
load_graph_container = st.container()
corr_matrix_container = st.container()
boxplot_container = st.container()

# title container: gets data
with title_container:
    st.markdown('# Analysis')
    st.write(
        '''Explore the visualizations below to view data trends and analysis!'''
    )
    
    load_df = get_load_data()
    weather_df = get_weather_data()
    full_df = merge_datasets(weather_df, load_df, 'time')
    
    full_df.to_csv('datasets/full_data.csv')

# container: generates line plots
with load_graph_container:
    st.header('Average Monthly Energy Consumption by Year')
    plot_2017, plot_2018, plot_2019, plot_2020, plot_2021 = st.tabs(['2017', '2018', '2019', '2020', '2021'])
    
    with plot_2017:
        create_load_line_graphs(load_df, 2017)

    with plot_2018:
        create_load_line_graphs(load_df, 2018)

    with plot_2019:
        create_load_line_graphs(load_df, 2019)

    with plot_2020:
        create_load_line_graphs(load_df, 2020)

    with plot_2021:
        create_load_line_graphs(load_df, 2021)


# container: generates correlation matrix
with corr_matrix_container:
    st.markdown("""----""")
    st.header('Correlation Matrix')
    create_corr_matrix(full_df, columns=['temp', 'dew_point', 'humidity', 'wind_speed', 'precip', 'feels_like', 'load'])

# container: generates boxplot
with boxplot_container:
    st.markdown("""----""")
    st.header('Box Plot Distribution')
    create_boxplot(full_df)

    

    
