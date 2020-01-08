# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
plt.style.use('ggplot')


def load_and_clean_emissions(datafile):
    '''
    Function to load and clean the emissions data. Keep only total CO2 emissions.
    
    Input: Emissions CSV Datafile
    
    Output: Cleaned pandas dataframe
    '''
    
    # Import Emissions Data
    emissions_df = pd.read_csv(datafile)
    # Drop unnecessary columns and rename remaining columns
    emissions_df.drop(labels = ['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring',
                                'Per Capita', 'Bunker fuels (Not in Total)'], axis = 1, inplace = True)
    emissions_df.columns = ['Year', 'Country', 'Fossil_Fuel_CO2_Emissions']
    
    return emissions_df


def select_countries(country):
    '''
    Helper function to select the countries wanted from the dataframe for further analysis
    
    Input: Will take a country from the dataframe when run through filter function
    
    Output: True or False if country should be included
    '''
    # country_list is user choice of countries
    selected_countries = country_picks
    if (country in selected_countries): 
        return True
    else: 
        return False

    
def create_country_df_dict(df):
    '''
    Function to filter the selected countries from the dataframe and return a dictionary with country dataframe names for keys
    and country dataframes for values
    
    Input: Cleaned emissions dataframe
    
    Output: Dictionary with country dataframe names for keys and country dataframes for values
    '''
    
    # Select countries and create dictionary of country/dataframe
    chosen_countries = filter(select_countries, list(df.Country.unique()))

    # Create country dataframe names
    country_df_names = []
    for country in chosen_countries:
        df_name = f"{country}_df"
        country_df_names.append(df_name)

    # Create country dataframes and dictionary holding the dataframes
    country_df_dict = {}
    for df_name in country_df_names:
        country_df_dict[df_name] = df[df.Country == df_name[:-3]]
        
    return country_df_dict


def format_time_series(df_dict):
    '''
    Function to format each country dataframe into a time series format
    
    Input: Dictionary with country dataframe names for keys and country dataframes for values.
    
    Output: Dictionary with country dataframe names for keys and country time series for values.
    '''
    
    # Convert each country dataframe into proper time series layout
    for key in df_dict.keys():
        df_dict[key].Year = pd.to_datetime(df_dict[key].Year, format = '%Y')
        df_dict[key].drop('Country', axis = 1, inplace = True)
        df_dict[key].set_index('Year', inplace = True)
        
    return df_dict


def plot_country_time_series(df_dict):
    '''
    Function to plot individual country time series plots
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Country Time Series Plots.
    '''
    
    # Visualize Time Series by Country
    for key in df_dict.keys(): 
        df_dict[key].plot(figsize=(10,5))
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions (million metric tons of C)')
        plt.title(f'{key[:-3]} Emissions over time')
        plt.show()
        

        
def drop_years(df_dict):
    '''
    Function to drop years from time series given user choice based on selected countries.
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Dictionary with country dataframe names for keys and country time series for values (selected years only).
    '''
    
    # Drop years prior to 1902 (China has gaps prior to this year)
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].loc[(df_dict[key].index >= pd.to_datetime(starting_year, format = '%Y'))]
    
    return df_dict


def plot_all_countries_time_series(df_dict):
    '''
    Function to visualize all of the selected countries time series together.
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Time Series Plot.
    '''
    
    # Visualize All Time Series Together
    fig, ax = plt.subplots(figsize = (15,7))

    for key in df_dict.keys(): 
        ax.plot(df_dict[key], label = key[:-3])
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (million metric tons of C)')
    plt.title('CO2 Emissions Over Time by Country')
    plt.legend(loc = 'best')
    plt.show()


def decomposition(df_dict):
    '''
    Function to plot the decompositions of each country time series
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Decomposition Plots by country
    '''
    
    # Decompose Country TSs
    for key in df_dict.keys():
        ts = df_dict[key]
        decomposition = seasonal_decompose(np.log(ts))
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
    
        # Plot with subplots
        plt.figure(figsize=(10,6))
        plt.subplot(411)
        plt.plot(np.log(ts), label='Original', color="blue")
        plt.legend(loc='best')
        plt.title(f'{key[:-3]}')
        plt.subplot(412)
        plt.plot(trend, label='Trend', color="blue")
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal,label='Seasonality', color="blue")
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals', color="blue")
        plt.legend(loc='best')
        plt.tight_layout()
        
        
def differencing(df_dict):
    '''
    Function to difference the time series for each country.
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Dictionary with country dataframe names for keys and differenced country time series for values.
    '''
    
    # Perform differencing on each time series
    diff_df_dict = {}
    for key in df_dict.keys():
        diff_df_dict[key] = df_dict[key].diff(periods = 1).dropna()
    
    return diff_df_dict


def Dickey_Fuller_Test(df_dict):
    '''
    Function to perform Dickey-Fuller Test to check for Stationarity of Time Series
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Dickey-Fuller Test Results
    '''
    
    # Dickey-Fuller Tests on 2nd Order Differenced TSs
    dftest_pvalue_dict = {}
    for key in df_dict.keys():
        dftest = adfuller(df_dict[key].Fossil_Fuel_CO2_Emissions)
        dfoutput = pd.Series(dftest[0:4], index = ['Test Stat', 'p-value', '# lags used', '# Observations used'])
        dftest_pvalue_dict[key] = dftest[1]

    print(f'{((np.array(list(dftest_pvalue_dict.values())) < .05).sum() / len(df_dict))*100}% pass Dickey-Fuller Test')
    return dftest_pvalue_dict


def plot_partial_autocorrelation(df_dict):
    '''
    Function to plot partial autocorrelation plots. Helps to determine AR terms.
    
    Input: Dictionary with country dataframe names for keys and stationary country time series for values.
    
    Output: Partial Autocorrelation plots.
    '''
    for key in df_dict.keys():
        plot_pacf(df_dict[key], lags = 10)
        plt.title(f'Partial Autocorrelation {key[:-3]}')
        plt.show()
        
        
def plot_autocorrelation(df_dict):
    '''
    Function to plot autocorrelation plots. Helps to determine AR terms.
    
    Input: Dictionary with country dataframe names for keys and stationary country time series for values.
    
    Output: Autocorrelation plots.
    '''
    for key in df_dict.keys():
        plot_acf(df_dict[key], lags = 10)
        plt.title(f'Autocorrelation {key[:-3]}')
        plt.show()
        
        
def calc_paris_goals(df_dict):
    '''
    Function to calculate the Paris Climate Agreement Goals for each country.
    
    Input: Dictionary with country dataframe names for keys and country time series for values.
    
    Output: Dictionary with country dataframe names for keys and country Paris Climate Agreement Goals for values.
    '''
    
    # Calculate 40% Cut of 1990 Emissions by Country
    paris_levels = {}
    for key in df_dict.keys():
        ts = df_dict[key]
        emissions_1990 = ts[ts.index == pd.to_datetime(1990, format = '%Y')].get_value(pd.to_datetime(1990, format = '%Y'), 'Fossil_Fuel_CO2_Emissions')
        paris_levels[key] = emissions_1990*.6
        
    return paris_levels