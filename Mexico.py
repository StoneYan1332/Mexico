# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %pip install plotly
# %pip install wbdata
# %pip install cufflinks
# %pip install eep153_tools
# !pip install wbdata
import wbdata
import numpy as np
import pandas as pd
import cufflinks as cf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
cf.go_offline()


# %%
def population(year, sex, age_range, place): #In [year] how many [people/males/females] aged [low] to [high] were living in [the world/region/country]?
    age_low = age_range[0]
    age_high = age_range[1]
    age_ranges = []
    if age_high > 80:
        for i in range(age_low, 80, 5):
            age_ranges.append(f"{i:02d}"+f"{i+4:02d}")
        age_ranges.append("80UP")
    else:
        if age_high % 5 == 0:
            for i in range(age_low, age_high, 5):
                age_ranges.append(f"{i:02d}"+f"{i+4:02d}")
        else:
            for i in range(age_low, age_high, 5):
                age_ranges.append(f"{i:02d}"+f"{i+4:02d}")
    
    if sex == "male":
        variables = {"SP.POP." + age_range + ".MA" : "Total Population in Mexico for " + sex + " from " + str(age_low) + " to " + str(age_high) for age_range in age_ranges}
    elif sex == "female":
        variables = {"SP.POP." + age_range + ".FE" : "Total Population in Mexico for " + sex + " from " + str(age_low) + " to " + str(age_high) for age_range in age_ranges}
        
    df = wbdata.get_dataframe(variables, country = place)
    df = df.filter(like = str(year), axis = 0)
    return df

population(2005, "male", (45, 49), "MEX")

# %%

# %%
SOURCE = 40
indicators = wbdata.get_indicators(source=SOURCE)

def population_df(region, year = 0): #A function that returns a pandas DataFrame indexed by Region or Country and Year, with columns giving counts of people in different age-sex groups.
    age_ranges = []
    for i in range(0,80,5):
        age_ranges.append(f"{i:02d}"+f"{i+4:02d}")
    age_ranges.append("80UP")
    
    male = {"SP.POP." + age_range + ".MA" : "Males " + age_range for age_range in age_ranges}
    female = {"SP.POP." + age_range + ".FE" : "Females " + age_range for age_range in age_ranges}
    vars = male
    vars.update(female)
    df = wbdata.get_dataframe(vars, country = region)
    if year == 0:
        df = df.filter(like = str(year), axis = 0)
    return df
population_df("MEX")


# %%
# general data frame for all indicies
import re
def dataframes(indicators):
    """
    Parameters:
    - indicators (list): A list of dictionaries where each dictionary contains 'id' and 'name' for World Bank indicators.
    
    Returns:
    - pandas.DataFrame: DataFrame containing the requested data for Mexico, indexed by Year.
    """
    # Initialize a dictionary to store the IDs and names of the indicators
    labels = {indicator['id']: indicator['name'] for indicator in indicators}

    def find_labels():
        """
        Filters through given indicators to select relevant population data IDs for Mexico.
        
        Returns:
        - dict: A dictionary with filtered indicator IDs as keys and their names as values.
        """
        # Regex to match relevant population indicators
        r = re.compile("SP\.POP\.[\d]{2}[A-Z0-9]{2}\.[MAFE]{2}$")
        
        # Filter indicators based on regex
        col_keys = [key for key in labels if r.match(key)]
        
        # Ensure total population is included
        col_keys.append('SP.POP.TOTL')
        
        # Filter and prepare the labels dictionary
        labels_filtered = {key: labels[key] for key in col_keys}
        
        return labels_filtered

    # Get filtered labels based on the defined criteria
    df_labels = find_labels()
    
    # Fetch data for the filtered labels, specifically for Mexico
    mexico_df = wbdata.get_dataframe(df_labels, country=["MEX"])

    return mexico_df


# %%
pop_df = dataframes(indicators)
pop_df.head()

# %%
# Check for Missing Values
# Check for any missing values in the DataFrame
missing_values = pop_df.isnull().sum()
missing_values

# Remove spaces, and standardize naming if necessary
pop_df.columns = pop_df.columns.str.replace(' ', '_').str.lower()

pop_df_reversed = pop_df.iloc[::-1]

# %%
# Data frame containing growth rates per year for every population
# Calculate the year-over-year growth rate for each column as before
growth_df = pop_df_reversed.pct_change()

# Convert the growth rate to percentage format
growth_df = growth_df * 100

# Rename columns to indicate these are growth rates
growth_df.columns = ['Growth Rate (%) - ' + col.replace('_', ' ').title() for col in growth_df.columns]

# Resetting the index to make 'date' a column if it's not already, for better table formatting
growth_df.reset_index(inplace=True)

# Optionally, format the DataFrame for presentation (e.g., rounding)
growth_df = growth_df.round(2)  # Round to two decimal places for clarity

# Display the formatted DataFrame
growth_df.tail()  # Display the first few rows to check the format

# %%
# Graph Pop Growth Rates for Every Age Group
# Convert 'date' from string to datetime to ensure proper plotting
growth_df['date'] = pd.to_datetime(growth_df['date'])

# Select a subset of columns to plot for clarity, for example, ages 00-04 and 20-24 for both genders
columns_to_plot = [
    'Growth Rate (%) - Population Ages 00-04, Female',
    'Growth Rate (%) - Population Ages 00-04, Male',
    'Growth Rate (%) - Population Ages 20-24, Female',
    'Growth Rate (%) - Population Ages 20-24, Male',
    'Growth Rate (%) - Population Ages 60-64, Female',
    'Growth Rate (%) - Population Ages 60-64, Male'
]

# Plotting
plt.figure(figsize=(12, 6))
for column in columns_to_plot:
    plt.plot(growth_df['date'], growth_df[column], label=column)

# Formatting the plot
plt.title('Population Growth Rates by Age Group and Gender in Mexico')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated date labels

# Display the plot
plt.show()

# %%
# Graph Total Pop Growth Rate
plt.figure(figsize=(12, 6))

# Plotting total population growth rate
plt.plot(growth_df['date'], growth_df['Growth Rate (%) - Population, Total'], label='Total Population', color='green', marker='^')

# Formatting the plot
plt.title('Total Population Growth Rate Over Time of Mexico')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# %%
# Pop Pyramid of Mexico for Given Year
# Assuming pop_df is your DataFrame and its index represents the year

# Select the most recent year's data from the DataFrame
most_recent_year = pop_df.index.max()
year_data = pop_df.loc[most_recent_year]

# Extract male and female population counts for each age group
# Age groups based on provided structure; adjust as necessary
age_groups = ['00-04', '05-09', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80_and_above']
male_population = [-year_data[f'population_ages_{age},_male'] for age in age_groups]
female_population = [year_data[f'population_ages_{age},_female'] for age in age_groups]

# Create the population pyramid plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plotting male population on the left by using negative values
ax.barh(age_groups, male_population, color='skyblue', label='Male')
# Plotting female population on the right
ax.barh(age_groups, female_population, color='pink', label='Female')

# Adding labels and title
ax.set_xlabel('Population Count')
ax.set_title(f'Population Pyramid of Mexico for {most_recent_year}')
ax.legend()

# Adding grid for better readability
ax.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# %%
# Economic Dataframes

def econ_dataframes(indicators):
    """
    Parameters:
    - indicators (list): A list of dictionaries where each dictionary contains 'id' and 'name' for World Bank indicators.
    
    Returns:
    - pandas.DataFrame: DataFrame containing the requested data for Mexico, indexed by Year.
    """
    # Initialize a dictionary to store the IDs and names of the indicators
    df_labels = {"NYGDPMKTPSACD" : "GDP, in USD"}
    print(df_labels)
    
    # Fetch data for the filtered labels, specifically for Mexico
    mexico_df = wbdata.get_dataframe(df_labels, country=["MEX"])

    return mexico_df

# Economic indicators
econ_indicators = wbdata.get_indicators(source=15)
gdp_df = econ_dataframes(indicators)
gdp_df.head()

# %%
# Check for Missing Values
# Check for any missing values in the DataFrame
missing_values = gdp_df.isnull().sum()
missing_values

# Remove spaces, and standardize naming if necessary
gdp_df.columns = gdp_df.columns.str.replace(' ', '_').str.lower()

gdp_df_reversed = gdp_df.iloc[::-1]

# %%
# Graph Changes to GDP
gdp_df['date'] = pd.to_datetime(gdp_df['date'])

plt.figure(figsize=(12, 6))

# Plotting GDP Growth
plt.plot(gdp_df["date"], gdp_df['GDP, in USD'], label='GDP', color='blue', marker='^')

# Formatting the plot
plt.title('Total GDP Growth Over Time of Mexico')
plt.xlabel('Year')
plt.ylabel('GDP ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# gdp_df.iplot(kind='scatter', mode='markers', symbol='circle-dot',
#          x="date",y="GDP, in USD",
#          xTitle="Log GDP per capita",yTitle="Total Fertility Rate",
#          title="Fact II: Women in Poorer Countries Have Higher Fertility")
