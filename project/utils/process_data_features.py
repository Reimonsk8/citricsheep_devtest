import pandas as pd
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

# Constants for API endpoints
DEMAND_TABLE_URL = 'http://localhost:5000/api/demands'
STATE_TABLE_URL = 'http://localhost:5000/api/states'

#################################################
# Loading and merging data to create dataframe
#################################################

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None
    
    
def get_direction_traveled(row):
    if row['floor_call'] < row['floor_state']: return 'Up'
    else: return 'Down' 
    
#get raw api data from table
demand_data = fetch_data(DEMAND_TABLE_URL)['demands']
state_data = fetch_data(STATE_TABLE_URL)['states']

#show table heads before merging
df_demand = pd.DataFrame(demand_data)
df_demand.head()

df_state = pd.DataFrame(state_data)
df_state.head()

df_merged = pd.merge(df_demand, df_state, on='id', how='left')
# Reorder columns
df_merged.insert(0, 'call_order', pd.NA)
df_merged = df_merged[['id'] + [col for col in df_merged.columns if col != 'id']]

# Rename columns, i was thinking on the most aprropiate names but couldn't decide on these:
#call_floor, request_floor, origin_floor, source_floor
#destination_floor, target_floor, arrival_floor, disembark_floor
df_merged = df_merged.rename(columns={
    'floor_x': 'floor_call', 
    'timestamp_x': 'timestamp_demand', 
    'floor_y': 'floor_state', 
    'timestamp_y': 'timestamp_state'
})

# timestamps to datetime objects
df_merged['timestamp_demand'] = pd.to_datetime(df_merged['timestamp_demand'], format='mixed')
df_merged['timestamp_state'] = pd.to_datetime(df_merged['timestamp_state'], format='mixed')

# Sort by timestamp_demand to ensure the order is correct  and assign call_order
df_merged = df_merged.sort_values(by='timestamp_demand')
df_merged['call_order'] = range(0, len(df_merged))

# calculate previous floor call, state and idle hours
df_merged['previous_floor_call'] = df_merged['floor_call'].shift(1)
df_merged['previous_floor_state'] = df_merged['floor_state'].shift(1)
df_merged['idle_hours'] = (df_merged['timestamp_demand'] - df_merged['timestamp_demand'].shift(1)).dt.total_seconds() / 3600

# Delete the first row
df_merged = df_merged.drop(df_merged.index[0])

# Add additional features
df_merged['direction_traveled'] = df_merged.apply(get_direction_traveled, axis=1)
df_merged['hour_of_day'] = df_merged['timestamp_demand'].dt.hour
df_merged['day_of_week'] = df_merged['timestamp_demand'].dt.day_name()
df_merged['seconds_traveled'] = (df_merged['timestamp_state'] - df_merged['timestamp_demand']).dt.total_seconds()
    




######################################################
# Data Analysis and Visualization for desicion making
#####################################################

def save_plot(file_name):
    output_folder = "../analytics"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, file_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Plot the DataFrame table for examples
def plot_data_head():
    df_merged_first = df_merged.head(25)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    table(ax, df_merged_first, loc='center')
    save_plot("dataframe_sample.png")


# Plot the frequencies
def plot_call_frequency():
    all_floors = list(range(1, 14))  # Assuming 13 floors
    # Fill missing frequencies with 0 for consistency
    floor_call_counts = df_merged["floor_call"].value_counts().reindex(all_floors, fill_value=0)
    floor_state_counts = df_merged["floor_state"].value_counts().reindex(all_floors, fill_value=0)
    freq_df = pd.DataFrame({
        "floor": all_floors,
        "floor_call": floor_call_counts.values,
        "floor_state": floor_state_counts.values
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x="floor", y="value", hue="variable", 
                data=pd.melt(freq_df, id_vars=["floor"], value_vars=["floor_call", "floor_state"]))
    plt.title('Floor Call and Floor State Frequency')
    plt.xlabel('Floor')
    plt.ylabel('Frequency')
    save_plot("floor_call_floor_state_frequency.png")


#scatterplot hourly demand and state
def plot_hourly_demand_state():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df_merged, x='floor_call', y='floor_state', hue='hour_of_day'
    )
    
    plt.title('Hourly Demand vs State')
    plt.xlabel('Floor Call')
    plt.ylabel('Floor State')

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    save_plot("hourly_demand_vs_state.png")


#plot how many days of week and what hours a day is the elevator idle
def plot_idle_hours():
    # Ensure 'day_of_week' is categorical and ordered correctly
    df_merged['day_of_week'] = pd.Categorical(
        df_merged['day_of_week'], 
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
        ordered=True
    )
    # Ensure 'floor_call' is categorical and includes all floors from 1 to 13
    df_merged['floor_call'] = pd.Categorical(
        df_merged['floor_call'], 
        categories=range(1, 14), 
        ordered=True
    )
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    bar_plot = sns.barplot(
        data=df_merged, 
        x='day_of_week', 
        y='idle_hours', 
        hue='floor_call', 
        palette='tab10',
        errorbar=None
    )
    plt.xlabel('Day of Week', fontsize=14, labelpad=10)
    plt.ylabel('Idle Hours', fontsize=14, labelpad=10)
    plt.title('Average Idle Hours vs Day of Week Colored by Floor', fontsize=16, pad=20)
    #hardcoded max y value
    MAX_Y = 6
    plt.ylim(0, MAX_Y)
    # Add floor number labels at the bottom of each bar
    for p in bar_plot.patches:
        x = p.get_x() + p.get_width() / 2
        height = int(p.get_height())
        if height >= MAX_Y:
            bar_plot.annotate(
                f'{height}',
                (x, 0),  
                ha='center',
                va='top',
                xytext=(0, -7),  # Offset for the text (5 points above the base)
                textcoords='offset points'  # Coordinate system for the offset
            )
    plt.tight_layout()
    save_plot("idle_hours_in_floor_per_week_barplot.png")

#Plot a heatmap of call frequency by day of week and hour of day.
def plt_weekly_hour_heat_map():
    # Dynamically get the min and max hour values from the data
    min_hour = df_merged["hour_of_day"].min() - 1
    max_hour = df_merged["hour_of_day"].max() + 1
    
    heatmap_data = df_merged.pivot_table(
        index="day_of_week", columns="hour_of_day", values="floor_call", aggfunc="count", fill_value=0, observed=False
    )
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(ordered_days)
    
    # Ensure integer values for the heatmap
    heatmap_data = heatmap_data.fillna(0).astype(int)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt='d', cbar=True, linewidths=0.5)
    
    # Convert hours to 12-hour format for the title
    min_hour_12 = min_hour if min_hour < 12 else min_hour - 12
    max_hour_12 = max_hour if max_hour < 12 else max_hour - 12
    min_period = "AM" if min_hour < 12 else "PM"
    max_period = "AM" if max_hour < 12 else "PM"
    
    plt.title(f'Call Frequency by Day of Week and Hour of Day ({min_hour_12} {min_period} to {max_hour_12} {max_period})')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    
    save_plot("weekly_hour_heat_map_calls.png")
    plt.show()

    
# Violin Plot (Horizontal) 
def seconds_traveled_distribution():   

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_merged, x='seconds_traveled', y='direction_traveled', hue="direction_traveled", inner="quart")
    plt.title('Distribution of Seconds Traveled by Direction')
    plt.xlabel('Seconds Traveled')
    plt.ylabel('Direction Traveled')
    
    save_plot("traveled_distribution_violin_plot_horizontal.png")
    plt.show()


plot_data_head()
plot_call_frequency()
plot_hourly_demand_state()
plt_weekly_hour_heat_map()
plot_idle_hours()
seconds_traveled_distribution()


#data engeiring for training we dont need data where elevator is occupied remove 'vacant' = false data from dta frame = df_merged
df_cleaned = df_merged
def plot_remove_occupied():
    df_cleaned = df_merged[df_merged['vacant'] != False]
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_merged, x='floor_state', color='blue', label='Before Removing Vacant=False', alpha=0.6)   
    sns.countplot(data=df_cleaned, x='floor_state', color='orange', label='After Removing Vacant=False', alpha=0.6)
    plt.title('Comparison of Floor State Before and After Removing Vacant=False')
    plt.xlabel('Floor State')
    plt.ylabel('Count')
    plt.legend()
    
    # Save and show the plot
    save_plot("remove_occupied_comparison.png")
    plt.show()
    
plot_remove_occupied()

balanced_data = df_cleaned
#balance data with more frequency to cap to the lowest frequency per call
def plot_balanced_distribution(): 
    # Balance the dataset by downsampling
    min_frequency = df_cleaned['floor_state'].value_counts().min()  # Get the minimum frequency
    balanced_data = df_cleaned.groupby('floor_state').apply(lambda x: x.sample(min_frequency)).reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_cleaned, x='floor_state', color='blue', label='Before Balancing', alpha=0.6)
    sns.countplot(data=balanced_data, x='floor_state', color='green', label='After Balancing', alpha=0.6)
    plt.title('Comparison of Floor State Before and After Balancing')
    plt.xlabel('Floor State')
    plt.ylabel('Count')
    plt.legend()
    
    # Save and show the plot
    save_plot("floor_state_balanced_comparison.png")
    plt.show()

# Call the function
plot_balanced_distribution()




'''
At this point, after analyzing the data, I see there are two approaches I can take to predict the next floor where the elevator will be called:

Approach 1 (t,x = y) Frequency-Based Demand Prediction: 
Based on how often the same input (time_stamp, floor_state) is used for frequent travel, I can predict the output (floor_demand) where the elevator was originally called from. This way, I can position the elevator closer to that floor.

Approach 2(pt, px, py = t,x,y) Time-Sequential Demand Prediction: 
If I order the table historically, I can observe that there is always a previous travel before an elevator demand and state. I can store this in the table and create a relationship where the input consists of the previous travel (time_stamp, previous_demand, previous_state), and the output is the next travel (time_stamp, floor_demand, floor_state).

I would like to implement both approaches and compare the results to make a decision based on the findings.

'''


