# Approach 2: Time-Sequential Demand Prediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Load the datasets
df_raw = pd.read_csv("../analytics/raw_balanced_elevator_data.csv")
df_onehotencoded = pd.read_csv("../analytics/original_encoded_elevator_data.csv")
#get all headers by indexs
encoded_table_headers = df_onehotencoded.columns
hour_of_day_headers = list(encoded_table_headers[55:67])
day_of_week_headers = list(encoded_table_headers[66:71])
state_headers = list(encoded_table_headers[19:31])
call_headers = list(encoded_table_headers[5:19])
# needs to include previous headers too

def save_plot(file_name):
    output_folder = "../analytics"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, file_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


''' TODO: Fix to handle correct one hot encoding
def time_sequential_prediction(df):
    """Predict floor_call based on previous travel data."""
    # Create features for previous travel
    df['previous_floor_call'] = df['floor_call'].shift(1)
    df['previous_floor_state'] = df['floor_state'].shift(1)
    df['previous_hour_of_day'] = df['hour_of_day'].shift(1)
    
    # Drop the first row (no previous data)
    df = df.dropna(subset=['previous_floor_call', 'previous_floor_state', 'previous_hour_of_day'])
    
    
    
    # Features and target TODO
    X = df[['previous_floor_call', 'previous_floor_state', 'previous_hour_of_day']]
    y = df['floor_call']
    
    # One-hot encode categorical features TODO
    X_encoded = []
    
    # Split into training and testing data 
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train a simple model (e.g., Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Time-Sequential Prediction Accuracy: {accuracy:.2f}")
    
    return model, encoder, X_test, y_test


model, encoder, X_test, y_test = time_sequential_prediction(df_onehotencoded)

# Generate confusion matrix
y_pred_time_seq = model.predict(X_test)
conf_matrix_time_seq = confusion_matrix(y_test, y_pred_time_seq)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_time_seq, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix for Time-Sequential Prediction')
plt.xlabel('Predicted Floor Call')
plt.ylabel('Actual Floor Call')
save_plot("confusion_matrix_time_sequential.png")

print("- Time-Sequential Prediction: More complex, based on historical sequences.")

'''