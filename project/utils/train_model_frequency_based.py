from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
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
call_headers = list(encoded_table_headers[7:19])


def save_plot(file_name):
    output_folder = "../analytics"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, file_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
#Frequency-Based Heuristic Prediction Predict floor_call based on the most frequent floor_call for each (hour_of_day, floor_state) pair.
def frequency_based_prediction(df):
    # Group by hour_of_day and floor_state, then calculate the mode of floor_call
    freq_prediction = df.groupby(['hour_of_day', 'floor_state'])['floor_call'].agg(lambda x: x.mode().iloc[0]).reset_index()
    freq_prediction = freq_prediction.rename(columns={'floor_call': 'predicted_floor_call'})
    
    # Merge predictions back into the original dataframe
    df = df.merge(freq_prediction, on=['hour_of_day', 'floor_state'], how='left')
    
    accuracy = accuracy_score(df['floor_call'], df['predicted_floor_call'])
    print(f"Frequency-Based Heuristic Prediction Accuracy: {accuracy:.2f}")
    return df

df_freq_raw = frequency_based_prediction(df_raw)
# Generate confusion matrix for raw data
y_true_freq_raw = df_freq_raw['floor_call']
y_pred_freq_raw = df_freq_raw['predicted_floor_call']
conf_matrix_freq_raw = confusion_matrix(y_true_freq_raw, y_pred_freq_raw)
# Plot confusion matrix for raw data
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_freq_raw, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_true_freq_raw), yticklabels=np.unique(y_true_freq_raw))
plt.title('Confusion Matrix for Frequency-Based Prediction (Raw Data)')
plt.xlabel('Predicted Floor Call')
plt.ylabel('Actual Floor Call')
save_plot('confusion_matrix_frequency_based_raw.png')






# Train Linear Regression Model
feature_columns = hour_of_day_headers + day_of_week_headers + state_headers # Features
X_encoded = df_onehotencoded[feature_columns]
y = df_onehotencoded[call_headers]  # Target variable multi-output

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()


model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')  # Average MSE across all targets
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')  # Average R² across all targets
print(f'LinearRegression Mean Squared Error: {mse:.2f}')
print(f'LinearRegression R-squared: {r2:.2f}')

# Plot the results for each target column
for i, call_header in enumerate(call_headers):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5, label='Predictions')
    plt.plot([min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], 
             [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], 
             color='red', linestyle='--', label='Ideal Line')
    plt.xlabel(f'Actual {call_header}')
    plt.ylabel(f'Predicted {call_header}')
    plt.title(f'Actual vs Predicted {call_header}')
    plt.legend()
    save_plot(f'linear_regression_actual_vs_predicted_{call_header}.png')






'''TODO

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'RandomForestClassifier Accuracy: {accuracy * 100:.2f}%')




# Train Support Vector Machine (SVM) Classifier
model = svm.SVC(kernel='linear', random_state=42)  # You can change the kernel to 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy * 100:.2f}%')


'''













''' testing one_hot encodings
# Apply Approach frequency_based_prediction to one-hot encoded data
df_freq_onehot = frequency_based_prediction(df_onehotencoded)

# Generate confusion matrix for one-hot encoded data
y_true_freq_onehot = df_freq_onehot['floor_call']
y_pred_freq_onehot = df_freq_onehot['predicted_floor_call']
conf_matrix_freq_onehot = confusion_matrix(y_true_freq_onehot, y_pred_freq_onehot)

# Plot confusion matrix for one-hot encoded data
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_freq_onehot, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_true_freq_onehot), yticklabels=np.unique(y_true_freq_onehot))
plt.title('Confusion Matrix for Frequency-Based Prediction (One-Hot Encoded Data)')
plt.xlabel('Predicted Floor Call')
plt.ylabel('Actual Floor Call')
save_plot('confusion_matrix_frequency_based_onehot.png')
'''