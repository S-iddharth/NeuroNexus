#SIDDHARTH GUPTA, FOR NEURONEXUS INNOVATIONS

# Here we import all the required libraries for our logistic regression model..
import pandas as pd #for reading the dataset
from sklearn.model_selection import train_test_split #to train and test our data pieces..
from sklearn.ensemble import RandomForestClassifier #helps to preprocess our RandomForest model..
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc #helps in creating other insights..
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for visualization..

data = pd.read_csv('Churn_Modelling.csv')

# Since churned or not churned is our overall output, we take 'Exited' as the target variable..
X = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
ranf_model = RandomForestClassifier(n_estimators=100, random_state=42)
ranf_model.fit(X_train, y_train)

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Model: {model.__class__.__name__}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{classification_rep}\n')

# Evaluate the model
evaluate_model(ranf_model, X_test, y_test)

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': ranf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.show()

# ROC Curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Predicted Probabilities for Positive Class (Churn)
y_scores_rf = ranf_model.predict_proba(X_test)[:, 1]

# Plot ROC Curve for Random Forest
plot_roc_curve(y_test, y_scores_rf)

def predict_churn_new_data(model, scaler, new_data):
    # Standardize the new data using the same scaler
    new_data_scaled = scaler.transform(new_data)
    
    # Predict churn
    churn_prediction = model.predict(new_data_scaled)
    churn_probability = model.predict_proba(new_data_scaled)[:, 1]

    return churn_prediction, churn_probability

# new data
new_data = pd.DataFrame({
    'CreditScore': [784],
    'Age': [28],
    'Tenure': [2],
    'Balance': [109960.06],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [170829.87],
})

# Standardize the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Predict churn
churn_prediction, churn_probability = predict_churn_new_data(ranf_model, scaler, new_data)

# Print the prediction and probability
print(f'Churn Prediction for the new data: {churn_prediction[0]}')
print(f'Churn Probability: {churn_probability[0]:.2%}')
