import pandas as pd #to read the dataset..
from sklearn.model_selection import train_test_split #to estabilish train and test set..
from sklearn.ensemble import GradientBoostingClassifier #to create GradientBoosting model..
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc #to get more insights..
import seaborn as sns #Data Visualization
import matplotlib.pyplot as plt #for plotting..

data = pd.read_csv('Churn_Modelling.csv')

# Since churned or not churned is our overall output, we take 'Exited' as the target variable..
X = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

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

    # Plot ROC curve
    y_scores = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_scores)

# plot a ROC curve
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

# Evaluate the model
evaluate_model(gb_model, X_test, y_test)

# Predict Churn for new data
new_data = pd.DataFrame({
    'CreditScore': [619],      # Lower credit score may suggest financial instability
    'Age': [42],               # Higher age might indicate stability, so set a moderate age
    'Tenure': [2],             # Shorter tenure might suggest a less loyal customer
    'Balance': [0],            # Low balance might indicate less engagement
    'NumOfProducts': [1],      # Fewer products used may suggest less commitment
    'HasCrCard': [1],           # Having a credit card may not have a significant impact
    'IsActiveMember': [1],      # Inactive members might be more likely to churn
    'EstimatedSalary': [101348.88],  # Lower salary may contribute to financial strain
})

# Predict churn for the new data
new_data_prediction = gb_model.predict(new_data)
print(f'Churn Prediction for the new data: {new_data_prediction}')
