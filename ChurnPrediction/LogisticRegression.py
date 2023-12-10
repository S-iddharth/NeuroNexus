#SIDDHARTH GUPTA, FOR NEURONEXUS INNOVATIONS

# Here we import all the required libraries for our logistic regression model..
import pandas as pd #for reading the dataset
from sklearn.model_selection import train_test_split #to train and test our data pieces..
from sklearn.preprocessing import StandardScaler #helps to preprocess and scale our data..
from sklearn.linear_model import LogisticRegression #defines the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #creates other helpful insights..
import seaborn as sns #helps in creating graphs and other visualizing tools..
import matplotlib.pyplot as plt #for plotting..

data = pd.read_csv('Churn_Modelling.csv')

# Since churned or not churned is our overall output, we take 'Exited' as the target variable..
X = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training
logic_reg = LogisticRegression()
logic_reg.fit(X_train, y_train)
y_pred = logic_reg.predict(X_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Model: {model.__class__.__name__}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{classification_rep}\n')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model(logic_reg, X_test, y_test)

#Confusion Matrix:
plot_confusion_matrix(y_test, y_pred)

#Letting the user enter new data to predict churn:
new_data = pd.DataFrame({
    'CreditScore': [670],      # Lower credit score may suggest financial instability
    'Age': [34],               # Higher age might indicate stability, so set a moderate age
    'Tenure': [3],             # Shorter tenure might suggest a less loyal customer
    'Balance': [80000],         # Low balance might indicate less engagement
    'NumOfProducts': [2],      # Fewer products used may suggest less commitment
    'HasCrCard': [1],           # Having a credit card may not have a significant impact
    'IsActiveMember': [1],      # Inactive members might be more likely to churn
    'EstimatedSalary': [988983.88], # Lower salary may contribute to financial strain
})

# Standardizing
new_data_scaled = scaler.transform(new_data)

# Predicting churn
churn_prediction = logic_reg.predict(new_data_scaled)
print(f'Churn Prediction for the new data: {churn_prediction}')
