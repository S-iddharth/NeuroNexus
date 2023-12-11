#importing the nessasary libraries fro our model..
import pandas as pd # To analyse and open the dataset
from sklearn.model_selection import train_test_split # For training and testing our data pieces..
from sklearn.feature_extraction.text import TfidfVectorizer # for TF-IDF model
from sklearn.naive_bayes import MultinomialNB #Multinomial for Discrete counts
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #more insights
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for visualization..

df = pd.read_csv('spam[1].csv', encoding='latin-1')
df = df[['v1', 'v2']] #v1 = specifier, v2 = message

X = df['v2']
y = df['v1'].map({'ham': 0, 'spam': 1}) #encoding the specifiers

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectors
tf_vectorizer = TfidfVectorizer()
X_train_tfidf = tf_vectorizer.fit_transform(X_train)
X_test_tfidf = tf_vectorizer.transform(X_test)

nb_classifier = MultinomialNB() # Naive Bayes Classifiers
nb_classifier.fit(X_train_tfidf, y_train)

y_pred = nb_classifier.predict(X_test_tfidf)# Evaluating

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')# Print accuracy and classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()# MAKING the Confusion Matrix

# Visualize Spam and Ham Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='v1', data=df)
plt.title('Distribution of Spam and Ham Messages')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

def test_new_data(new_message):
    new_message_tfidf = tf_vectorizer.transform([new_message])
    prediction = nb_classifier.predict(new_message_tfidf)
    return 'spam' if prediction[0] == 1 else 'ham'
#Using the model
print("Enter the message: ")
new_message = input()
result = test_new_data(new_message)
print(f'The message is classified as: {result}')
