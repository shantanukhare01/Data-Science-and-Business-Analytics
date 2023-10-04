import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Prepare your dataset with text samples and corresponding labels
# Replace 'messages' and 'labels' with your dataset.
messages = ["text sample 1", "text sample 2", "text sample 3"]
labels = ["class A", "class B", "class A"]

# Step 2: Text Preprocessing (customize as needed)
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(messages)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Example usage for classifying a new text sample
new_text_sample = ["new text sample"]
new_text_sample_vectorized = vectorizer.transform(new_text_sample)
prediction = classifier.predict(new_text_sample_vectorized)
print(f"Predicted class for the new text sample: {prediction[0]}")
