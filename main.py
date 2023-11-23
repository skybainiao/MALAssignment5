import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

reviews_path = "C:/Users/45527/Desktop/reviews.txt"
labels_path = "C:/Users/45527/Desktop/labels.txt"

reviews = pd.read_csv(reviews_path, header=None)
labels = pd.read_csv(labels_path, header=None)

Y = (labels == 'positive').astype(int)

X_train, X_test, y_train, y_test = train_test_split(reviews, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

vectorizer = CountVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train[0])
X_val_vec = vectorizer.transform(X_val[0])
X_test_vec = vectorizer.transform(X_test[0])

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='adam', verbose=10, random_state=1,
                    learning_rate_init=.01)

mlp.fit(X_train_vec, y_train[0])

y_val_pred = mlp.predict(X_val_vec)
accuracy_val = accuracy_score(y_val[0], y_val_pred)
print("Validation Set Accuracy:", accuracy_val)

y_test_pred = mlp.predict(X_test_vec)
accuracy_test = accuracy_score(y_test[0], y_test_pred)
print("Test Set Accuracy:", accuracy_test)

print(classification_report(y_test[0], y_test_pred))

sentences = ["This movie is great!", "I did not like this movie."]
sentences_vec = vectorizer.transform(sentences)

sentences_pred = mlp.predict(sentences_vec)
print("Sentences Predictions:", sentences_pred)
