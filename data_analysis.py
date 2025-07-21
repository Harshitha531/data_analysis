import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time


print("Generating large dataset...")
start = time.time()
X, y = make_classification(n_samples=1000000, n_features=20, n_informative=15,
                           n_classes=2, random_state=42)
end = time.time()
print(f"Dataset generated in {round(end - start, 2)} seconds.")

print("Saving dataset to 'large_data.csv'...")
feature_names = [f"feature_{i}" for i in range(1, 21)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df.to_csv("large_data.csv", index=False)
print("Saved as 'large_data.csv'.")

print("Loading dataset...")
df = pd.read_csv("large_data.csv")

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("Training logistic regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nEvaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
