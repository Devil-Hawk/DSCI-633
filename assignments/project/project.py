import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\job_train.csv")


# Drop irrelevant columns
data = data.drop(['title', 'location'], axis=1)

# Handle missing values if necessary
data = data.dropna()

# Split the data into features (X) and target variable (y)
X = data['description']
y = data['fraudulent']

# Step 2: Feature Engineering
# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed

# Fit and transform the text data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 4: Model Selection
# Initialize the Logistic Regression model
model = LogisticRegression()

# Step 5: Model Training
model.fit(X_train, y_train)

# Step 6: Model Evaluation
# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 7: Model Deployment
# Once satisfied with the model's performance, you can deploy it in a production environment.
