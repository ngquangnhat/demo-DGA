
import joblib
import pandas as pd
import numpy as np
import random

# Machine Learning Packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Read CSV file
data = pd.read_csv(r"D:\BTL_ATBMTT\combined_domains.csv")



# Function to create tokens
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')  # Tokenize by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # Tokenize by dash
        tkns_ByDot = []
        for j in range(len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')  # Tokenize by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))  # Remove duplicate tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')  # Removing 'com' from tokens
    return total_Tokens

# Labels (assuming 'label' column exists in your CSV)
y = data["label"]
# Features (using the correct column name 'domain_name')
url_list = data["domain_name"]

# Using Custom Tokenizer with TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
# Store vectors in X as features
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model Building
#using logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)
# Accuracy of Our Model

print("Accuracy: ", logit.score(X_test, y_test))

# Save the model and vectorizer
joblib.dump(logit, 'model.job')  # Save the model
joblib.dump(vectorizer, 'vectorizer.job')  # Save the vectoriz