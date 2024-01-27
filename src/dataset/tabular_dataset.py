from ucimlrepo import fetch_ucirepo

# Fetch the SpamBase dataset
spambase = fetch_ucirepo(id=94)
X_spam = spambase.data.features
y_spam = spambase.data.targets

# Fetch the Statlog (German Credit Data) dataset
statlog_german_credit_data = fetch_ucirepo(id=144)
X_credit = statlog_german_credit_data.data.features
y_credit = statlog_german_credit_data.data.targets

# Print the head of each dataset
print("SpamBase Dataset - Features:")
print(X_spam.head())
print("\nSpamBase Dataset - Targets:")
print(y_spam.head())
print("\nStatlog German Credit Data - Features:")
print(X_credit.head())
print("\nStatlog German Credit Data - Targets:")
print(y_credit.head())
