from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
messages = [
    "Win a free iPhone now",
    "Congratulations you won lottery",
    "Call me when you are free",
    "Hey how are you",
    "Limited offer buy now",
    "Let's meet tomorrow"
]

labels = [
    "spam",
    "spam",
    "ham",
    "ham",
    "spam",
    "ham"
]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train model
model = MultinomialNB()
model.fit(X, labels)

print("📧 Spam Classifier Ready")

while True:
    msg = input("\nEnter message (or type exit): ")

    if msg.lower() == "exit":
        print("Goodbye 👋")
        break

    data = vectorizer.transform([msg])
    result = model.predict(data)

    print("Result:", result[0].upper())
