import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("IRIS.csv")
print("Dataset imported")

encoder = LabelEncoder()
data["species"] = encoder.fit_transform(data["species"])

features = data.iloc[:, :-1]
target = data.iloc[:, -1]

train_x, test_x, train_y, test_y = train_test_split(
    features, target, test_size=0.25, random_state=10
)

tree_model = DecisionTreeClassifier()
tree_model.fit(train_x, train_y)
tree_score = accuracy_score(test_y, tree_model.predict(test_x))
print("\nDecision Tree Score:", tree_score)

log_model = LogisticRegression(max_iter=500)
log_model.fit(train_x, train_y)
log_score = accuracy_score(test_y, log_model.predict(test_x))
print("Logistic Regression Score:", log_score)

selected = tree_model if tree_score >= log_score else log_model
print("\nSelected Model:", "Decision Tree" if selected == tree_model else "Logistic Regression")

sample_input = [[4.9, 3.1, 1.5, 0.1]]
result = selected.predict(sample_input)
label = encoder.inverse_transform(result)
print("\nPrediction:", label[0])