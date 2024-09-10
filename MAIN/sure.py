from sklearn.metrics import accuracy_score

from MAIN.my_constans import DATA_PATH
from test_model import test_model

model_name = "VGG16_best_92_from_0.keras"
test_path = DATA_PATH / 'test'
result = test_model(model_name, test_path)
summary = result["summary"]
most_sure = list(filter(lambda x: x is not None, map(lambda x: (x[0], x[1]) if not 0.04 < x[2] < 0.96 else None, summary)))
true, pred = list(zip(*most_sure))
print(result["accuracy"], len(summary))
print(accuracy_score(true, pred), len(most_sure))
