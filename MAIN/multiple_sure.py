#%%
import sys
sys.path.extend(['C:\\STUDIA\\Magisterka\\MAIN'])
from sklearn.metrics import accuracy_score

from MAIN.ensemble import load_models, predict_models, process_results, voting, calc_metrics
from MAIN.my_constans import DATA_PATH
from test_model import test_model
#%%
models_names = [
    "VGG16_best_92_from_0.keras",
    "MobileNetV2_best_92_from_0.keras",
    "EfficientNetB5_90_from_zero_only_one_layer.keras",
    "ResNet50V2_92_from_zero_only_one_layer.keras"
    ]
models_fcn = load_models(models_names)
#%%
summaries = predict_models(DATA_PATH / "test", models_fcn)
#%%
true, predicted, val = process_results(summaries, voting, ret_val=True)
calc_metrics(true, predicted)
summary = zip(true, predicted, val)
most_sure = list(filter(lambda x: x is not None, map(lambda x: (x[0], x[1]) if not 0.04 < x[2] < 0.96 else None, summary)))
true, pred = list(zip(*most_sure))
print(accuracy_score(true, pred), len(most_sure))
