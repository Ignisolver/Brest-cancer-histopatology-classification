
from my_constans import MODEL_DST

from my_constans import DATA_PATH
from test_model import test_model
from train_model import train_model

"""TODO Treining:
1. Oryginalne
2. Przetworzone
3. Większe
4. Mylące
5. Trening przed mylącymi
6. Mobile Net ensemble

TODO Test
1. Oryginalne
2. Przetworzone
3. Większe normalnie
4. Większe pojedyńczo
5. Ensemble zwykłe / MobileNet
6. Pewności - zwykłe + ensemble
7. Mylące
"""
class H:
    def __init__(self):
        self.history = {"val_accuracy": [None]}
def train_final(model_base_name, suffix, data_path, model_dst, v_s, e1_e2, size, epoch, raport_name, test_data_path, noumber,
                test=True
                ):
    suffix = "__" + suffix + "__" + str(noumber) + ".keras"
    model_name = model_base_name + suffix
    model_path = model_dst / model_name
    print(model_name)
    try:
        history = train_model(model_base_name, suffix,
                    data_path, model_dst,
                    v_s=v_s, e1_e2=e1_e2,
                    size=size, epoch=epoch)
    except KeyboardInterrupt:
        history = H()
    if test:
        res = test_model(model_path, model_name, test_data_path)
        with open(raport_name, 'a') as file:
            file.write(f"________\n"
                       f"Model: {name},\n"
                       f" acc: {res['accuracy']};"
                       f" rec: {res['recall']}\n"
                       f"history: {history.history['val_accuracy']}\n")

# Originalne
print("### ORIGINAL")
model_names = [("MobileNetV2",(14, 25), 35),
                ("ResNet50V2", (20, 30), 40),
                  ("DenseNet169", (6, 15), 22),
                    ("InceptionV3", (14, 25), 35),
                   ("EfficientNetV2M", (10, 20), 30),
                    # ("VGG19", (14, 25), 35),
                    ("VGG16", (14, 25), 35),
               ]

for n, (name, e1_e2, epoch) in enumerate(model_names):
    train_final(model_base_name=name,
                suffix="FIN_original",
                data_path=DATA_PATH / "original" / "train",
                model_dst=MODEL_DST / "original",
                v_s=0.002,
                e1_e2=e1_e2,
                size=(50, 50),
                epoch=epoch,
                raport_name="final_original.txt",
                test_data_path=DATA_PATH / "original" / "test",
                noumber=n
                )
# # Przetworzone
# print("### PREPROCESSED")
# model_names = [("MobileNetV2", (14, 25), 3),
#                 ("ResNet50V2", (20, 30), 40),
#                   ("DenseNet169", (6, 15), 22),
#                     ("InceptionV3", (14, 25), 35),
#                    ("EfficientNetV2M", (10, 20), 30),
#                     # ("VGG19", (14, 25), 35),
#                     ("VGG16", (14, 25), 35),
#                ]
#
# for n, (name, e1_e2, epoch) in enumerate(model_names):
#     train_final(model_base_name=name,
#                 suffix="FIN_ready",
#                 data_path=DATA_PATH / "ready" / "train",
#                 model_dst=MODEL_DST / "ready",
#                 v_s=0.002,
#                 e1_e2=e1_e2,
#                 size=(50, 50),
#                 epoch=epoch,
#                 raport_name="final_preprocessed.txt",
#                 test_data_path=DATA_PATH / "ready" / "test",
#                 noumber=n
#                 )

# Fakes
print("### FAKES")
model_names = [("MobileNetV2", (7, 14), 20),
("MobileNetV2", (7, 14), 20),
("DenseNet169", (6, 10), 15),
                  ("DenseNet169", (6, 10), 15),
("DenseNet169", (6, 10), 15),
               ]
for n, (name, e1_e2, epoch) in enumerate(model_names):
    train_final(model_base_name=name,
                suffix="FIN_fakes",
                data_path=DATA_PATH / "fake_detection" / "train_base" / "train",
                model_dst=MODEL_DST / "fakes",
                v_s=0.002,
                e1_e2=e1_e2,
                size=(50, 50),
                epoch=epoch,
                raport_name="final_fakes.txt",
                test_data_path=DATA_PATH / "ready" / "test",
                noumber=n,
                test=True
                )
# # multiple
# print("### MULTIPLE")
# name, e1_e2, epoch = ("DenseNet169", (6, 10), 15)
# for n in range(13):
#     train_final(model_base_name=name,
#                 suffix="FIN_multiple",
#                 data_path=DATA_PATH / "multiple_train" / f"{n}" / "train",
#                 model_dst=MODEL_DST / "ensemble",
#                 v_s=0.025,
#                 e1_e2=e1_e2,
#                 size=(50, 50),
#                 epoch=epoch,
#                 raport_name="final_multiple.txt",
#                 test_data_path=DATA_PATH / "multiple_train" / "13" / "test",
#                 noumber=n,
#                 )

# Large
print("### LARGE")
model_names = [("MobileNetV2", (10, 20), 30),
("MobileNetV2", (8, 13), 20),]

for i, (name, e1_e2, epoch) in enumerate(model_names):
    train_final(model_base_name=name,
                suffix=f"FIN_large_{i+2}",
                data_path=DATA_PATH / "large" / f"l_{i+2}_ready" / "train",
                model_dst=MODEL_DST / "large",
                v_s=0.025,
                e1_e2=e1_e2,
                size=((i+2)*50, (i+2)*50),
                epoch=epoch,
                raport_name="final_large_23.txt",
                test_data_path=DATA_PATH / "large" / f"l_{i+2}_ready" / "test",
                noumber=i,
                )