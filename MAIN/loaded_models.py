from tensorflow.keras.applications import (
ResNet50V2,
EfficientNetB5,
VGG16,
DenseNet169,
InceptionV3,
MobileNetV2,
ResNet152V2,
EfficientNetV2M,
VGG19
)
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_efficientnet_v2
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet_v2

models = {
    "ResNet50V2": {"model": ResNet50V2, "func": preprocess_input_resnet_v2},
    "ResNet152V2": {"model": ResNet152V2, "func": preprocess_input_resnet_v2},
    "EfficientNetB5": {"model": EfficientNetB5, "func": preprocess_input_efficientnet},
    "EfficientNetV2M": {"model": EfficientNetV2M, "func": preprocess_input_efficientnet_v2},
    "VGG16": {"model": VGG16, "func": preprocess_input_vgg16},
    "VGG19": {"model": VGG19, "func": preprocess_input_vgg19},
    "DenseNet169": {"model": DenseNet169, "func": preprocess_input_densenet},
    "InceptionV3": {"model": InceptionV3, "func": preprocess_input_inception_v3},
    "MobileNetV2": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV20": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV21": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV22": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV23": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV24": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV25": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV26": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV27": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV28": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV29": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV210": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV211": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV212": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV213": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV214": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV215": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
    "MobileNetV216": {"model": MobileNetV2, "func": preprocess_input_mobilenet},
}