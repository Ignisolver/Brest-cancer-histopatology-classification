from keras.models import load_model

# Wczytanie modelu
model = load_model(r"C:\STUDIA\Magisterka\MAIN\models\MobileNetV2_final_preprocessed_10.keras")

# Wyświetlenie architektury modelu
print("Architektura modelu:")
print(model.summary())

# Wyświetlenie parametrów modelu
print("\nParametry modelu:")
for layer in model.layers:
    print(layer.get_config())
    print()

# Wyświetlenie parametrów optymalizatora
print("\nParametry optymalizatora:")
print(model.optimizer.get_config())
