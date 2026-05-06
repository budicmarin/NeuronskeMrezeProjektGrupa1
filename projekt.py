import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


# --- FUNKCIJA ZA UČITAVANJE ---
def load_data(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0].astype(int)
    x = data[:, 1:]
    y[y == -1] = 0  # Klase su -1 i 1, prebacujemo u 0 i 1
    return x, y


# Učitavanje (sada će raditi jer smo ih preuzeli gore)
x_train_full, y_train_full = load_data("FordA_TRAIN.txt")
x_test_a, y_test_a = load_data("FordA_TEST.txt")
x_test_b, y_test_b = load_data("FordB_TEST.txt")

# Normalizacija (Zero-mean, Unit-variance)
x_train_full = (x_train_full - x_train_full.mean()) / x_train_full.std()
x_test_a = (x_test_a - x_test_a.mean()) / x_test_a.std()
x_test_b = (x_test_b - x_test_b.mean()) / x_test_b.std()

# Reshape za CNN/LSTM (samples, timesteps, features)
x_train_full = x_train_full.reshape((x_train_full.shape[0], x_train_full.shape[1], 1))
x_test_a = x_test_a.reshape((x_test_a.shape[0], x_test_a.shape[1], 1))
x_test_b = x_test_b.reshape((x_test_b.shape[0], x_test_b.shape[1], 1))

# Kreiranje validacijskog skupa
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# --- MODELI ---
input_shape = (x_train.shape[1], 1)

# 1. CNN Model
model_cnn = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv1D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation="sigmoid")
], name="CNN_Model")

# 2. LSTM Model
model_lstm = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(64),
    layers.Dense(1, activation="sigmoid")
], name="LSTM_Model")

models = [model_cnn, model_lstm]
histories = {}

# --- TRENING I VIZUALIZACIJA PROCESA ---
plt.figure(figsize=(14, 5))

for i, model in enumerate(models):
    print(f"\nTreniranje: {model.name}")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20, batch_size=64, verbose=0
    )
    histories[model.name] = history

    # Graf točnosti
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history['accuracy'], label='Trening')
    plt.plot(history.history['val_accuracy'], label='Validacija')
    plt.title(f'Točnost učenja: {model.name}')
    plt.xlabel('Epoha')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()

# --- EVALUACIJA (Testiranje) ---
datasets = [("FordA_TEST", x_test_a, y_test_a), ("FordB_TEST", x_test_b, y_test_b)]

for model in models:
    print(f"\n" + "=" * 30)
    print(f" REZULTATI ZA MODEL: {model.name} ")
    print("=" * 30)

    for name, x_t, y_t in datasets:
        y_pred = (model.predict(x_t, verbose=0) > 0.5).astype(int)

        print(f"\nIzvještaj za {name}:")
        print(classification_report(y_t, y_pred))

        # Matrica zabune (Confusion Matrix)
        cm = confusion_matrix(y_t, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title(f'{model.name} - {name}')
        plt.xlabel('Predviđeno')
        plt.ylabel('Stvarno')
        plt.show()