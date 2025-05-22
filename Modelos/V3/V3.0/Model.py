import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Cargar y preparar datos
data = pd.read_csv('DataSet_v3.csv')
X = data.drop('Label', axis=1).values
y = data['Label'].values

# Normalización de características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Codificación de etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Arquitectura del modelo optimizada
model = models.Sequential([
    # Capa de entrada con más neuronas para capturar patrones complejos
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Capa intermedia
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Capa intermedia
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

      # Capa intermedia
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Capa intermedia adicional
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Capa de salida
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Callbacks para mejorar el entrenamiento
early_stopping = EarlyStopping(
    monitor='loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.2,
    patience=10,
    min_lr=0.00001
)

# Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=1024,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluación final
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.4f}')

# Guardar el modelo y el scaler
model.save('Model.keras')
np.save('scaler_params.npy', {
    'mean': scaler.mean_,
    'scale': scaler.scale_
})

