import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Cargar y preparar datos
data = pd.read_csv('DataSet_v2.csv')
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

# Función para aumentar datos
def augment_data(X, y, noise_factor=0.05):
    X_aug = X.copy()
    y_aug = y.copy()
    
    # Añadir ruido gaussiano
    noise = np.random.normal(0, noise_factor, X.shape)
    X_aug += noise
    
    # Añadir pequeñas variaciones aleatorias
    variation = np.random.uniform(-0.1, 0.1, X.shape)
    X_aug += variation
    
    return np.vstack((X, X_aug)), np.vstack((y, y_aug))

# Aumentar datos de entrenamiento
X_train_aug, y_train_aug = augment_data(X_train, y_train)

# Arquitectura del modelo optimizada con dropout
model = models.Sequential([
    # Capa de entrada con más neuronas para capturar patrones complejos
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Capas intermedias con dropout progresivo
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.45),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Capa de salida
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Callbacks para mejorar el entrenamiento
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Compilación del modelo con learning rate más bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo con datos aumentados
history = model.fit(
    X_train_aug, y_train_aug,
    epochs=5,
    batch_size=256,  # Batch size más pequeño para mejor generalización
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

