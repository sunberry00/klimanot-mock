import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- 0. Modell-Definitionsfunktion ---
# DIES IST DER WICHTIGSTE TEIL: Server und Clients MÜSSEN
# exakt die gleiche Funktion verwenden, um die Modellarchitektur zu erstellen.

def create_model():
    """Erstellt und kompiliert das Standard-Keras-Modell."""
    
    # Passen Sie 'input_features' an Ihre Daten an.
    input_features = 16 # NUR EIN BEISPIELWERT

    # Definieren des Modells (Ihr Code)
    model = Sequential([
        # 1. Versteckte Schicht (Hidden Layer)
        Dense(16, activation='relu', input_shape=(input_features,)),
        # 2. Versteckte Schicht
        Dense(8, activation='relu'),
        # Ausgabe-Schicht (Output Layer)
        Dense(1, activation='sigmoid')
    ])

    # --- Kompilieren des Modells ---
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model
