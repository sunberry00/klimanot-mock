import numpy as np
import tensorflow as tf
from model_definition import create_model  # shared architecture


def run_client_training(client_id, X_data, y_data, global_weights_file):
    """
    Simulates one client: load the global weights (already present in node folder),
    train locally on provided data, and save updated weights.
    """
    print(f"\n--- CLIENT {client_id} (Training) ---")
    model = create_model()
    # Ensure model is built before loading/saving weights (TF >=2.15 safety)
    model.build((None, 16))
    model.load_weights(global_weights_file)
    print(f"Client {client_id} loaded global weights: {global_weights_file}")

    # Local training
    if len(X_data) and len(y_data):
        print(f"Client {client_id} training locally on {len(X_data)} samples...")
        model.fit(X_data, y_data, epochs=3, verbose=0, batch_size=32)
    else:
        print(f"Client {client_id} has no data. Skipping training.")

    # Save updated weights
    updated_weights_file = f"client_{client_id}_updated.weights.h5"
    model.save_weights(updated_weights_file)
    print(f"Client {client_id} saved updated weights: {updated_weights_file}")
    return updated_weights_file


# --- EXECUTION ENTRYPOINT (when run via exec() or standalone) ---
if __name__ == "__main__":
    import numpy as np
    import os

    # Load serialized local training data
    if not os.path.exists("local_data.npz"):
        print("[ERROR] No local_data.npz found. Exiting.")
    else:
        data = np.load("local_data.npz")
        X_data, y_data = data["X"], data["y"]

        client_id = globals().get("CLIENT_ID", "unknown")
        global_weights = globals().get("GLOBAL_WEIGHTS", "global_latest.weights.h5")

        updated = run_client_training(
            client_id=client_id,
            X_data=X_data,
            y_data=y_data,
            global_weights_file=global_weights,
        )
        # For possible external usage
        globals()["UPDATED_WEIGHTS_FILE"] = updated
