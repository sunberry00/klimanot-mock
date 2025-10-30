import numpy as np
import tensorflow as tf
from model_definition import create_model  # shared architecture

def initialize_global_model(initial_weights_file):
    """Create global model and save initial weights (.weights.h5)."""
    print("\n--- SERVER (Initialization) ---")
    model = create_model()
    # Build to ensure weight structure exists
    model.build((None, 16))
    model.save_weights(initial_weights_file)
    print(f"Global model initialized and weights saved: {initial_weights_file}")

def aggregate_weights(client_weight_files, new_global_weights_file, total_data_points, client_data_points):
    """Federated Averaging (weighted by client sample counts)."""
    print("\n--- SERVER (Aggregation) ---")
    if not client_weight_files:
        print("No client weights provided.")
        return

    # Load all client weights into memory using a temp model
    temp = create_model()
    temp.build((None, 16))
    all_weights = []
    for f in client_weight_files:
        temp.load_weights(f)
        all_weights.append([w.copy() for w in temp.get_weights()])

    # Weighted average
    factors = [n / total_data_points for n in client_data_points]
    num_layers = len(all_weights[0])
    avg = []
    for i in range(num_layers):
        layer_stack = [w[i] * factors[k] for k, w in enumerate(all_weights)]
        avg.append(np.sum(layer_stack, axis=0))

    # Set averaged weights and save as new global
    global_model = create_model()
    global_model.build((None, 16))
    global_model.set_weights(avg)
    global_model.save_weights(new_global_weights_file)
    print(f"New global weights saved: {new_global_weights_file}")
