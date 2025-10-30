import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from client import run_client_training
from server import initialize_global_model, aggregate_weights
from mock_sql_and_preprocessing import extract_one

# ============================================
#               Configuration
# ============================================

BASE = Path("/home/aivanets/klimanot-mock/")
DWH_BASE = BASE / "dwh_nodes"
NODES = ["node_a", "node_b", "node_c"]
ROUNDS = 2
INITIAL_WEIGHTS = "global_weights_round_0.weights.h5"
SEED = 42
np.random.seed(SEED)

# Scripts to copy to each node's model dir (initial distribution of code)
SCRIPTS_TO_SEND = ["client.py", "model_definition.py", "mock_sql_and_preprocessing.py"]

# Metrics file
METRICS_CSV = BASE / "federated_metrics.csv"

# ============================================
# Feature engineering (16-dim; matches model_definition input)
# ============================================

TRIAGE_LEVELS = ["1", "2", "3", "4", "5"]
TAGESZEIT_BUCKETS = ["Morgen", "Nachmittag", "Abend", "Nacht"]


def _one_hot(value, categories):
    vec = np.zeros(len(categories), dtype=float)
    try:
        idx = categories.index(str(value))
        vec[idx] = 1.0
    except ValueError:
        pass
    return vec


def _hash_one_hot(value, k=4):
    vec = np.zeros(k, dtype=float)
    if value is None:
        return vec
    h = abs(hash(str(value))) % k
    vec[h] = 1.0
    return vec


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _make_features(row):
    triage_oh = _one_hot(row.get("Triage-Score"), TRIAGE_LEVELS)  # 5
    tageszeit_oh = _one_hot(row.get("Tageszeit"), TAGESZEIT_BUCKETS)  # 4
    age_feat = np.array([_safe_float(row.get("Alter"), 0.0) / 100.0], dtype=float)  # 1
    cedis_oh = _hash_one_hot(row.get("CEDIS"), k=4)  # 4
    padding = np.zeros(2, dtype=float)  # 2
    feats = np.concatenate(
        [triage_oh, tageszeit_oh, age_feat, cedis_oh, padding], axis=0
    )
    assert feats.shape[0] == 16
    return feats


def _derive_label(row):
    code_key = "stationäre Aufnahme (code)"
    if code_key not in row:
        for k in row.keys():
            if "station" in k.lower() and "code" in k.lower():
                code_key = k
                break
    code = (row.get(code_key) or "").strip().lower() if code_key in row else ""
    if code:
        in_mark = {
            "i",
            "ip",
            "inpatient",
            "stationaer",
            "stationär",
            "ja",
            "yes",
            "y",
            "1",
            "true",
        }
        out_mark = {
            "o",
            "op",
            "outpatient",
            "ambulant",
            "nein",
            "no",
            "n",
            "0",
            "false",
        }
        if code in in_mark:
            return 1
        if code in out_mark:
            return 0
        try:
            return 1 if float(code) > 0 else 0
        except Exception:
            pass
    triage = str(row.get("Triage-Score") or "").strip()
    if triage in {"1", "2"}:
        return 1
    if triage in {"4", "5"}:
        return 0
    return np.random.randint(0, 2)


def _row_to_dict(row_obj):
    if isinstance(row_obj, dict):
        return row_obj
    return {k: row_obj[k] for k in row_obj.index}


def _load_from_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        return [_row_to_dict(r) for _, r in df.iterrows()]
    except Exception as e:
        print(f"[WARN] Failed to read CSV {csv_path}: {e}")
        return []


def _detect_csv_for_node(node_dir: Path, node_name: str):
    candidates = [
        node_dir / "res" / "cda_summary.csv",
        BASE / f"{node_name}_cda_summary.csv",
        BASE / "cda_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_node_dataset(node_name: str):
    node_dwh = DWH_BASE / node_name / "dwh"
    if not node_dwh.exists():
        print(f"[WARN] Node {node_name}: directory not found: {node_dwh}")
        return np.empty((0, 16), dtype=float), np.empty((0,), dtype=int)

    rows = []
    csv_path = _detect_csv_for_node(node_dwh, node_name)
    if csv_path is not None:
        print(f"Node {node_name}: loading CSV summary: {csv_path}")
        rows = _load_from_csv(csv_path)

    if not rows:
        xml_files = sorted(node_dwh.rglob("*.xml"))
        if not xml_files:
            print(f"[WARN] Node {node_name}: no XML files found in {node_dwh}")
            return np.empty((0, 16), dtype=float), np.empty((0,), dtype=int)
        for p in xml_files:
            try:
                rows.append(extract_one(p))
            except Exception as e:
                print(f"[WARN] Node {node_name}: failed to parse {p}: {e}")

    X = np.array([_make_features(r) for r in rows], dtype=float)
    y = np.array([_derive_label(r) for r in rows], dtype=int)
    print(f"Node {node_name}: loaded {len(rows)} records")
    return X, y


def simple_train_test_split(X, y, test_ratio=0.25, seed=SEED):
    if len(X) == 0:
        return (X, X, y, y)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_ratio))
    return X[:cut], X[cut:], y[:cut], y[cut:]


# ============================================
# Node I/O simulation (send code & weights)
# ============================================


def ensure_model_dir(node_name: str) -> Path:
    model_dir = DWH_BASE / node_name / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def send_code_to_node(node_name: str):
    model_dir = ensure_model_dir(node_name)
    for script in SCRIPTS_TO_SEND:
        src = BASE / "central" / script
        if src.exists():
            dest = model_dir / script
            shutil.copy2(src, dest)
            print(f"[SEND CODE] {src.name} -> {dest}")
        else:
            print(f"[WARN] Missing script in central: {src}")


def send_global_to_node(current_global_path: Path, node_name: str) -> Path:
    model_dir = ensure_model_dir(node_name)
    dest = model_dir / current_global_path.name
    shutil.copy2(current_global_path, dest)
    # Also keep a 'latest' copy for convenience
    latest = model_dir / "global_latest.weights.h5"
    shutil.copy2(current_global_path, latest)
    print(f"[SEND WEIGHTS] {current_global_path.name} -> {dest} (and -> {latest.name})")
    return dest


def broadcast_global_to_all_nodes(current_global_path: Path):
    for node in NODES:
        send_global_to_node(current_global_path, node)


def receive_client_from_node(
    node_name: str, client_local_weights: Path, round_id: int
) -> Path:
    central_name = f"{node_name}_updated_round_{round_id}.weights.h5"
    dest = BASE / central_name
    shutil.copy2(client_local_weights, dest)
    print(f"[RECV WEIGHTS] {client_local_weights} -> {dest}")
    return dest


def run_local_training_in_node(node_name: str, X_tr, y_tr, local_global_weights: Path):
    """
    Executes training on the node using exec() to simulate remote script execution.
    """
    model_dir = local_global_weights.parent
    cwd_backup = Path.cwd()
    try:
        os.chdir(model_dir)
        # Save local training data to file
        np.savez("local_data.npz", X=X_tr, y=y_tr)

        # Prepare environment for exec()
        exec_env = {
            "__name__": "__main__",
            "CLIENT_ID": node_name,
            "GLOBAL_WEIGHTS": local_global_weights.name,
        }

        # Execute client script within node folder
        with open("client.py", "r") as f:
            code = f.read()
        exec(code, exec_env)

        updated_rel = exec_env.get(
            "UPDATED_WEIGHTS_FILE", f"client_{node_name}_updated.weights.h5"
        )
        updated_path = model_dir / updated_rel
        print(f"[NODE {node_name}] Updated weights at {updated_path}")
        return updated_path
    finally:
        os.chdir(cwd_backup)


def evaluate_global(weights_file: Path, X_test, y_test):
    from model_definition import create_model

    model = create_model()
    model.build((None, 16))
    model.load_weights(str(weights_file))
    if len(X_test) == 0:
        print("[INFO] No test data available for evaluation.")
        return None
    preds = (model.predict(X_test, verbose=0).reshape(-1) >= 0.5).astype(int)
    acc = (preds == y_test).mean()
    print(
        f"[EVAL] Global model @ {weights_file.name}: accuracy = {acc:.3f} (n={len(y_test)})"
    )
    return float(acc)


def log_metric(round_id: int, acc: float, n_test: int):
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "round": round_id,
        "accuracy": acc if acc is not None else np.nan,
        "n_test": n_test,
    }
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(METRICS_CSV, index=False)
    print(f"[METRICS] appended to {METRICS_CSV} -> {row}")


# ============================================
# Main Orchestration
# ============================================


def main():
    print("\n==== MOCK FEDERATED LEARNING (code+weights sent to nodes) ====")
    os.chdir(BASE)

    for node in NODES:
        send_code_to_node(node)

    # 1) Load datasets per node and split
    node_sets = {}
    for node in NODES:
        X, y = load_node_dataset(node)
        X_tr, X_te, y_tr, y_te = simple_train_test_split(
            X, y, test_ratio=0.25, seed=SEED
        )
        node_sets[node] = {"train": (X_tr, y_tr), "test": (X_te, y_te), "n": len(X_tr)}

    total_train = sum(info["n"] for info in node_sets.values())
    if total_train == 0:
        print(
            "[ERROR] No training data on any node. Populate XMLs or CSV summaries and retry."
        )
        return

    # 2) Initialize model on central (creates INITIAL_WEIGHTS if missing)
    print("\n-- Initialize model on central (TensorFlow) --")
    initialize_global_model(INITIAL_WEIGHTS)
    current_global = BASE / INITIAL_WEIGHTS

    # 2b) Initial broadcast to all nodes (so each node has latest)
    print("\n-- Broadcast initial global weights to nodes --")
    broadcast_global_to_all_nodes(current_global)

    # 3) Federated rounds
    for r in range(1, ROUNDS + 1):
        print(f"\n========== ROUND {r} ==========")
        client_weight_files_central = []
        client_points = []
        total_points = 0

        # For each node: ensure latest weights, then train locally -> receive back
        for node in NODES:
            X_tr, y_tr = node_sets[node]["train"]
            if len(X_tr) == 0:
                print(f"Node {node}: no training data, skipping.")
                continue

            # (Re)send latest global to the node for this round
            local_copy = send_global_to_node(current_global, node)

            # TRAIN LOCALLY in node folder
            updated_local = run_local_training_in_node(node, X_tr, y_tr, local_copy)

            # RECEIVE updated weights to central
            central_copy = receive_client_from_node(node, updated_local, round_id=r)

            client_weight_files_central.append(str(central_copy))
            n_pts = len(X_tr)
            client_points.append(n_pts)
            total_points += n_pts

        if not client_weight_files_central:
            print("[WARN] No updates received; stopping.")
            break

        # AGGREGATE on central
        new_global = BASE / f"global_weights_round_{r}.weights.h5"
        aggregate_weights(
            client_weight_files=client_weight_files_central,
            new_global_weights_file=str(new_global),
            total_data_points=total_points,
            client_data_points=client_points,
        )
        current_global = new_global

        print("-- Broadcast updated global weights to nodes --")
        broadcast_global_to_all_nodes(current_global)

        # EVALUATE on combined held-out
        X_tests, y_tests = [], []
        for node in NODES:
            X_te, y_te = node_sets[node]["test"]
            if len(X_te):
                X_tests.append(X_te)
                y_tests.append(y_te)
        if X_tests:
            X_eval = np.vstack(X_tests)
            y_eval = np.concatenate(y_tests)
        else:
            X_eval = np.empty((0, 16), dtype=float)
            y_eval = np.empty((0,), dtype=int)
        acc = evaluate_global(current_global, X_eval, y_eval)
        n_test = int(len(y_eval))
        log_metric(r, acc, n_test)

    # Final summary
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
        print("\n===== FINAL SUMMARY =====")
        print(df.to_string(index=False))
    print("\n==== DONE ====")


if __name__ == "__main__":
    main()
