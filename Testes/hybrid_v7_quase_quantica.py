# -*- coding: utf-8 -*-

# =========================================================
# 0) PARAMETROS AJUSTAVEIS
# =========================================================

THREADS = 12

# --------------------------------------------------------
# BENCHMARK INICIAL
# --------------------------------------------------------

# Benchmark rapido para escolher CPU/GPU, batch_size, threads e workers.
# Ele executa poucos passos de treino em uma subamostra, para testar qual configuração funciona melhor
# Altere para True caso queira verificar e aplicar automaticamente a melhor configuração no PC atual
BENCHMARK_MODE = False  
BENCHMARK_SAMPLE_SIZE = 2048
BENCHMARK_STEPS = 2
BENCHMARK_DATALOADER_STEPS = 8
BENCHMARK_BATCH_CANDIDATES = [64, 128, 256, 512, 1024]          # lista de valores a serem testados
BENCHMARK_THREAD_CANDIDATES = [max(1, THREADS // 2), THREADS]   # lista de valores a serem testados
BENCHMARK_WORKER_CANDIDATES = [max(1, THREADS // 2), THREADS]   # lista de valores a serem testados
BENCHMARK_SAVE_RESULTS = True

# -------------------------------------------------------

OUTPUT_ROOT = "outputs_v7"
DATA_PATH = "features_geometry_70.csv"

# Parâmetros de salvamento parcial dos dados:
SAVE_PROGRESS_EVERY_EPOCH = True
SAVE_MODEL_EVERY_EPOCH = True
PROGRESS_DIR_NAME = "progress"

# Retomar a partir de um checkpoint no caso de parada parcial
RESUME_TRAINING = False                                                              # Alterar para True se quiser carregar um checkpoint salvo
RESUME_CHECKPOINT_PATH = ("outputs_direct_reupload/20260511-133349_direct_reupload_cpu_seed42_sample500_q8/progress/checkpoint_latest.pt") # Ajustar com o caminho do checkpoint salvo

SEED = 42

USE_TRAIN_SAMPLING = False
TRAIN_SAMPLE_SIZE = 1000
TEST_SIZE = 0.20
VAL_SIZE = 0.10

# Opcoes validas:
# "coords_only"                   -> usa apenas x/y dos landmarks: 42 colunas
# "extras_only"                   -> usa apenas as features extras detectadas: ate 9 colunas
# "coords_plus_selected_extras"   -> usa coords + extras selecionadas: ate 51 colunas
# "extras_plus_selected_coords"   -> usa extras + coords selecionadas
DIRECT_FEATURE_POLICY = "extras_only"

# Para extras_plus_selected_coords.
# Aceita indices de landmarks, nomes de colunas ou "all".
# Exemplos:
#   []                         -> nenhuma coordenada adicional
#   [0, 4, 8, 12, 16, 20]      -> x/y desses landmarks
#   ["x0", "y0", "x8", "y8"] -> colunas especificas
#   "all"                      -> todas as coordenadas
DIRECT_FEATURE_SELECTED_COORDS = []

# Para coords_plus_selected_extras.
# Use "all" para todas as extras encontradas em EXTRA_FEATURE_CANDIDATES,
# ou uma lista de nomes, por exemplo: ["dist_thumb_index", "angle_index"].
DIRECT_FEATURE_SELECTED_EXTRAS = "all"

# Sugestoes:
#   coords_only: 42
#   coords_plus_selected_extras: 51 quando DIRECT_FEATURE_SELECTED_EXTRAS="all"
#   extras_only: 9 quando todas as extras estiverem no CSV
#   extras_plus_selected_coords: 9 + 2 * quantidade_de_landmarks_selecionados
DIRECT_MAX_FEATURES = 8

N_CLASSES = None        # Definido depois com base na quantidade de classes na planilha
N_LANDMARKS = 21
N_QUBITS = 8
N_QUANTUM_INPUTS = None # Definido depois com base na quantidade de colunas de dados

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
DROPOUT = 0.30

LR_PATIENCE = 2
LR_FACTOR = 0.5

# Para treino com TorchConnector, o gradiente exato/padrao fica caro.
# SPSA usa poucas avaliacoes por passo.
USE_SPSA_GRADIENT = True
SPSA_EPSILON = 0.01

NUM_WORKERS = THREADS
PERSISTENT_WORKERS = True

AER_MAX_PARALLEL_THREADS = THREADS
AER_MAX_MEMORY_MB = 14000

USE_AER_GPU = True
AER_FALLBACK_TO_CPU = True
AER_DEVICE = "GPU" if USE_AER_GPU else "CPU"
AER_ENABLE_CUSTATEVEC = True
AER_METHOD = "statevector"

# Nome das colunas extras esperadas.
EXTRA_FEATURE_CANDIDATES = [
    "dist_thumb_index",
    "dist_index_middle",
    "dist_middle_ring",
    "dist_ring_pinky",
    #"dist_thumb_pinky",
    "angle_index",
    "angle_middle",
    "angle_ring",
    "angle_pinky",
]

# =========================================================
# 1) THREADS
# =========================================================
import os

for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[var] = str(THREADS)

# =========================================================
# 2) IMPORTS
# =========================================================
import copy
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, TensorDataset

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSAEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN


torch.set_num_threads(min(THREADS, os.cpu_count()))
torch.set_num_interop_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_AER_GPU else "cpu")

PIN_MEMORY = torch.cuda.is_available()

# =========================================================
# 3) UTILITARIOS GERAIS
# =========================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(text, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def append_experiment_summary(summary_row, filepath):
    filepath = Path(filepath)
    row_df = pd.DataFrame([summary_row])

    if filepath.exists():
        old_df = pd.read_csv(filepath)
        full_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        full_df = row_df

    full_df.to_csv(filepath, index=False)


def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_ints(values, minimum=0):
    out = []
    for value in values:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value >= minimum and value not in out:
            out.append(value)
    return out


def sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_class_weight_tensor(y, n_classes, device):
    weights = np.ones(int(n_classes), dtype=np.float32)
    present_classes = np.unique(y)

    if len(present_classes) > 0:
        present_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=y,
        ).astype(np.float32)
        weights[present_classes] = present_weights

    return torch.tensor(weights, dtype=torch.float32, device=device)


def validate_numeric_columns(df, feature_cols):
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"As colunas selecionadas precisam ser numericas: {non_numeric}")


# =========================================================
# 4) RELATORIOS E GRAFICOS
# =========================================================
def get_experiment_params(coord_cols, extra_cols, direct_feature_cols, benchmark_results=None):
    return {
        "THREADS": THREADS,
        "DATA_PATH": DATA_PATH,
        "SEED": SEED,
        "TEST_SIZE": TEST_SIZE,
        "VAL_SIZE": VAL_SIZE,
        "USE_TRAIN_SAMPLING": USE_TRAIN_SAMPLING,
        "TRAIN_SAMPLE_SIZE": TRAIN_SAMPLE_SIZE,
        "N_CLASSES": N_CLASSES,
        "N_LANDMARKS": N_LANDMARKS,
        "N_QUBITS": N_QUBITS,
        "N_QUANTUM_INPUTS": N_QUANTUM_INPUTS,
        "DIRECT_FEATURE_POLICY": DIRECT_FEATURE_POLICY,
        "DIRECT_FEATURE_SELECTED_COORDS": DIRECT_FEATURE_SELECTED_COORDS,
        "DIRECT_FEATURE_SELECTED_EXTRAS": DIRECT_FEATURE_SELECTED_EXTRAS,
        "DIRECT_MAX_FEATURES": DIRECT_MAX_FEATURES,
        "USE_SPSA_GRADIENT": USE_SPSA_GRADIENT,
        "SPSA_EPSILON": SPSA_EPSILON,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT": DROPOUT,
        "LR_PATIENCE": LR_PATIENCE,
        "LR_FACTOR": LR_FACTOR,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        "PERSISTENT_WORKERS": PERSISTENT_WORKERS,
        "AER_MAX_PARALLEL_THREADS": AER_MAX_PARALLEL_THREADS,
        "AER_MAX_MEMORY_MB": AER_MAX_MEMORY_MB,
        "USE_AER_GPU": USE_AER_GPU,
        "AER_FALLBACK_TO_CPU": AER_FALLBACK_TO_CPU,
        "AER_DEVICE": AER_DEVICE,
        "AER_ENABLE_CUSTATEVEC": AER_ENABLE_CUSTATEVEC,
        "AER_METHOD": AER_METHOD,
        "DEVICE": str(DEVICE),
        "BENCHMARK_MODE": BENCHMARK_MODE,
        "coord_columns": list(coord_cols),
        "extra_columns": list(extra_cols),
        "direct_feature_columns": list(direct_feature_cols),
        "benchmark_results": benchmark_results,
    }

def save_epoch_progress(
    exp_dir,
    epoch,
    history,
    model,
    optimizer,
    scheduler,
    best_state,
    best_val_f1,
    best_epoch,
    stale,
):
    ensure_dir(exp_dir)

    pd.DataFrame(history).to_csv(exp_dir / "history_live.csv", index=False)

    status = {
        "last_finished_epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "stale": int(stale),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(status, exp_dir / "status_live.json")

    checkpoint = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_state": best_state,
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "stale": int(stale),
        "history": history,
    }

    torch.save(checkpoint, exp_dir / "checkpoint_latest.pt")

    if best_state is not None:
        torch.save(best_state, exp_dir / "best_model_live.pt")


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    map_location=None,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_state": checkpoint.get("best_state"),
        "best_val_f1": checkpoint.get("best_val_f1", -1.0),
        "best_epoch": checkpoint.get("best_epoch", 0),
        "stale": checkpoint.get("stale", 0),
        "history": checkpoint.get("history", []),
    }

def plot_and_save_confusion_matrix(cm, class_names, filepath):
    fig_w = max(10, len(class_names) * 0.55)
    fig_h = max(8, len(class_names) * 0.45)
    font_size = max(5, min(10, int(180 / max(1, len(class_names)))))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Matriz de Confusao")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color='white' if cm[i, j] < cm.max()/2 else 'black',
                fontsize=font_size,
            )

    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()


def save_per_class_metrics_csv(report_dict, filepath):
    rows = []
    for class_name, metrics in report_dict.items():
        if class_name in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append(
            {
                "class": class_name,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )

    df_metrics = pd.DataFrame(rows).sort_values(by="f1_score", ascending=False)
    df_metrics.to_csv(filepath, index=False)
    return df_metrics


def generate_experiment_id():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sampling = f"sample{TRAIN_SAMPLE_SIZE}" if USE_TRAIN_SAMPLING else "fulltrain"
    device_tag = str(AER_DEVICE).lower()
    return f"{timestamp}_direct_reupload_{device_tag}_seed{SEED}_{sampling}_q{N_QUBITS}"


def build_split_summary(total_n, train_n, val_n, test_n):
    return {
        "total_samples": int(total_n),
        "train_samples": int(train_n),
        "val_samples": int(val_n),
        "test_samples": int(test_n),
        "train_fraction_total": float(train_n / total_n),
        "val_fraction_total": float(val_n / total_n),
        "test_fraction_total": float(test_n / total_n),
    }


def save_experiment_report(
    output_root,
    experiment_id,
    params,
    split_summary,
    history,
    best_state,
    best_val_f1,
    best_epoch,
    test_acc,
    test_f1,
    y_true,
    y_pred,
    class_names,
):
    output_root = Path(output_root)
    exp_dir = output_root / experiment_id
    ensure_dir(exp_dir)

    save_json(params, exp_dir / "params.json")
    save_json(split_summary, exp_dir / "split_summary.json")
    pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    save_json(report_dict, exp_dir / "classification_report.json")
    save_text(report_txt, exp_dir / "classification_report.txt")
    save_per_class_metrics_csv(report_dict, exp_dir / "per_class_metrics.csv")

    cm = confusion_matrix(y_true, y_pred)
    np.save(exp_dir / "confusion_matrix.npy", cm)
    plot_and_save_confusion_matrix(cm, class_names, exp_dir / "confusion_matrix.png")

    final_summary = {
        "experiment_id": experiment_id,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        **params,
        **split_summary,
    }
    save_json(final_summary, exp_dir / "summary.json")
    torch.save(best_state, exp_dir / "best_model.pt")
    append_experiment_summary(final_summary, output_root / "experiments_summary.csv")
    return exp_dir


# =========================================================
# 5) DETECCAO E SELECAO DE COLUNAS
# =========================================================
def detect_coordinate_columns(columns, n_landmarks=N_LANDMARKS):
    cols = list(columns)
    detected = []

    for i in range(n_landmarks):
        x_candidates = [f"x{i}", f"x_{i}"]
        y_candidates = [f"y{i}", f"y_{i}"]

        x_col = next((c for c in x_candidates if c in cols), None)
        y_col = next((c for c in y_candidates if c in cols), None)

        if x_col is None or y_col is None:
            raise ValueError(
                f"Nao foi possivel localizar as colunas do landmark {i}. "
                f"Esperado algo como x{i}/y{i} ou x_{i}/y_{i}."
            )
        detected.extend([x_col, y_col])

    return detected


def detect_extra_columns(columns, candidates):
    return [c for c in candidates if c in columns]


def resolve_selected_coordinate_columns(coord_cols, selected_coords):
    if selected_coords is None or selected_coords == []:
        return []
    if isinstance(selected_coords, str):
        if selected_coords.lower() == "all":
            return list(coord_cols)
        selected_coords = [selected_coords]

    selected = []
    for item in selected_coords:
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            landmark_idx = int(item)
            if not 0 <= landmark_idx < N_LANDMARKS:
                raise ValueError(f"Landmark invalido em DIRECT_FEATURE_SELECTED_COORDS: {item}")
            selected.extend([coord_cols[2 * landmark_idx], coord_cols[2 * landmark_idx + 1]])
        elif isinstance(item, str) and item in coord_cols:
            selected.append(item)
        else:
            raise ValueError(
                "Entrada invalida em DIRECT_FEATURE_SELECTED_COORDS. "
                f"Use indices 0..{N_LANDMARKS - 1}, nomes de colunas, ou 'all'. Valor: {item}"
            )

    return list(dict.fromkeys(selected))


def resolve_selected_extra_columns(extra_cols, selected_extras):
    if selected_extras is None or selected_extras == []:
        return []
    if isinstance(selected_extras, str):
        if selected_extras.lower() == "all":
            return list(extra_cols)
        selected_extras = [selected_extras]

    missing = [c for c in selected_extras if c not in extra_cols]
    if missing:
        raise ValueError(
            "As seguintes extras foram solicitadas, mas nao foram encontradas no CSV: "
            f"{missing}. Extras detectadas: {extra_cols}"
        )

    return list(dict.fromkeys(selected_extras))


def build_direct_feature_columns(
    df_columns,
    coord_cols,
    extra_cols,
    policy,
    selected_coords,
    selected_extras,
    max_features,
):
    valid_policies = {
        "coords_only",
        "extras_only",
        "coords_plus_selected_extras",
        "extras_plus_selected_coords",
    }
    if policy not in valid_policies:
        raise ValueError(f"DIRECT_FEATURE_POLICY invalida: {policy}. Opcoes: {sorted(valid_policies)}")

    selected_coord_cols = resolve_selected_coordinate_columns(coord_cols, selected_coords)
    selected_extra_cols = resolve_selected_extra_columns(extra_cols, selected_extras)

    if policy == "coords_only":
        feature_cols = list(coord_cols)
    elif policy == "extras_only":
        feature_cols = list(extra_cols)
    elif policy == "coords_plus_selected_extras":
        feature_cols = list(coord_cols) + selected_extra_cols
    else:  # extras_plus_selected_coords
        feature_cols = list(extra_cols) + selected_coord_cols

    feature_cols = list(dict.fromkeys(feature_cols))
    missing = [c for c in feature_cols if c not in df_columns]
    if missing:
        raise ValueError(f"Colunas selecionadas nao existem no CSV: {missing}")

    if max_features is not None and max_features > 0 and len(feature_cols) > max_features:
        print(
            f"[INFO] Limitando direct_feature_cols de {len(feature_cols)} "
            f"para {max_features} colunas para manter o custo do circuito controlado."
        )
        feature_cols = feature_cols[:max_features]

    if not feature_cols:
        raise ValueError(
            "Nenhuma coluna foi selecionada para entrada direta. "
            "Revise DIRECT_FEATURE_POLICY, DIRECT_FEATURE_SELECTED_COORDS e DIRECT_FEATURE_SELECTED_EXTRAS."
        )

    return feature_cols, selected_coord_cols, selected_extra_cols


# =========================================================
# 6) DATASET E PREPROCESSAMENTO DIRETO
# =========================================================
class FeatureDataset(Dataset):
    def __init__(self, x, labels):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


class Preprocessor:
    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)
        self.scaler = MinMaxScaler(feature_range=(0.0, math.pi))

    def fit_transform(self, df_train):
        train_x = df_train[self.feature_cols].to_numpy(dtype=np.float32)
        return self.scaler.fit_transform(train_x).astype(np.float32)

    def transform(self, df):
        x = df[self.feature_cols].to_numpy(dtype=np.float32)
        return self.scaler.transform(x).astype(np.float32)


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers):
    num_workers = int(num_workers)
    effective_persistent_workers = bool(persistent_workers and num_workers > 0)
    effective_pin_memory = bool(pin_memory and torch.cuda.is_available())

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_persistent_workers,
    )


def make_data_loaders(x_train, y_train, x_val, y_val, x_test, y_test):
    train_ds = FeatureDataset(x_train, y_train)
    val_ds = FeatureDataset(x_val, y_val)
    test_ds = FeatureDataset(x_test, y_test)

    train_loader = make_dataloader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )
    return train_loader, val_loader, test_loader


# =========================================================
# 7) SPLIT E AMOSTRAGEM
# =========================================================
def stratified_sample_training_df(df, idx_train, y_all):
    if not USE_TRAIN_SAMPLING:
        return df.iloc[idx_train].copy().reset_index(drop=True), y_all[idx_train]

    desired = min(TRAIN_SAMPLE_SIZE, len(idx_train))
    if desired >= len(idx_train):
        return df.iloc[idx_train].copy().reset_index(drop=True), y_all[idx_train]

    df_train_full = df.iloc[idx_train].copy()
    y_train_full = y_all[idx_train]
    parts = []

    classes, counts = np.unique(y_train_full, return_counts=True)
    raw_targets = counts / counts.sum() * desired
    per_class = np.floor(raw_targets).astype(int)
    remainder = int(desired - per_class.sum())

    if remainder > 0:
        order = np.argsort(raw_targets - per_class)[::-1]
        for j in order[:remainder]:
            per_class[j] += 1

    for cls, n_take in zip(classes, per_class):
        cls_idx = np.where(y_train_full == cls)[0]
        if n_take <= 0:
            continue
        replace = n_take > len(cls_idx)
        chosen = np.random.choice(cls_idx, size=n_take, replace=replace)
        parts.append(df_train_full.iloc[chosen])

    df_train = pd.concat(parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df_train, None


def prepare_dataframes(df):
    if "class" not in df.columns:
        raise ValueError("O CSV precisa conter a coluna 'class'.")

    if TEST_SIZE + VAL_SIZE >= 1.0:
        raise ValueError("TEST_SIZE + VAL_SIZE precisa ser menor que 1.0.")

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["class"].to_numpy())
    class_names = label_encoder.classes_.tolist()

    idx_all = np.arange(len(df))

    if USE_TRAIN_SAMPLING:
        train_target = min(TRAIN_SAMPLE_SIZE, len(df))
        val_target = max(1, int(round(train_target * VAL_SIZE)))
        test_target = max(1, int(round(train_target * TEST_SIZE)))
        total_target = train_target + val_target + test_target

        if total_target > len(df):
            raise ValueError(
                f"Amostra total solicitada ({total_target}) excede o total do CSV ({len(df)})."
            )

        sampled_idx, _, sampled_y, _ = train_test_split(
            idx_all,
            y_all,
            train_size=total_target,
            random_state=SEED,
            stratify=y_all,
        )

        idx_temp, idx_test, y_temp, _ = train_test_split(
            sampled_idx,
            sampled_y,
            test_size=test_target,
            random_state=SEED,
            stratify=sampled_y,
        )

        idx_train, idx_val, _, _ = train_test_split(
            idx_temp,
            y_temp,
            test_size=val_target,
            random_state=SEED,
            stratify=y_temp,
        )

        df_train = df.iloc[idx_train].copy().reset_index(drop=True)
        y_train = y_all[idx_train]

    else:
        idx_temp, idx_test, y_temp, _ = train_test_split(
            idx_all,
            y_all,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y_all,
        )

        relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

        idx_train, idx_val, _, _ = train_test_split(
            idx_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=SEED,
            stratify=y_temp,
        )

        df_train = df.iloc[idx_train].copy().reset_index(drop=True)
        y_train = y_all[idx_train]

    df_val = df.iloc[idx_val].copy().reset_index(drop=True)
    y_val = y_all[idx_val]

    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    y_test = y_all[idx_test]

    split_summary = build_split_summary(
        total_n=len(df),
        train_n=len(df_train),
        val_n=len(df_val),
        test_n=len(df_test),
    )

    return df_train, y_train, df_val, y_val, df_test, y_test, label_encoder, class_names, split_summary

# =========================================================
# 8) CIRCUITO QUANTICO COM DATA RE-UPLOADING
# =========================================================
def z_observables(n):
    obs = []

    # Z
    for i in range(n):
        label = ["I"] * n
        label[i] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))

    # ZZ
    for i in range(n - 1):
        label = ["I"] * n
        label[i] = "Z"
        label[i + 1] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))

    # XX
    for i in range(n - 1):
        label = ["I"] * n
        label[i] = "X"
        label[i + 1] = "X"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))

    return obs


def build_data_reupload_circuit(num_qubits, num_features, data_vector):
    qc_reupload = QuantumCircuit(num_qubits)
    num_reuploads = math.ceil(num_features / num_qubits)
    feature_idx = 0

    for _ in range(num_reuploads):
        for q in range(num_qubits):
            if feature_idx < num_features:
                qc_reupload.ry(data_vector[feature_idx], q)
                feature_idx += 1

        for q in range(num_qubits - 1):
            qc_reupload.cz(q, q + 1)

        qc_reupload.barrier()

    return qc_reupload


def build_quantum_components(n_quantum_inputs):
    data_vector = ParameterVector("x", n_quantum_inputs)
    reupload = build_data_reupload_circuit(
        num_qubits=N_QUBITS,
        num_features=n_quantum_inputs,
        data_vector=data_vector,
    )
    ansatz = real_amplitudes(num_qubits=N_QUBITS, reps=2)

    qc = QuantumCircuit(N_QUBITS)
    qc.compose(reupload, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc = qc.decompose(reps=2)

    observables = z_observables(N_QUBITS)
    return qc, data_vector, ansatz, observables


def make_aer_estimator(aer_device=None, max_parallel_threads=None):
    aer_device = (aer_device or AER_DEVICE).upper()
    use_gpu = aer_device == "GPU"
    max_parallel_threads = AER_MAX_PARALLEL_THREADS if max_parallel_threads is None else int(max_parallel_threads)

    backend_options = {
        "method": AER_METHOD,
        "device": aer_device,
        "max_parallel_threads": max_parallel_threads,
        "max_parallel_experiments": 0,
        "max_memory_mb": int(AER_MAX_MEMORY_MB) if AER_MAX_MEMORY_MB else 0,
    }

    if use_gpu:
        backend_options["batched_shots_gpu"] = True
        if AER_ENABLE_CUSTATEVEC:
            backend_options["cuStateVec_enable"] = True

    return AerEstimator(
        options={
            "backend_options": backend_options,
            "run_options": {
                "shots": None,
            },
        }
    )


def sanity_check_aer_estimator(estimator, expected_device):
    test_qc = QuantumCircuit(1)
    test_qc.h(0)
    test_obs = SparsePauliOp.from_list([("Z", 1.0)])
    try:
        job = estimator.run([(test_qc, test_obs)])
        _ = job.result()
        print(f"[AER] Estimator inicializado com device={expected_device}.")
        return True
    except Exception as exc:
        print(f"[AER] Falha ao inicializar device={expected_device}: {exc}")
        return False


def get_working_estimator(aer_device=None, max_parallel_threads=None):
    requested_device = (aer_device or AER_DEVICE).upper()
    estimator = make_aer_estimator(requested_device, max_parallel_threads=max_parallel_threads)
    if sanity_check_aer_estimator(estimator, requested_device):
        return estimator, requested_device

    if requested_device == "GPU" and AER_FALLBACK_TO_CPU:
        print("[AER] Usando fallback para CPU.")
        estimator = make_aer_estimator("CPU", max_parallel_threads=max_parallel_threads)
        if sanity_check_aer_estimator(estimator, "CPU"):
            return estimator, "CPU"

    raise RuntimeError(
        f"Aer device={requested_device} solicitado, mas indisponivel. "
        "Verifique qiskit-aer, qiskit-aer-gpu, CUDA ou use AER_FALLBACK_TO_CPU=True."
    )


def make_quantum_layer(qc, data_vector, ansatz, observables, estimator):
    gradient = None
    if USE_SPSA_GRADIENT:
        gradient = SPSAEstimatorGradient(
            estimator=estimator,
            epsilon=SPSA_EPSILON,
            seed=SEED,
        )

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(data_vector),
        weight_params=list(ansatz.parameters),
        observables=observables,
        estimator=estimator,
        gradient=gradient,
        input_gradients=False,
    )
    return TorchConnector(qnn), qnn


# =========================================================
# 9) MODELO DIRETO DATA RE-UPLOADING
# =========================================================
class HybridModel(nn.Module):
    def __init__(self, quantum_layer, n_observables, n_classes, dropout=DROPOUT):
        super().__init__()
        self.quantum = quantum_layer
        self.head = nn.Sequential(
            nn.Linear(n_observables, n_classes),
            )   

    def forward(self, x):
        q_out = self.quantum(x)
        return self.head(q_out)


def build_model(qc, data_vector, ansatz, observables, n_classes, aer_device, max_parallel_threads=None):
    estimator, actual_aer_device = get_working_estimator(aer_device, max_parallel_threads=max_parallel_threads)
    quantum_layer, qnn = make_quantum_layer(qc, data_vector, ansatz, observables, estimator)
    torch_device = torch.device("cuda" if torch.cuda.is_available() and actual_aer_device == "GPU" else "cpu")

    model = HybridModel(
        quantum_layer=quantum_layer,
        n_observables=len(observables),
        n_classes=n_classes,
        dropout=DROPOUT,
    ).to(torch_device)

    return model, qnn, estimator, actual_aer_device, torch_device


# =========================================================
# 10) BENCHMARK RAPIDO
# =========================================================
def make_benchmark_subset(x_train, y_train, sample_size):
    sample_size = min(int(sample_size), len(x_train))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x_train), size=sample_size, replace=False)
    return x_train[idx], y_train[idx]


def benchmark_dataloader_workers(x_subset, y_subset, batch_size, worker_candidates):
    dataset = FeatureDataset(x_subset, y_subset)
    results = []

    for workers in unique_ints(worker_candidates, minimum=0):
        try:
            loader = make_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS,
            )
            seen = 0
            start = time.perf_counter()
            for step, (xb, yb) in enumerate(loader):
                seen += len(yb)
                if step + 1 >= BENCHMARK_DATALOADER_STEPS:
                    break
            elapsed = max(time.perf_counter() - start, 1e-9)
            results.append(
                {
                    "num_workers": int(workers),
                    "elapsed_sec": float(elapsed),
                    "samples_per_sec": float(seen / elapsed),
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "num_workers": int(workers),
                    "elapsed_sec": None,
                    "samples_per_sec": 0.0,
                    "status": "error",
                    "error": str(exc),
                }
            )

    ok_results = [r for r in results if r["status"] == "ok"]
    best_workers = max(ok_results, key=lambda r: r["samples_per_sec"])["num_workers"] if ok_results else 0
    return best_workers, results


def benchmark_training_config(
    x_subset,
    y_subset,
    qc,
    data_vector,
    ansatz,
    observables,
    n_classes,
    aer_device,
    batch_size,
    threads,
):
    try:
        torch.set_num_threads(int(threads))
    except RuntimeError:
        pass

    model = None
    try:
        model, _, _, actual_aer_device, torch_device = build_model(
            qc=qc,
            data_vector=data_vector,
            ansatz=ansatz,
            observables=observables,
            n_classes=n_classes,
            aer_device=aer_device,
            max_parallel_threads=threads,
        )

        class_weights = make_class_weight_tensor(y_subset, n_classes, torch_device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        dataset = FeatureDataset(x_subset, y_subset)
        loader = make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=PIN_MEMORY,
            persistent_workers=False,
        )

        model.train()
        seen = 0
        sync_if_cuda(torch_device)
        start = time.perf_counter()

        for step, (xb, yb) in enumerate(loader):
            xb = xb.to(torch_device, non_blocking=True)
            yb = yb.to(torch_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            seen += xb.size(0)
            if step + 1 >= BENCHMARK_STEPS:
                break

        sync_if_cuda(torch_device)
        elapsed = max(time.perf_counter() - start, 1e-9)
        return {
            "aer_device_requested": aer_device,
            "aer_device_actual": actual_aer_device,
            "torch_device": str(torch_device),
            "batch_size": int(batch_size),
            "threads": int(threads),
            "elapsed_sec": float(elapsed),
            "samples_seen": int(seen),
            "samples_per_sec": float(seen / elapsed),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "aer_device_requested": aer_device,
            "aer_device_actual": None,
            "torch_device": None,
            "batch_size": int(batch_size),
            "threads": int(threads),
            "elapsed_sec": None,
            "samples_seen": 0,
            "samples_per_sec": 0.0,
            "status": "error",
            "error": str(exc),
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_quick_benchmark(x_train, y_train, qc, data_vector, ansatz, observables, n_classes):
    print("\n===== BENCHMARK RAPIDO =====")
    x_subset, y_subset = make_benchmark_subset(x_train, y_train, BENCHMARK_SAMPLE_SIZE)

    batch_candidates = unique_ints([*BENCHMARK_BATCH_CANDIDATES, BATCH_SIZE], minimum=8)
    thread_candidates = unique_ints([*BENCHMARK_THREAD_CANDIDATES, THREADS], minimum=4)
    worker_candidates = unique_ints([*BENCHMARK_WORKER_CANDIDATES, NUM_WORKERS], minimum=0)

    device_candidates = ["CPU"]
    if USE_AER_GPU and torch.cuda.is_available():
        device_candidates.insert(0, "GPU")

    training_results = []
    for aer_device in device_candidates:
        for threads in thread_candidates:
            for batch_size in batch_candidates:
                print(f"[BENCH] device={aer_device} threads={threads} batch={batch_size}")
                result = benchmark_training_config(
                    x_subset=x_subset,
                    y_subset=y_subset,
                    qc=qc,
                    data_vector=data_vector,
                    ansatz=ansatz,
                    observables=observables,
                    n_classes=n_classes,
                    aer_device=aer_device,
                    batch_size=batch_size,
                    threads=threads,
                )
                training_results.append(result)
                if result["status"] == "ok":
                    print(f"        {result['samples_per_sec']:.2f} amostras/s")
                else:
                    print(f"        erro: {result.get('error')}")

    ok_training = [r for r in training_results if r["status"] == "ok"]
    if ok_training:
        best_training = max(ok_training, key=lambda r: r["samples_per_sec"])
    else:
        best_training = {
            "aer_device_actual": AER_DEVICE,
            "torch_device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "threads": THREADS,
            "samples_per_sec": 0.0,
            "status": "fallback",
        }

    best_workers, worker_results = benchmark_dataloader_workers(
        x_subset=x_subset,
        y_subset=y_subset,
        batch_size=int(best_training["batch_size"]),
        worker_candidates=worker_candidates,
    )

    best_config = {
        "aer_device": best_training.get("aer_device_actual") or AER_DEVICE,
        "torch_device": best_training.get("torch_device") or str(DEVICE),
        "batch_size": int(best_training["batch_size"]),
        "threads": int(best_training["threads"]),
        "num_workers": int(best_workers),
        "training_samples_per_sec": float(best_training.get("samples_per_sec", 0.0)),
    }

    benchmark_results = {
        "sample_size": int(len(x_subset)),
        "steps_per_training_config": int(BENCHMARK_STEPS),
        "dataloader_steps": int(BENCHMARK_DATALOADER_STEPS),
        "best_config": best_config,
        "training_results": training_results,
        "worker_results": worker_results,
    }

    print("[BENCH] Melhor configuracao rapida:", best_config)
    print("==============================\n")

    if BENCHMARK_SAVE_RESULTS:
        ensure_dir(OUTPUT_ROOT)
        save_json(benchmark_results, Path(OUTPUT_ROOT) / "benchmark_last.json")

    return benchmark_results


def apply_benchmark_config(benchmark_results):
    global THREADS, BATCH_SIZE, NUM_WORKERS, USE_AER_GPU, AER_DEVICE, AER_MAX_PARALLEL_THREADS, DEVICE

    if not benchmark_results or "best_config" not in benchmark_results:
        return

    best = benchmark_results["best_config"]
    THREADS = int(best["threads"])
    BATCH_SIZE = int(best["batch_size"])
    NUM_WORKERS = int(best["num_workers"])
    AER_DEVICE = str(best["aer_device"]).upper()
    USE_AER_GPU = AER_DEVICE == "GPU"
    AER_MAX_PARALLEL_THREADS = THREADS
    DEVICE = torch.device(best["torch_device"])

    try:
        torch.set_num_threads(THREADS)
    except RuntimeError:
        pass


# =========================================================
# 11) TREINO E AVALIACAO
# =========================================================
def train_model(model, train_loader, val_loader, y_train, n_classes, device, progress_dir=None, resume_checkpoint_path=None,):
    class_weights = make_class_weight_tensor(y_train, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start_epoch = 0

    if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
        print(f"[CHECKPOINT] Carregando: {resume_checkpoint_path}")

        checkpoint_data = load_training_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )

        start_epoch = checkpoint_data["epoch"]

        best_state = checkpoint_data["best_state"]
        best_val_f1 = checkpoint_data["best_val_f1"]
        best_epoch = checkpoint_data["best_epoch"]
        stale = checkpoint_data["stale"]
        history = checkpoint_data["history"]

        print(
            f"[CHECKPOINT] Retomando da epoch {start_epoch} "
            f"(best_val_f1={best_val_f1:.4f})"
        )

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro")

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_f1": float(train_f1),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": float(epoch_time),
            }
        )

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1

        if SAVE_PROGRESS_EVERY_EPOCH and progress_dir is not None:
            save_epoch_progress(
                exp_dir=progress_dir,
                epoch=epoch + 1,
                history=history,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_state=best_state,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                stale=stale,
            )

        if stale >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Treinamento nao produziu checkpoint valido.")

    model.load_state_dict(best_state)
    return best_state, best_val_f1, best_epoch, history


def evaluate_model(model, test_loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(yb.numpy())

    test_acc = accuracy_score(targets, preds)
    test_f1 = f1_score(targets, preds, average="macro")
    return test_acc, test_f1, targets, preds


# =========================================================
# 12) MAIN
# =========================================================
def main():
    global N_CLASSES, N_QUANTUM_INPUTS, DEVICE, AER_DEVICE, USE_AER_GPU

    start_time = time.perf_counter()
    set_reproducibility(SEED)

    print(f"Torch CUDA disponivel: {torch.cuda.is_available()}")
    print(f"Device inicial: {DEVICE}")

    df = pd.read_csv(DATA_PATH)
    coord_cols = detect_coordinate_columns(df.columns, n_landmarks=N_LANDMARKS)
    extra_cols = detect_extra_columns(df.columns, EXTRA_FEATURE_CANDIDATES)

    ignored_cols = {"class", "image_name", *coord_cols, *extra_cols}
    unused_cols = [c for c in df.columns if c not in ignored_cols]
    if unused_cols:
        print("[INFO] Colunas adicionais nao usadas pela entrada direta:", unused_cols)

    direct_feature_cols, selected_coord_cols, selected_extra_cols = build_direct_feature_columns(
        df_columns=df.columns,
        coord_cols=coord_cols,
        extra_cols=extra_cols,
        policy=DIRECT_FEATURE_POLICY,
        selected_coords=DIRECT_FEATURE_SELECTED_COORDS,
        selected_extras=DIRECT_FEATURE_SELECTED_EXTRAS,
        max_features=DIRECT_MAX_FEATURES,
    )
    validate_numeric_columns(df, direct_feature_cols)
    N_QUANTUM_INPUTS = len(direct_feature_cols)

    (
        df_train,
        y_train,
        df_val,
        y_val,
        df_test,
        y_test,
        _label_encoder,
        class_names,
        split_summary,
    ) = prepare_dataframes(df)
    N_CLASSES = len(class_names)

    print("\n===== DADOS DO EXPERIMENTO =====")
    print(f"Total:        {len(df)}")
    print(f"Treino:       {len(df_train)} ({len(df_train) / len(df):.2%})")
    print(f"Validacao:    {len(df_val)} ({len(df_val) / len(df):.2%})")
    print(f"Teste:        {len(df_test)} ({len(df_test) / len(df):.2%})")
    print(f"Coord cols:   {len(coord_cols)}")
    print(f"Extra cols:   {len(extra_cols)} -> {extra_cols}")
    print(f"Selected coord cols: {len(selected_coord_cols)} -> {selected_coord_cols}")
    print(f"Selected extra cols: {len(selected_extra_cols)} -> {selected_extra_cols}")
    print(f"Direct cols:  {len(direct_feature_cols)}")
    print(f"Direct policy:{DIRECT_FEATURE_POLICY}; max_features={DIRECT_MAX_FEATURES}")
    print("=================================\n")

    preprocessor = Preprocessor(feature_cols=direct_feature_cols)
    x_train = preprocessor.fit_transform(df_train)
    x_val = preprocessor.transform(df_val)
    x_test = preprocessor.transform(df_test)

    qc, data_vector, ansatz, observables = build_quantum_components(N_QUANTUM_INPUTS)
    print("Qubits:", qc.num_qubits)
    print("Quantum inputs:", N_QUANTUM_INPUTS)
    print("Trainable ansatz parameters:", len(ansatz.parameters))
    print("Circuit parameters:", qc.num_parameters)
    print("Observables:", len(observables))

    benchmark_results = None
    if BENCHMARK_MODE:
        benchmark_results = run_quick_benchmark(
            x_train=x_train,
            y_train=y_train,
            qc=qc,
            data_vector=data_vector,
            ansatz=ansatz,
            observables=observables,
            n_classes=N_CLASSES,
        )
        apply_benchmark_config(benchmark_results)
        print(
            f"[BENCH] Configuracao aplicada: AER_DEVICE={AER_DEVICE}, DEVICE={DEVICE}, "
            f"BATCH_SIZE={BATCH_SIZE}, THREADS={THREADS}, NUM_WORKERS={NUM_WORKERS}"
        )

    train_loader, val_loader, test_loader = make_data_loaders(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    model, qnn, _estimator, actual_aer_device, actual_torch_device = build_model(
        qc=qc,
        data_vector=data_vector,
        ansatz=ansatz,
        observables=observables,
        n_classes=N_CLASSES,
        aer_device=AER_DEVICE,
        max_parallel_threads=AER_MAX_PARALLEL_THREADS,
    )
    AER_DEVICE = actual_aer_device
    USE_AER_GPU = AER_DEVICE == "GPU"
    DEVICE = actual_torch_device

    print("QNN inputs:", qnn.num_inputs)
    print("QNN weights:", qnn.num_weights)
    print("QNN output_shape:", qnn.output_shape)
    print(f"Treinando com AER_DEVICE={AER_DEVICE}; torch DEVICE={DEVICE}")

    experiment_id = generate_experiment_id()
    progress_dir = Path(OUTPUT_ROOT) / experiment_id / PROGRESS_DIR_NAME
    ensure_dir(progress_dir)

    best_state, best_val_f1, best_epoch, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=y_train,
        n_classes=N_CLASSES,
        device=DEVICE,
        progress_dir=progress_dir,
        resume_checkpoint_path=(
            RESUME_CHECKPOINT_PATH
            if RESUME_TRAINING
            else None
            ),
    )

    test_acc, test_f1, targets, preds = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
    )

    print(f"TEST ACC = {test_acc:.4f}")
    print(f"TEST F1  = {test_f1:.4f}")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Tempo total: {total_time / 3600:.2f} horas")

    params = get_experiment_params(
        coord_cols=coord_cols,
        extra_cols=extra_cols,
        direct_feature_cols=direct_feature_cols,
        benchmark_results=benchmark_results,
    )
    params["best_epoch"] = int(best_epoch)
    params["N_OBSERVABLES"] = int(len(observables))
    params["total_runtime_sec"] = float(total_time)
    params["total_runtime_min"] = float(total_time / 60)
    params["total_runtime_hours"] = float(total_time / 3600)

    exp_dir = save_experiment_report(
        output_root=OUTPUT_ROOT,
        experiment_id=experiment_id,
        params=params,
        split_summary=split_summary,
        history=history,
        best_state=best_state,
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        test_acc=test_acc,
        test_f1=test_f1,
        y_true=targets,
        y_pred=preds,
        class_names=class_names,
    )

    print(f"Relatorio salvo em: {exp_dir}")


if __name__ == "__main__":
    main()
