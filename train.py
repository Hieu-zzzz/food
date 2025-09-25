import argparse
import json
import os
import re
import sys
import pickle
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


def load_data(json_path: str) -> Tuple[List[str], List[float]]:
    """
    Đọc dữ liệu từ JSON và trả về danh sách tên sản phẩm và giá (float).
    JSON kỳ vọng dạng { "Tên": { "...": "...", "Giá": "23.000đ" }, ... }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    product_names: List[str] = []
    prices: List[float] = []

    for key, value in data.items():
        # Lấy tên sản phẩm: ưu tiên key, fallback trường văn bản đầu tiên
        name = str(key)
        if isinstance(value, dict):
            # Tìm trường giá
            price_text = value.get("Giá")
            # Nếu name có thể rõ ràng hơn trong value (ví dụ lặp lại), giữ nguyên key làm tên
        else:
            price_text = None

        if price_text is None:
            # Bỏ qua mục không có giá
            continue

        price_value = parse_price_to_float(price_text)
        if price_value is None:
            continue

        product_names.append(name)
        prices.append(price_value)

    if len(product_names) == 0:
        raise ValueError("Không có mẫu hợp lệ (có trường Giá) trong JSON.")

    return product_names, prices


def parse_price_to_float(price_text: str) -> float:
    """
    Chuyển chuỗi giá dạng "23.000đ" -> 23000.0
    Hỗ trợ các biến thể có dấu chấm, dấu phẩy, ký tự tiền tệ.
    """
    if price_text is None:
        return None
    # Loại bỏ ký tự không phải số, dấu chấm, dấu phẩy
    cleaned = re.sub(r"[^0-9.,]", "", str(price_text))
    if cleaned == "":
        return None
    # Nếu có cả dấu chấm và phẩy, giả định chấm là phân tách nghìn, phẩy là thập phân
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "")
        cleaned = cleaned.replace(",", ".")
    else:
        # Nếu chỉ có dấu chấm: giả định là phân tách nghìn → bỏ chấm
        if "." in cleaned and cleaned.count(".") >= 1 and cleaned.split(".")[-1].__len__() == 3:
            cleaned = cleaned.replace(".", "")
        # Nếu chỉ có dấu phẩy: thay phẩy bằng chấm (thập phân)
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def build_char_index(texts: List[str]) -> Dict[str, int]:
    """
    Tạo từ điển ký tự -> index, padding dùng 0. Index bắt đầu từ 1.
    """
    charset = set()
    for t in texts:
        if t is None:
            continue
        charset.update(list(str(t).lower()))
    # Sắp xếp để cố định thứ tự
    sorted_chars = sorted(list(charset))
    char_to_index: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(sorted_chars)}
    return char_to_index


def texts_to_padded_sequences(texts: List[str], char_to_index: Dict[str, int], max_len: int) -> np.ndarray:
    sequences = []
    for t in texts:
        t = str(t).lower()
        seq = [char_to_index.get(ch, 0) for ch in t]
        if len(seq) >= max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences, dtype=np.int32)


def build_cnn_model(vocab_size: int, embedding_dim: int, max_len: int) -> tf.keras.Model:
    """
    Mô hình Conv1D hồi quy giá. Đầu vào là chuỗi ký tự đã mã hoá.
    """
    inputs = layers.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(inputs)
    x = layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_and_evaluate_cnn(
    texts: List[str],
    prices: List[float],
    max_len: int = None,
    embedding_dim: int = 48,
    batch_size: int = 8,
    epochs: int = 300,
    val_ratio: float = 0.25,
) -> Tuple[tf.keras.Model, Dict[str, int], Dict[str, float], float]:
    """
    Huấn luyện mô hình CNN trên dữ liệu ký tự. Trả về (model, char_index, config, val_mae).
    """
    # Xây tokenizer
    char_to_index = build_char_index(texts)
    # Độ dài tối đa
    true_max_len = max(len(str(t)) for t in texts)
    if max_len is None:
        max_len = min(40, max(12, true_max_len))
    X = texts_to_padded_sequences(texts, char_to_index, max_len)

    y = np.array(prices, dtype=np.float32)
    # Chuẩn hoá giá để dễ học
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + 1e-6)
    y_norm = (y - y_mean) / y_std

    # Tách train/val
    n = len(X)
    split = max(1, int((1 - val_ratio) * n)) if n > 3 else n
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_norm[:split], y_norm[split:]

    model = build_cnn_model(vocab_size=len(char_to_index), embedding_dim=embedding_dim, max_len=max_len)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]

    if len(X_val) == 0:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
        )
        val_mae = float(history.history.get("mae", [0.0])[-1]) * y_std
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb,
            verbose=0,
        )
        val_mae_norm = min(history.history.get("val_mae", [0.0])) if "val_mae" in history.history else 0.0
        val_mae = float(val_mae_norm) * y_std

    config = {
        "max_len": int(max_len),
        "embedding_dim": int(embedding_dim),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, char_to_index, config, val_mae


def save_artifacts_tf(model: tf.keras.Model, char_index: Dict[str, int], config: Dict[str, float], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "price_cnn.h5")
    model.save(model_path)
    with open(os.path.join(out_dir, "char_index.json"), "w", encoding="utf-8") as f:
        json.dump(char_index, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình CNN dự đoán giá từ tên sản phẩm (VN)")
    parser.add_argument("--data", type=str, default="person_info.json", help="Đường dẫn file JSON đầu vào")
    parser.add_argument("--out", type=str, default="artifacts", help="Thư mục lưu mô hình")
    parser.add_argument("--max_len", type=int, default=None, help="Độ dài tối đa chuỗi ký tự (mặc định tự chọn)")
    parser.add_argument("--epochs", type=int, default=300, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=8, help="Kích thước batch")
    args = parser.parse_args(argv)

    print(f"Đọc dữ liệu từ: {args.data}")
    names, prices = load_data(args.data)
    print(f"Số mẫu hợp lệ: {len(names)}")

    model, char_index, config, val_mae = train_and_evaluate_cnn(
        texts=names,
        prices=prices,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Huấn luyện xong. MAE (ước lượng): {val_mae:.2f} (đơn vị VNĐ)")

    save_artifacts_tf(model, char_index, config, args.out)
    print(f"Đã lưu mô hình vào thư mục: {args.out}")
    print("Cách dùng dự đoán:")
    print("\nPython:")
    print("""
import json
import numpy as np
import tensorflow as tf

def load_predictor():
    with open('artifacts/char_index.json', 'r', encoding='utf-8') as f:
        char_index = json.load(f)
    with open('artifacts/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    model = tf.keras.models.load_model('artifacts/price_cnn.h5')
    return model, char_index, config

def texts_to_padded_sequences(texts, char_to_index, max_len):
    seqs = []
    for t in texts:
        t = str(t).lower()
        ids = [char_to_index.get(ch, 0) for ch in t]
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        seqs.append(ids)
    return np.array(seqs, dtype=np.int32)

model, char_index, config = load_predictor()
X = texts_to_padded_sequences(["Bánh Donut"], char_index, config['max_len'])
y_norm = model.predict(X, verbose=0).reshape(-1)
y = y_norm * config['y_std'] + config['y_mean']
print(y)
""".strip())

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


