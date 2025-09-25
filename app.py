import os
from typing import Dict

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import json


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(APP_ROOT, 'artifacts_cnn')
UPLOAD_DIR = os.path.join(APP_ROOT, 'static', 'uploads')
MODEL_PATH = os.path.join(ART_DIR, 'cnn_best.keras')
CLASSES_TXT = os.path.join(ART_DIR, 'cnn_classes.txt')
PRICE_JSON = os.path.join(APP_ROOT, 'person_info.json')

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)


def load_classes(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split('\t')
            mapping[int(idx)] = name
    return mapping


def load_prices(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prices: Dict[str, str] = {}
        for name, info in data.items():
            if isinstance(info, dict) and ('Giá' in info or 'Price' in info):
                prices[name] = info.get('Giá', info.get('Price', 'N/A'))
        return prices
    except Exception:
        return {}


model = load_model(MODEL_PATH)
index_to_name = load_classes(CLASSES_TXT)
prices = load_prices(PRICE_JSON)
IMG_W = 224
IMG_H = 224
CONF_THRESH = 0.75


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    img = load_img(save_path, target_size=(IMG_H, IMG_W))
    arr = img_to_array(img).astype('float32')/255.0
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    cls = int(np.argmax(probs))
    name = index_to_name.get(cls, str(cls))
    price = prices.get(name, 'N/A')
    conf = float(np.max(probs))

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [
        {
            'index': int(i),
            'name': index_to_name.get(int(i), str(int(i))),
            'prob': float(probs[i])
        }
        for i in top5_idx
    ]

    verdict = 'Không chắc chắn' if conf < CONF_THRESH else name
    return render_template('index.html',
                           filename=filename,
                           result={'index': cls, 'name': verdict, 'price': price, 'conf': conf},
                           top5=top5,
                           conf_thresh=CONF_THRESH)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


