
import pandas as pd
import numpy as np
import re, requests, time, warnings
from io import BytesIO
from PIL import Image
import pytesseract
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

def ocr_from_url(url):
    try:
        img = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
        img = img.convert("L").resize((img.width*2, img.height*2))
        return pytesseract.image_to_string(img).strip()
    except:
        return ""

def engineer_text_features(text_series):
    text = text_series.fillna("").str.lower()
    df = pd.DataFrame(index=text.index)
    df["text_length"] = text.str.len()
    df["word_count"] = text.str.split().apply(len)
    df["has_number"] = text.str.contains(r"\d").astype(int)
    def extract(p): return pd.to_numeric(text.str.extract(p, flags=re.IGNORECASE)[0], errors="coerce").fillna(0)
    df["ounces"] = extract(r"(\d+(?:\.\d+)?)\s*(oz|ounce|ounces)")
    df["grams"] = extract(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)")
    df["ml"] = extract(r"(\d+(?:\.\d+)?)\s*(ml|milliliters?)")
    df["pack"] = extract(r"(?:pack of|pk|x)\s*(\d+)")
    for k in ["organic","premium","genuine","bundle","new","original"]:
        df[f"kw_{k}"] = text.str.contains(k).astype(int)
    return df

print("📥 Loading data...")
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("🧠 Generating OCR text...")
train["ocr_text"] = [ocr_from_url(u) for u in tqdm(train["image_link"])]
test["ocr_text"] = [ocr_from_url(u) for u in tqdm(test["image_link"])]

cat_train, cat_test = train["catalog_content"].astype(str), test["catalog_content"].astype(str)
ocr_train, ocr_test = train["ocr_text"].astype(str), test["ocr_text"].astype(str)

eng_train = engineer_text_features(cat_train).values
eng_test = engineer_text_features(cat_test).values

print("🤗 Loading MiniLM model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🔎 Encoding catalog text...")
cat_train_emb = embed_model.encode(cat_train.tolist(), batch_size=64, show_progress_bar=True)
cat_test_emb = embed_model.encode(cat_test.tolist(), batch_size=64, show_progress_bar=True)

print("🔎 Encoding OCR text...")
ocr_train_emb = embed_model.encode(ocr_train.tolist(), batch_size=64, show_progress_bar=True)
ocr_test_emb = embed_model.encode(ocr_test.tolist(), batch_size=64, show_progress_bar=True)

X_train = np.hstack([eng_train, cat_train_emb, ocr_train_emb])
X_test = np.hstack([eng_test, cat_test_emb, ocr_test_emb])
y = train["price"].values

print("🚀 Training model...")
model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.1, random_state=42)
model.fit(X_tr, y_tr)
val = model.predict(X_val)
smape = np.mean(2*np.abs(val-y_val)/(np.abs(val)+np.abs(y_val)+1e-8))*100
print(f"✅ Validation SMAPE: {smape:.2f}%")

print("📦 Training final model on full data...")
model.fit(X_train, y)

print("🔮 Predicting on test...")
pred = model.predict(X_test)
out = pd.DataFrame({"sample_id": test["sample_id"], "price": pred})
out.to_csv("outputs/final_predictions.csv", index=False)

print("🎉 Done! File saved: outputs/final_predictions.csv")
