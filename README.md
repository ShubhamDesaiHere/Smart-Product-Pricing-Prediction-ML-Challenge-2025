# Price Prediction (MiniLM + OCR)

## 📂 Folder Structure
```
project/
 ┣ dataset/
 ┃ ┣ train.csv
 ┃ ┗ test.csv
 ┣ src/
 ┃ ┗ main.py
 ┣ outputs/
 ┃ ┗ final_predictions.csv (auto generated)
 ┣ requirements.txt
```
## 📥 Input file is like this


```
dataset/train.csv
dataset/test.csv
```

### train.csv is:
`sample_id, catalog_content, image_link, price`

### test.csv is:
`sample_id, catalog_content, image_link`

## ▶️ Run
```
pip install -r requirements.txt
python src/main.py
```

## 📤 Output Location
After run, prediction file is here:

```
outputs/final_predictions.csv
```