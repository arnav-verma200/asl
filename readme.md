# ASL Sign Language Translator

Real-time ASL hand gesture → text + speech using CV + ML.

## Accuracy
| Model | Accuracy |
|---|---|
| SVM | 99.15% |
| Random Forest | 98.36% |
| MLP | 96.52% |
| KNN | 95.74% |

## Stack
`MediaPipe` `OpenCV` `Scikit-learn` `Flask` `pyttsx3`

## Setup
```bash
git clone https://github.com/yourusername/asl-translator
cd asl-translator
pip install -r requirements.txt
```

## Dataset
[ASL Alphabet — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) → place in `data/raw/`

## Run
```bash
python src/feature_extraction.py
python src/data_preprocessing.py
python src/train.py
python app/app.py
```
Open `http://localhost:5000`

## Features
- 26 letters (A-Z) including motion gestures J and Z
- Word autocomplete
- Text to speech
- Laptop + phone camera support
