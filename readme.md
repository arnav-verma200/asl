# 🤟 ASL Sign Language Translator

Real-time ASL hand gesture to text and speech using Computer Vision + ML.

## Results
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
git clone https://github.com/yourusername/sign-language-translator
cd sign-language-translator
pip install -r requirements.txt
```

## Run Pipeline
```bash
python src/feature_extraction.py
python src/data_preprocessing.py
python src/train.py
```

## Run Web App
```bash
python app/app.py
```
Open `http://localhost:5000`

## Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) → place in `data/raw/`

## Author
Arnav Verma — [GitHub](https://github.com/arnav-verma200)
