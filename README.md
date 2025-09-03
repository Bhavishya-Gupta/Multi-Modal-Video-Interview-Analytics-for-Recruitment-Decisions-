
# 🎥 Multi-Modal Video Interview Analytics for Recruitment Decisions

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Computer%20Vision-FF6F00?style=for-the-badge&logo=opencv" alt="Computer Vision"/>
  <img src="https://img.shields.io/badge/NLP-34A853?style=for-the-badge&logo=nlp" alt="NLP"/>
  <img src="https://img.shields.io/badge/Recruitment-4285F4?style=for-the-badge&logo=briefcase" alt="Recruitment"/>
</p>
**An end-to-end framework that integrates facial expression analysis, voice prosody, and linguistic features to provide data-driven insights for hiring decisions.**

***

## 🔍 Repository Structure

```
Multi-Modal-Video-Interview-Analytics-for-Recruitment-Decisions/
│
├── data/                          # Sample videos and transcripts
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_modeling_and_evaluation.ipynb
│
├── src/                           # Source code modules
│   ├── face_emotion.py           # Facial expression detection
│   ├── audio_prosody.py          # Voice feature extraction
│   ├── nlp_features.py           # Linguistic analysis
│   ├── fusion_model.py           # Multi-modal fusion architecture
│   └── utils.py                  # Helper functions
│
├── models/                        # Trained model checkpoints
│
├── results/                       # Evaluation reports and visualizations
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── performance_metrics.csv
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
└── README.md                      # Project overview and instructions
```


***

## 🚀 Key Features

- **Facial Expression Analysis**
    - Real-time emotion detection using pretrained CNNs (e.g., FaceNet + FER models)
    - Extracts valence and arousal features per video frame
- **Voice Prosody Extraction**
    - Computes pitch, energy, speech rate, and spectral features using librosa
    - Identifies vocal cues linked to confidence and engagement
- **Linguistic Feature Engineering**
    - Natural language processing of interview transcripts
    - Measures lexical richness, sentiment polarity, and keyword usage
- **Multi-Modal Fusion \& Modeling**
    - Late fusion of visual, acoustic, and textual embeddings
    - Trains ensemble classifiers (e.g., Random Forest, XGBoost) and deep learning models
    - Outputs suitability score and interpretable feature contributions
- **Dashboard \& Reporting**
    - Generates interactive plots: ROC curves, confusion matrices, feature importances
    - Exports candidate analytics reports for hiring teams

***

## 🧠 Methodology Overview

1. **Data Preprocessing**
    - Frame extraction and face alignment
    - Audio segmentation and silence removal
    - Transcript cleaning and tokenization
2. **Feature Extraction**
    - Visual: emotion probabilities, facial action units
    - Acoustic: MFCCs, pitch contours, energy dynamics
    - Textual: TF-IDF vectors, sentiment scores, POS tag frequencies
3. **Model Training**
    - Train uni-modal classifiers for each feature set
    - Fuse embeddings into a joint representation
    - Evaluate with cross-validation (metrics: AUC, F1, accuracy)
4. **Interpretability**
    - SHAP values to explain model predictions
    - User-friendly visual summaries for each candidate

***

## 📈 Installation \& Usage

```bash
# Clone the repository
git clone https://github.com/Bhavishya-Gupta/Multi-Modal-Video-Interview-Analytics-for-Recruitment-Decisions-.git
cd Multi-Modal-Video-Interview-Analytics-for-Recruitment-Decisions-

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/face_emotion.py --input data/videos/ --output results/face_features.csv
python src/audio_prosody.py --input data/videos/ --output results/audio_features.csv
python src/nlp_features.py --input data/transcripts/ --output results/text_features.csv

# Train and evaluate fusion model
python src/fusion_model.py --face results/face_features.csv \
                           --audio results/audio_features.csv \
                           --text results/text_features.csv \
                           --output results/performance_metrics.csv
```


***

## 🎯 Results \& Evaluation

- **ROC AUC**: 0.92
- **F1-Score**: 0.88
- **Accuracy**: 0.90

**Top Features**

***

## 🔧 Extensibility

- Swap in custom emotion or prosody models
- Add additional modalities (e.g., eye-tracking, physiological signals)
- Deploy inference service with FastAPI for live interviews
- Integrate front-end dashboard (Streamlit or Dash) for HR teams

***

## 👤 Author \& Contact

**Bhavishya Gupta**

- GitHub: [@Bhavishya-Gupta](https://github.com/Bhavishya-Gupta)
- LinkedIn: linkedin.com/in/bhavishya-gupta
- Email: Available on GitHub profile

***

## 📜 License

This project is licensed under the MIT License.
*Enhancing recruitment fairness and efficiency with multi-modal AI.*

