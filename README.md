# 🎭 Emotion Classification using Deep Learning

## 📖 Overview
This project focuses on building a deep learning model that classifies **human emotions from text**. Using LSTM-based neural networks, the model predicts emotions such as *joy, anger, sadness, fear, surprise,* and *love*. It leverages NLP techniques for text cleaning, tokenization, and embedding generation to achieve robust emotion recognition.

---

## 🧠 Model & Approach
- **Architecture:** LSTM (Long Short-Term Memory)
- **Frameworks:** TensorFlow | Keras
- **Techniques Used:**
  - Text preprocessing (stopword removal, tokenization, padding)
  - Word embedding using **GloVe** or **Word2Vec**
  - Deep neural network model for emotion detection
  - Evaluation using **accuracy, precision, recall, F1-score**

---

## 🗂️ Dataset
- **Dataset Name:** Emotion Dataset  
- **Source:** [Kaggle Emotion Dataset](https://www.kaggle.com/datasets)  
- **Files Included:**
  - `training.csv`
  - `validation.csv`
  - `test.csv`

Each CSV contains text samples and their corresponding emotion labels.

---

## 🚀 How to Run

 1️⃣ Clone the Repository

git clone https://github.com/vinod8930/Emotion-Detection-of-text-using-Deep-Learning-models.git


2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed, then run:

pip install -r requirements.txt

3️⃣ Run the Notebook

Open and run the Jupyter Notebook file:

jupyter notebook "Emotion Classification Major Project (1).ipynb"

📦 Requirements

1.Create a requirements.txt file with the following dependencies:
txt
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
nltk
spacy
wordcloud
regex
tqdm


2.Install all dependencies:

pip install -r requirements.txt

📊 Results
Metric	Score
Accuracy	~85–90%
Precision	High
Recall	High
F1-Score	Strong overall model performance

The model efficiently captures context and sentiment in text sequences, performing well on unseen validation data.

📈 Future Enhancements

1.Integrate Transformer-based models (BERT, DistilBERT)

2.Implement real-time emotion detection using a web app (Flask/Streamlit)

3.Add multi-lingual emotion classification support

📁 Project Structure
Emotion-Classification/
│
├── Emotion Classification Major Project (1).ipynb   # Main Notebook
├── training.csv
├── validation.csv
├── test.csv
├── requirements.txt
└── README.md

👨‍💻 Author

Vinod Penkey
Deep Learning | NLP | Emotion Classification
VIT-AP University

📧 vinodpenkey@email.com

🏷️ License

This project is released under the MIT License — free to use and modify for educational and research purposes.
