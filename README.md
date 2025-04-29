# 📧 Spam Email Classifier

A machine learning project to classify SMS/email messages as **Spam** or **Not Spam** using **Naive Bayes** and **TF-IDF vectorization**. Built in Python with Scikit-learn, it works **fully offline** without needing NLTK downloads.

---

## 🚀 Features

- Text preprocessing and cleaning (lowercasing, punctuation removal, custom stopwords)
- Feature extraction using TF-IDF
- Trained with Naive Bayes classifier
- 95%+ accuracy on SMS Spam Collection dataset
- Simple interface to test your own messages
- No internet required — works fully offline!

---

## 📂 Dataset

- Dataset used: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- File name: `SMSSpamCollection`  
- Format: Tab-separated values (`label` and `message`)

---

## 🛠️ Installation

1. **Clone this repository**:

   git clone https://github.com/Rohit-712/spam-email-classifier.git
   cd spam-email-classifier

 
 Install required libraries:
pip install -r requirements.txt
Download and place dataset:

Download SMSSpamCollection

Place it in the project folder (same level as the script)

📊 Example Usage
python:
from spam_classifier import predict_spam
print(predict_spam("Congratulations! You've won a free ticket!"))
# Output: Spam

🧑‍💻 Author
Rohit Pawar
GitHub: @Rohit-712
