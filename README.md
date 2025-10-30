# 🛰️ 5G Flow Forensics — NLP-Based Attack Detection

A Natural Language Processing (NLP)-inspired digital forensics prototype for automated classification of 5G network traffic flows.  
This project demonstrates how structured network flow data can be analyzed using TF-IDF feature extraction and Logistic Regression to identify anomalous or malicious activity.

---

## 🧠 Overview

The system converts raw 5G network flows into textual **“key=value”** form, vectorizes them using **TF–IDF**, and classifies each record as either **Rapid Reset**, **SMA Subscribe Notify**, or **Benign**.  

This approach allows interpretable forensic analysis of network activity without requiring deep packet inspection, making it suitable for academic demonstrations and proof-of-concept investigations.

---

## 🧩 Architecture Pipeline

- **Input:** raw 5G network flows in `key=value` format  
- **Processing:** TF-IDF vectorization of flow attributes  
- **Classification:** Logistic Regression model predicts attack type  
- **Output:** predicted label, probability, top contributing tokens, optional timeline plots  

---

## ⚙️ Features

- Interactive **web-based prototype** built with Flask and HTML/CSS  
- Accepts **manual flow input** or **uploaded CSV data** (optional)  
- Performs **real-time classification** using a pre-trained Logistic Regression model  
- Displays predicted label, confidence score, and **token-level feature contributions**  
- Clean, minimal UI suitable for presentations and teaching  

---

## 🧰 Tech Stack

- **Python 3.x**
- **Flask** – Web interface
- **scikit-learn** – Machine learning model training
- **pandas**, **numpy** – Data processing
- **matplotlib** – Timeline plotting
- **HTML5 / CSS3** – Frontend interface  

---

## 🚀 How to Run the Prototype

### 1. Install required Python packages
Open a terminal and run:

```bash
pip install flask joblib pandas scikit-learn matplotlib
```


###  2. Start the Flask app

Make sure you are in the project directory (where app.py is located), then run:
```bash
python app.py
```
### 3. Open the web interface

Once Flask starts, open a browser and navigate to:
```bash
http://127.0.0.1:5000/
```
### 4. Use the prototype

- Paste a raw 5G flow in the text box OR fill in the form fields.

- Click Run Prediction to see:

   - The predicted label
  
   - Probability of the prediction
  
   - Top contributing tokens



