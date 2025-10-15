# 🛰️ 5G Flow Forensics — NLP-Based Attack Detection

A Natural Language Processing (NLP)-inspired digital forensics prototype for automated classification of 5G network traffic flows.  
This project demonstrates how structured network flow data can be analyzed using TF-IDF feature extraction and Logistic Regression to identify anomalous or malicious activity.

---

## 🧠 Overview

The system converts raw 5G network flows into textual “key=value” form, vectorizes them using **TF–IDF**, and classifies each record as either **Rapid Reset**, **SMA Subscribe Notify**, or **Benign**.  
The approach allows interpretable forensic analysis of network activity without needing deep packet inspection.

---

## ⚙️ Features

- Interactive **web-based prototype** built with Flask and HTML/CSS  
- Accepts manual flow input or uploaded CSV data  
- Performs **real-time classification** using a pre-trained Logistic Regression model  
- Displays predicted label, confidence score, and token-level feature contributions  
- Clean and minimal UI suitable for demonstrations and academic presentations  

---

## 🧩 Architecture Pipeline

5G Network Dataset → Data Preprocessing → TF-IDF Vectorization → Logistic Regression Classifier → Web-based Prototype Output

yaml
Copy code

---

## 🧰 Tech Stack

- **Python 3.x**
- **Flask** (for web interface)
- **scikit-learn** (for model training)
- **pandas**, **numpy**
- **HTML5 / CSS3** (for the frontend)

---

