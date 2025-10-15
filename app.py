# # app.py
# import os
# import joblib
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, send_file, redirect, url_for
# from werkzeug.utils import secure_filename
# import matplotlib.pyplot as plt
# from io import BytesIO
# from datetime import datetime
# from sklearn.linear_model import LogisticRegression

# # ---------- CONFIG ----------
# MODEL_FILE = "5g_attack_model.pkl"
# VECTORIZER_FILE = "5g_vectorizer.pkl"
# TEST_CSV = "5g_attack_test_data.csv"   # optional test csv (20% you saved)
# UPLOAD_FOLDER = "uploads"
# REPORTS_FOLDER = "reports"
# PLOTS_FOLDER = "static/plots"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(REPORTS_FOLDER, exist_ok=True)
# os.makedirs(PLOTS_FOLDER, exist_ok=True)

# # ---------- LOAD MODEL ----------
# model = joblib.load(MODEL_FILE)
# vectorizer = joblib.load(VECTORIZER_FILE)

# # For multiclass: get classes and feature names
# classes = list(model.classes_)
# try:
#     feature_names = np.array(vectorizer.get_feature_names_out())
# except:
#     feature_names = np.array(vectorizer.get_feature_names())

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # ---------- UTIL: Explain prediction by token contribution ----------
# def explain_prediction(flow_text, top_k=8):
#     """
#     Returns predicted label, probability, and top contributing tokens for that label.
#     Works for linear models (LogisticRegression) using coef * tfidf_value.
#     """
#     x_vec = vectorizer.transform([flow_text])
#     probs = model.predict_proba(x_vec)[0]
#     pred_idx = int(model.predict(x_vec)[0] == model.classes_[0]) if False else None
#     pred_label = model.predict(x_vec)[0]
#     pred_prob = probs[list(model.classes_).index(pred_label)]

#     # if model has coef_
#     if hasattr(model, "coef_"):
#         # multiclass: pick coef row corresponding to predicted label
#         class_index = list(model.classes_).index(pred_label)
#         coef = model.coef_[class_index]  # shape (n_features,)
#         # multiply coef * tfidf value to get contribution
#         dense = x_vec.toarray()[0]
#         contrib = coef * dense
#         # pick top features by positive contribution
#         top_idx = np.argsort(contrib)[-top_k:][::-1]
#         top_tokens = []
#         for i in top_idx:
#             if dense[i] > 0:
#                 top_tokens.append({
#                     "token": feature_names[i],
#                     "tfidf": float(dense[i]),
#                     "coef": float(coef[i]),
#                     "score": float(contrib[i])
#                 })
#         return pred_label, float(pred_prob), top_tokens
#     else:
#         # fallback
#         return pred_label, float(pred_prob), []

# # ---------- UTIL: generate timeline plot for predicted label ----------
# def generate_timeline_plot(df_path, label):
#     """
#     Generates a per-minute timeline PNG for the given label using df_path.
#     Returns relative path to saved PNG (in static/plots).
#     """
#     if not os.path.exists(df_path):
#         return None
#     df = pd.read_csv(df_path, parse_dates=['Timestamp'], dayfirst=True, infer_datetime_format=True)
#     # filter label and resample
#     df = df.dropna(subset=['Timestamp'])
#     # create binary column for label
#     df['is_label'] = (df['Label'] == label).astype(int)
#     # set index and resample per minute (or per hour if too many points)
#     df = df.set_index('Timestamp')
#     # try per-minute; if too sparse or too long span, use hourly
#     try:
#         timeline = df['is_label'].resample('1Min').sum()
#         if len(timeline) > 2000:
#             timeline = df['is_label'].resample('1H').sum()
#             xlabel = 'Time (hour)'
#         else:
#             xlabel = 'Time (minute)'
#     except Exception:
#         timeline = df['is_label'].resample('1H').sum()
#         xlabel = 'Time (hour)'

#     if timeline.sum() == 0:
#         return None

#     plt.figure(figsize=(10,4))
#     timeline.plot()
#     plt.title(f"Occurrences of '{label}' over time")
#     plt.ylabel("Count")
#     plt.xlabel(xlabel)
#     plt.tight_layout()
#     fname = f"{label.replace(' ','_')}_timeline.png"
#     outpath = os.path.join(PLOTS_FOLDER, secure_filename(fname))
#     plt.savefig(outpath)
#     plt.close()
#     return outpath

# # ---------- ROUTES ----------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     explanation = None
#     timeline_img = None
#     uploaded = None

#     if request.method == "POST":
#         # check if CSV uploaded to use for timeline generation
#         if 'testcsv' in request.files and request.files['testcsv'].filename != '':
#             f = request.files['testcsv']
#             filename = secure_filename(f.filename)
#             path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             f.save(path)
#             uploaded = filename
#         # read form fields
#         srcip = request.form.get("srcip", "").strip()
#         dstip = request.form.get("dstip", "").strip()
#         protocol = request.form.get("protocol", "").strip()
#         fwdpkts = request.form.get("fwdpkts", "0").strip()
#         bwdpkts = request.form.get("bwdpkts", "0").strip()
#         duration = request.form.get("duration", "0").strip()
#         fwdbytelen = request.form.get("fwdbytelen", "0").strip()
#         bwdbytelen = request.form.get("bwdbytelen", "0").strip()

#         flow_text = (f"srcip={srcip} dstip={dstip} protocol={protocol} "
#                      f"fwdpkts={fwdpkts} bwdpkts={bwdpkts} duration={duration} "
#                      f"fwdbytelen={fwdbytelen} bwdbytelen={bwdbytelen}")

#         # prediction and explanation
#         pred_label, pred_prob, top_tokens = explain_prediction(flow_text, top_k=8)

#         result = {
#             "label": pred_label,
#             "prob": round(pred_prob, 4),
#             "flow_text": flow_text
#         }
#         explanation = top_tokens

#         # timeline: prefer uploaded CSV else default TEST_CSV if exists
#         timeline_source = path if uploaded else (TEST_CSV if os.path.exists(TEST_CSV) else None)
#         if timeline_source:
#             timeline_img = generate_timeline_plot(timeline_source, pred_label)
#             if timeline_img:
#                 # convert to relative path for template
#                 timeline_img = timeline_img.replace('\\','/')
#                 if timeline_img.startswith('static/'):
#                     timeline_img = '/' + timeline_img
#                 else:
#                     timeline_img = '/' + timeline_img

#         # build a downloadable report HTML
#         timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#         report_name = f"forensic_report_{timestamp}.html"
#         report_path = os.path.join(REPORTS_FOLDER, report_name)
#         # render report template and save file
#         with open(report_path, "w", encoding="utf-8") as fh:
#             report_html = render_template(
#                 "report_template.html",
#                 result=result,
#                 explanation=explanation,
#                 timeline_img=(timeline_img if timeline_img else None),
#                 classes=classes
#             )
#             fh.write(report_html)

#         return render_template("index.html",
#                                result=result,
#                                explanation=explanation,
#                                timeline_img=timeline_img,
#                                report_file=report_name)

#     return render_template("index.html")

# @app.route("/download_report/<report_name>")
# def download_report(report_name):
#     path = os.path.join(REPORTS_FOLDER, secure_filename(report_name))
#     if not os.path.exists(path):
#         return "Report not found", 404
#     return send_file(path, as_attachment=True, download_name=report_name)

# # ---------- RUN ----------
# if __name__ == "__main__":
#     app.run(debug=True)
# # To run: pip install flask joblib pandas scikit-learn matplotlib
# # Then: python app.py   


# app.py
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- CONFIG ----------
MODEL_FILE = "5g_attack_model.pkl"
VECTORIZER_FILE = "5g_vectorizer.pkl"
TEST_CSV = "5g_attack_test_data.csv"   # optional test csv (20% you saved)
UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"
PLOTS_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ---------- LOAD MODEL ----------
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# For multiclass: get classes and feature names
classes = list(model.classes_)
try:
    feature_names = np.array(vectorizer.get_feature_names_out())
except:
    feature_names = np.array(vectorizer.get_feature_names())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =========================
# Helper: predict_flow_multi
# =========================
def predict_flow_multi_from_text(flow_text):
    """
    Accepts flow_text like:
    'srcip=90.247.10.100 dstip=111.6.10.100 protocol=0 fwdpkts=2 bwdpkts=0 duration=30 fwdbytelen=0 bwdbytelen=0'
    Returns: (pred_label, pred_prob)
    """
    X = vectorizer.transform([flow_text])
    pred = model.predict(X)[0]
    try:
        prob = float(np.max(model.predict_proba(X)))
    except Exception:
        prob = None
    return pred, prob

# ---------- UTIL: Explain prediction by token contribution ----------
def explain_prediction(flow_text, top_k=8):
    """
    Returns predicted label, probability, and top contributing tokens for that label.
    Works for linear models (LogisticRegression) using coef * tfidf_value.
    """
    x_vec = vectorizer.transform([flow_text])
    probs = None
    try:
        probs = model.predict_proba(x_vec)[0]
    except Exception:
        probs = None

    pred_label = model.predict(x_vec)[0]
    pred_prob = probs[list(model.classes_).index(pred_label)] if probs is not None else None

    # if model has coef_
    if hasattr(model, "coef_"):
        # multiclass: pick coef row corresponding to predicted label
        class_index = list(model.classes_).index(pred_label)
        coef = model.coef_[class_index]  # shape (n_features,)
        # multiply coef * tfidf value to get contribution
        dense = x_vec.toarray()[0]
        contrib = coef * dense
        # pick top features by positive contribution
        top_idx = np.argsort(contrib)[-top_k:][::-1]
        top_tokens = []
        for i in top_idx:
            if dense[i] > 0:
                top_tokens.append({
                    "token": feature_names[i],
                    "tfidf": float(dense[i]),
                    "coef": float(coef[i]),
                    "score": float(contrib[i])
                })
        return pred_label, float(pred_prob) if pred_prob is not None else None, top_tokens
    else:
        # fallback
        return pred_label, float(pred_prob) if pred_prob is not None else None, []

# ---------- UTIL: generate timeline plot for predicted label ----------
def generate_timeline_plot(df_path, label):
    """
    Generates a per-minute timeline PNG for the given label using df_path.
    Returns relative path to saved PNG (in static/plots).
    """
    if not os.path.exists(df_path):
        return None
    df = pd.read_csv(df_path, parse_dates=['Timestamp'], dayfirst=True, infer_datetime_format=True)
    # filter label and resample
    df = df.dropna(subset=['Timestamp'])
    # create binary column for label
    df['is_label'] = (df['Label'] == label).astype(int)
    df = df.set_index('Timestamp')

    try:
        timeline = df['is_label'].resample('1Min').sum()
        xlabel = 'Time (minute)'
        if len(timeline) > 2000:
            timeline = df['is_label'].resample('1H').sum()
            xlabel = 'Time (hour)'
    except Exception:
        timeline = df['is_label'].resample('1H').sum()
        xlabel = 'Time (hour)'

    if timeline.sum() == 0:
        return None

    plt.figure(figsize=(10,4))
    timeline.plot()
    plt.title(f"Occurrences of '{label}' over time")
    plt.ylabel("Count")
    plt.xlabel(xlabel)
    plt.tight_layout()
    fname = f"{label.replace(' ','_')}_timeline.png"
    outpath = os.path.join(PLOTS_FOLDER, secure_filename(fname))
    plt.savefig(outpath)
    plt.close()
    # return path that the template can access
    return '/' + outpath.replace('\\','/')

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    explanation = None
    timeline_img = None
    report_name = None

    if request.method == "POST":
        # handle uploaded CSV for timeline
        uploaded_path = None
        if 'testcsv' in request.files and request.files['testcsv'].filename != '':
            f = request.files['testcsv']
            filename = secure_filename(f.filename)
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(uploaded_path)

        # If rawflow provided, parse it; else use form fields
        rawflow = request.form.get('rawflow', '').strip()
        flow_text = None
        provided_true = None

        if rawflow:
            # If user pasted "flow_text,truetype" split on last comma
            if ',' in rawflow:
                # split at last comma to allow commas inside values (unlikely here)
                parts = rawflow.rsplit(',', 1)
                maybe_flow = parts[0].strip()
                maybe_label = parts[1].strip()
                flow_text = maybe_flow
                provided_true = maybe_label if maybe_label != '' else None
            else:
                flow_text = rawflow
        else:
            # collect individual fields (fall back to empty strings if missing)
            srcip = request.form.get("srcip", "").strip()
            dstip = request.form.get("dstip", "").strip()
            protocol = request.form.get("protocol", "").strip()
            fwdpkts = request.form.get("fwdpkts", "0").strip()
            bwdpkts = request.form.get("bwdpkts", "0").strip()
            duration = request.form.get("duration", "0").strip()
            fwdbytelen = request.form.get("fwdbytelen", "0").strip()
            bwdbytelen = request.form.get("bwdbytelen", "0").strip()

            flow_text = (f"srcip={srcip} dstip={dstip} protocol={protocol} "
                         f"fwdpkts={fwdpkts} bwdpkts={bwdpkts} duration={duration} "
                         f"fwdbytelen={fwdbytelen} bwdbytelen={bwdbytelen}")

        # Ensure flow_text is non-empty
        if not flow_text or flow_text.strip() == '':
            return render_template("index.html", error="No flow provided")

        # Use explain_prediction for full output (includes probability and top tokens)
        pred_label, pred_prob, top_tokens = explain_prediction(flow_text, top_k=8)
        # In case explain didn't provide prob (should), fallback to predict_flow_multi_from_text
        if pred_prob is None:
            pred_label, pred_prob = predict_flow_multi_from_text(flow_text)

        result = {
            "label": pred_label,
            "prob": pred_prob if pred_prob is not None else 0.0,
            "flow_text": flow_text
        }
        if provided_true:
            result["true_label"] = provided_true

        explanation = top_tokens

        # timeline source prefers uploaded csv, else TEST_CSV
        timeline_source = uploaded_path if uploaded_path else (TEST_CSV if os.path.exists(TEST_CSV) else None)
        if timeline_source:
            timeline_img = generate_timeline_plot(timeline_source, pred_label)

        # build report
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_name = f"forensic_report_{timestamp}.html"
        report_path = os.path.join(REPORTS_FOLDER, report_name)
        with open(report_path, "w", encoding="utf-8") as fh:
            report_html = render_template("report_template.html",
                                          result=result,
                                          explanation=explanation,
                                          timeline_img=timeline_img,
                                          classes=classes)
            fh.write(report_html)

        return render_template("index.html",
                               result=result,
                               explanation=explanation,
                               timeline_img=timeline_img,
                               report_file=report_name)

    return render_template("index.html")

@app.route("/download_report/<report_name>")
def download_report(report_name):
    path = os.path.join(REPORTS_FOLDER, secure_filename(report_name))
    if not os.path.exists(path):
        return "Report not found", 404
    return send_file(path, as_attachment=True, download_name=report_name)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
