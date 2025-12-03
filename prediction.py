import joblib

def predict(data):
    # Memuat model yang sudah disimpan sebelumnya
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)