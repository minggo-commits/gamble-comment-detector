# Deteksi Komentar Judi Online

🚀 Proyek Machine Learning untuk mendeteksi komentar yang mengandung **judi online**.  
Menggunakan pipeline otomatis (GitHub Actions) dan aplikasi demo berbasis Gradio.

## Alur Proyek
1. Dataset komentar (`data/komentar.csv`)
2. Preprocessing (cleaning teks, stopwords)
3. TF-IDF + Logistic Regression
4. Model disimpan ke `models/`
5. Web demo dengan Gradio
6. Pipeline otomatis dengan GitHub Actions

## Cara Menjalankan
```bash
pip install -r requirements.txt
python src/train.py
python app.py
