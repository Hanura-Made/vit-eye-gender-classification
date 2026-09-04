#  Gender Classification from Eye Images (ViT)

Demo publik: klasifikasi jenis kelamin berbasis citra mata manusia menggunakan
**Vision Transformer (ViT-Base/16)** — model fine-tuned dari skripsi
*"Klasifikasi Jenis Kelamin Berbasis Citra Mata Manusia dengan Pendekatan Vision Transformer (ViT)"*.

## Live Demo

**https://vit-eye-gender-classification-mpumsh4jmabrhrsadawrjp.streamlit.app/**

## Fitur

- Auto eye detection + crop (Haar cascade)
- Prediksi Female/Male + confidence + bar chart interaktif (Plotly)
- Model di-load langsung dari Hugging Face Hub

## Struktur

```
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
└── README.md
```

## Model

- Arsitektur: ViT-Base-patch16-224 (85.8M params)
- Checkpoint: D0 (grayscale preprocessing)
- Akurasi: val 97.2% / test 95.7%
- Berat model publik di: https://huggingface.co/Hanura22/vit-eye-gender-classification

> Model untuk tujuan penelitian. Bukan untuk produksi tanpa validasi lebih lanjut.
