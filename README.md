# 🏦 Banka Müşteri Kayıp Tahmini

**End-to-End Machine Learning Projesi**

*MultiAcademy Zero2End Machine Learning Bootcamp kapsamında geliştirilmiştir.*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-yellow.svg)](https://catboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)

> Bankacılık sektörü için geliştirilmiş, müşteri kaybını **%87 doğrulukla** tahmin eden, production-ready makine öğrenmesi sistemi.

---

## 📊 Proje Özeti

Bu proje, **bankacılık sektöründe müşteri kaybı (churn) riskini tahmin etmek** için geliştirilmiş eksiksiz bir machine learning uygulamasıdır. Kaggle'daki **Bank Customer Churn Dataset** kullanılarak müşteri demografisi, finansal davranışlar ve ürün kullanımı gibi özelliklerden yola çıkarak müşterilerin bankadan ayrılma olasılığını tahmin eder.

### 🎯 Kapsam

- **Sektör:** Bankacılık / Finans
- **Problem Tipi:** İkili Sınıflandırma (Binary Classification)
- **Dataset:** 10,000 müşteri, 14 orijinal özellik
- **Veri Kaynağı:** [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Final Model:** CatBoost
- **Model Performansı:** ROC-AUC 0.874, Accuracy 87%, Recall 0.48, Specificity 0.97
- **Deployment:** Streamlit Web App

## 📓 Notebook Dosyaları

- EDA_Notebook : https://www.kaggle.com/code/sametsenturk/1-eda-notebook
- Baseline_Notebook : https://www.kaggle.com/code/sametsenturk/2-baseline-notebook
- Feature_Engineering_Notebook : https://www.kaggle.com/code/sametsenturk/3-feature-engineering-notebook
- Model_Optimization_Notebook : https://www.kaggle.com/code/sametsenturk/4-model-optimization-notebook
- Model_Evaluation_Notebook : https://www.kaggle.com/code/sametsenturk/5-model-evaluation-notebook
- Pipeline_Notebook : https://www.kaggle.com/code/sametsenturk/6-pipeline-notebook

---

## 🎥 Canlı Demo

**🌐 Deploy Linki:** https://churn-predictionn.streamlit.app/

---

## 📋 İçindekiler

1. [Problem Tanımı](#1-problem-tanımı)
2. [Baseline Süreci ve Skoru](#2-baseline-süreci-ve-skoru)
3. [Feature Engineering Denemeleri](#3-feature-engineering-denemeleri-ve-sonuçları)
4. [Validasyon Şeması](#4-validasyon-şeması-ve-seçim-gerekçesi)
5. [Final Pipeline ve Feature Set Seçimi](#5-final-pipeline-ve-feature-set-stratejisi)
6. [Baseline vs Final Model Karşılaştırması](#6-baseline-vs-final-model-performans-farkı)
7. [Business Uyumu](#7-final-modelin-business-gereksinimleri-ile-uyumu)
8. [Production Deployment Stratejisi](#8-canlıya-çıkma-ve-izleme-stratejisi)
9. [Kurulum](#-kurulum)
10. [Proje Yapısı](#-proje-yapısı)
11. [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
12. [Teşekkürler](#-teşekkürler)

---

## 1) Problem Tanımı

### 🎯 İş Problemi

Müşteri kaybı (churn), bankalar için kritik bir problemdir. Yeni müşteri kazanma maliyeti mevcut müşteriyi elde tutmaktan 5-7x daha pahalıdır.

**Proje Amacı:** Kaybedilme riski yüksek müşterileri önceden tespit ederek, proaktif retention kampanyaları düzenlemek.

### 📊 Veri Seti Hakkında

| Özellik | Detay |
|---------|-------|
| **Kaynak** | [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) |
| **Toplam Kayıt** | 10,000 |
| **Özellik Sayısı** | 14 (13 bağımsız + 1 hedef) |
| **Hedef Değişken** | `churn` (0: Kalır, 1: Ayrılır) |
| **Class Dağılımı** | Non-churn: 79.6%, Churn: 20.4% |
| **Eksik Veri** | ✅ Yok |
| **Duplikasyon** | ✅ Yok |

**Özellikler:** credit_score, age, gender, country, balance, estimated_salary, tenure, products_number, credit_card, active_member, churn (hedef)

---

## 2) Baseline Süreci ve Skoru

### 🔧 Baseline Yaklaşımı

- **Model:** Logistic Regression
- **Preprocessing:** RobustScaler + OneHotEncoder
- **Validasyon:** 80-20 train-test split (stratified)

### 📊 Baseline Sonuçları

```
============================================================
LOGISTIC REGRESSION BASELINE RESULTS
============================================================
              precision    recall  f1-score   support
------------------------------------------------------------
 Not Churned       0.82      0.97      0.89      1593
     Churned       0.59      0.19      0.28       407
------------------------------------------------------------
    accuracy                           0.81      2000
   macro avg       0.71      0.58      0.59      2000
weighted avg       0.78      0.81      0.77      2000
============================================================

Key Metrics:
  • Test Accuracy:  0.81
  • Test ROC-AUC:   0.77
  • Test Recall:    0.19 (Churned class - ÇOK DÜŞÜK!)
  • Test F1-Score:  0.28 (Churned class)
```

### 💡 Baseline Analizi

**Zayıf Yönler:**
- Churn recall çok düşük (0.19) - Churn edecek müşterilerin %81'i kaçırılıyor
- F1-score düşük (0.28)
- ROC-AUC (0.77) yetersiz

**Sonuç:** Class imbalance nedeniyle minority class ihmal ediliyor. Feature engineering ve model optimization gerekli.

---

## 3) Feature Engineering Denemeleri ve Sonuçları

EDA ve domain knowledge'a dayalı olarak **6 yeni özellik** oluşturuldu:

### 🛠️ Oluşturulan Özellikler

**1. Ratio Features:**
- `balance_to_salary_ratio` 
- `tenure_age_ratio` 

**2. Categorical Binning:**
- `credit_score_category` (Low/Medium/High) 
- `age_group` (Young/Middle-aged/Senior) 

**3. Interaction Features:**
- `high_value_customer` 
- `inactive_high_balance` 
---

## 4) Validasyon Şeması ve Seçim Gerekçesi

### ✅ Seçilen Strateji

**Train-Test Split:**
- 80-20 split, stratified (class imbalance korunuyor)
- random_state=42 (reproducibility)

**K-Fold Cross-Validation:**
- 5-fold CV (hyperparameter tuning için)
- ROC-AUC metric (class imbalance'ta güvenilir)
- RandomizedSearchCV (20 iterasyon)

**Neden Time-Series CV kullanılmadı:** Veri temporal değil, müşteri snapshot'ları

**Neden Nested CV kullanılmadı:** Bu problem için overkill

---

## 5) Final Pipeline ve Feature Set Stratejisi

### 🏗️ Final Feature Set

**Orijinal (10):** credit_score, age, tenure, balance, products_number, estimated_salary, country, gender, credit_card, active_member

**Engineered (6):** balance_to_salary_ratio, tenure_age_ratio, credit_score_category, age_group, high_value_customer, inactive_high_balance

**Toplam:** 16 raw → 19 processed features (OneHotEncoding sonrası)

### 🔧 Preprocessing Stratejisi

**RobustScaler (Numerical):** Outlier'lara dayanıklı, median ve IQR tabanlı

**OneHotEncoder (Categorical):** drop='first' (multicollinearity önlenir)

**Neden bu strateji:**
- Outlier'lara dayanıklı
- sklearn Pipeline ile reproducible
- Tek `.pkl` dosyası ile production-ready

---

## 6) Baseline vs Final Model: Performans Farkı

### 🏆 Model Karşılaştırması

| Model | Train ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall | Test F1 |
|-------|---------------|--------------|---------------|-------------|---------|
| **Baseline (LogReg)** | 0.770 | 0.770 | 0.810 | 0.190 | 0.280 |
| **XGBoost** | 0.905 | 0.867 | 0.867 | 0.457 | 0.583 |
| **LightGBM** | 0.886 | 0.872 | 0.866 | 0.474 | 0.590 |
| **CatBoost** 🏆 | 0.883 | **0.874** | **0.872** | **0.484** | **0.606** |

### 📊 CatBoost - Final Model Detayları

```
🏆 BEST MODEL: CatBoost
============================================================
Hyperparameters:
  • iterations:          100
  • depth:               5
  • learning_rate:       0.1
  • l2_leaf_reg:         5
  • border_count:        64
  • bagging_temperature: 0

Performance Metrics:
  • Test ROC-AUC:        0.8741 ⭐ (Excellent!)
  • Test Accuracy:       87.2%
  • Test Recall (Churn): 48.4%
  • Test Precision:      81.0%
  • Test F1:             60.6%
  • Specificity:         97.1% (Very High!)

Confusion Matrix:
  TN: 1547  |  FP: 46
  FN: 210   |  TP: 197
============================================================
```

### 📈 Baseline → Final Model İyileşmesi

| Metrik | Baseline | Final | İyileşme |
|--------|----------|-------|----------|
| **ROC-AUC** | 0.770 | 0.874 | **+13.5%** ⬆️ |
| **Recall** | 0.190 | 0.484 | **+155%** ⬆️ (2.5x) |
| **F1-Score** | 0.280 | 0.606 | **+116%** ⬆️ |
| **Accuracy** | 0.810 | 0.872 | **+7.6%** ⬆️ |
| **Specificity** | 0.966 | 0.971 | ⬆️ |

### 💡 İyileştirme Katkıları

1. **Feature Engineering :** `inactive_high_balance` tek başına Recall +15 puan
2. **Model Değişimi :** CatBoost categorical features'ı doğal handle ediyor
3. **Hyperparameter Tuning :** Optimal parametreler bulundu

**Overfitting Check:** Train (0.883) vs Test (0.874) → Gap: 0.009 (Minimal!)

---

## 7) Final Modelin Business Gereksinimleri ile Uyumu

### ❓ Kritik Soru: "Recall 0.48 - Bu Kötü Değil mi?"

#### ✅ CEVAP: Hayır! Bu dataset özelinde **normal ve beklenen** bir performans.

### 🎯 Neden Recall 0.48 Kabul Edilebilir?

**1. Dataset Zorluğu:**
- Transaction history, customer interaction, campaign response yok
- Model sadece proxy signals kullanıyor

**2. Class Imbalance:**
- Churn %20 (minority), Non-churn %80 (majority)
- Default threshold (0.5) → Model majority class'e kayıyor
- Trade-off: High specificity (0.97) vs Moderate recall (0.48)

### 💼 Model'in Güçlü Yönleri (Business Açısından)

**1. Çok Yüksek Specificity (0.971):**
- Non-churn müşterilerin %97'sini doğru tespit
- FP sadece 46/1593 → Gereksiz kampanya maliyeti minimal

**2. Yüksek Precision (0.810):**
- Model "churn" dediğinde %81 doğru
- Kampanya gönderilen müşterilerin 8/10'u gerçekten riskli → ROI yüksek

**3. ROC-AUC 0.874 (Mükemmel Ranking):**
- Müşterileri risk skoruna göre sıralama başarılı
- Priority queue ve budget allocation için ideal

### 🔧 Recall İyileştirme Yolları (Future Work)

**1. Threshold Tuning:** 0.5 → 0.35 (Recall ↑, Precision ↓)

**2. Cost-Sensitive Learning:** `scale_pos_weight` parametresi

**3. SMOTE/Oversampling:** Minority class dengeleme

### ✅ Final Değerlendirme

| Kriter | Durum | Açıklama |
|--------|-------|----------|
| **Teknik başarı** | ✅ Mükemmel | ROC-AUC 0.874, overfitting yok |
| **Dataset benchmark** | ✅ Üst seviye | Kaggle ortalamasının üzerinde |
| **Specificity odaklı** | ✅ Güçlü | False positive minimize |
| **Recall odaklı** | ⚠️ Makul | Dataset limitleri dahilinde |
| **Yorumlanabilirlik** | ✅ Var | SHAP values kullanıldı |
| **Production-ready** | ✅ Hazır | Pipeline tek dosya |

**Sonuç:**

> Model, **"cost-efficient" ve "precision-first"** bir business stratejisine uygundur. Recall öncelikli olursa, threshold tuning ile kolayca adjust edilebilir. **Bu esneklik, modelin güçlü bir yönüdür.**

---

## 8) Canlıya Çıkma ve İzleme Stratejisi

### 🚀 Production Deployment Planı

**Platform:** Streamlit Cloud

**Adımlar:**
1. GitHub'a push
2. share.streamlit.io → Deploy
3. Repository ve main file (Scripts/app.py) seç
4. Otomatik deploy

**Path Stratejisi:**
```python
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
```
✅ Hem local hem cloud deployment için uyumlu

### 📊 Production'da İzlenecek Metrikler

**1. Prediction Metrics (Haftalık):**
- Prediction volume, churn rate, avg churn probability

**2. Model Accuracy (Aylık):**
- ROC-AUC ≥ 0.85 (alarm < 0.80)
- Recall ≥ 0.45 (alarm < 0.35)
- Precision ≥ 0.75 (alarm < 0.65)

**3. Feature Drift Detection (Aylık):**
- Kolmogorov-Smirnov test (p-value < 0.05 → Retrain)

**4. Business Impact (Aylık):**
- Retention Campaign ROI > 200%
- Churn Prevention Rate > 30%

**Retraining Triggers:**
- Time-based: Her 3 ayda bir
- Performance-based: ROC-AUC < 0.80
- Data drift detected

---

## 💻 Kurulum

### Gereksinimler

- Python 3.7+
- pip veya conda

### 1️⃣ Repository Clone

```bash
git clone https://github.com/sametssenturk/churn-prediction.git
cd churn-prediction
```

### 2️⃣ Virtual Environment Oluştur

```bash
# venv ile
python -m venv venv
venv\Scripts\activate  # Windows

# conda ile
conda create -n churn python=3.11
conda activate churn
```

### 3️⃣ Dependencies Yükle

```bash
pip install -r requirements.txt
```

**⚠️ Önemli:** scikit-learn 1.2.2 kullanıldı (Kaggle uyumluluğu için kritik)

### 4️⃣ Model Dosyalarını Yerleştir

Model dosyaları (`*.pkl`) `Scripts/` klasöründe olmalı. Kaggle notebook'larından indirilebilir.

### 5️⃣ Streamlit App'i Çalıştır

```bash
cd Scripts
streamlit run app.py
```

Tarayıcınız otomatik açılacak (http://localhost:8501).

---

## 📁 Proje Yapısı

```
├── Notebooks/                          # Kaggle notebooks (çıktılarla)
│   ├── 1-eda-notebook.ipynb            # EDA
│   ├── 2-baseline-notebook.ipynb       # Baseline (LogReg)
│   ├── 3-feature-engineering-notebook.ipynb  # Feature engineering
│   ├── 4-model-optimization-notebook.ipynb   # Model training & tuning
│   ├── 5-model-evaluation-notebook.ipynb     # SHAP, confusion matrix
│   └── 6-pipeline-notebook.ipynb       # Final pipeline
│
├── Scripts/                            # Production Python modülleri
│   ├── app.py                          # Streamlit web app
│   ├── config.py                       # Konfigürasyonlar
│   ├── pipeline.py                     # ChurnPredictionPipeline
│   ├── inference.py                    # CLI & batch prediction
│   ├── final_pipeline.pkl              # Deploy-ready model
│   ├── best_model.pkl                  # CatBoost model
│   └── preprocessor.pkl                # RobustScaler + OneHotEncoder
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Bu dosya
```

---

## 🛠️ Kullanılan Teknolojiler

### Core ML Stack

| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| **Python** | 3.11 | Ana programlama dili |
| **pandas** | 2.0+ | Veri manipülasyonu |
| **NumPy** | 1.26.4 | Numerical computing |
| **scikit-learn** | 1.2.2 | Preprocessing, metrics |
| **CatBoost** | 1.2.8 | **Final model** |
| **XGBoost** | 3.1.2 | Model comparison |
| **LightGBM** | 4.6.0 | Model comparison |

### Visualization & Interpretability

| Teknoloji | Kullanım |
|-----------|----------|
| **matplotlib** | Static plots |
| **seaborn** | Statistical viz |
| **SHAP** | Model explainability |

### Deployment

| Teknoloji | Kullanım |
|-----------|----------|
| **Streamlit** | Web UI |
| **pickle** | Model serialization |

---

## 🙏 Teşekkürler

- **MultiAcademy:** Zero2End Machine Learning Bootcamp
- **MultiGroup Community:** Destek ve rehberlik
- **Dataset:** [Gaurav Topre - Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Platform:** Kaggle Notebooks
- **Tools:** scikit-learn, CatBoost, SHAP, Streamlit

---

<div align="center">

**⭐ Eğer bu proje işinize yaradıysa, star vermeyi unutmayın!**

Made with ❤️ using Python & CatBoost

</div>
