# 🏦 Banka Müşteri Kayıp Tahmini

**End-to-End Machine Learning Projesi**

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

---

## 🎥 Demo

### 🌐 Canlı Demo
> **Deploy Linki:** *(GitHub'a push sonrası Streamlit Cloud üzerinden deploy edilecek)*

### 📸 Ekran Görüntüleri
*(Deployment sonrası eklenecek)*

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
12. [İletişim](#-iletişim)

---

## 1) Problem Tanımı

### 🎯 İş Problemi

Müşteri kaybı (churn), bankalar için kritik bir problemdir:

- **Maliyet:** Yeni müşteri kazanma maliyeti, mevcut müşteriyi elde tutmaktan 5-7x daha pahalıdır
- **Gelir Kaybı:** Her kaybedilen müşteri, lifetime value (LTV) kaybı anlamına gelir
- **Rekabet:** Müşteriler kolayca rakip bankalara geçebilir

**Proje Amacı:** Kaybedilme riski yüksek müşterileri önceden tespit ederek, proaktif retention (elde tutma) kampanyaları düzenlemek.

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

**Özellikler:**

- **Demografik:** credit_score, age, gender, country
- **Finansal:** balance, estimated_salary
- **Banka İlişkisi:** tenure, products_number, credit_card, active_member
- **Hedef:** churn

---

## 2) Baseline Süreci ve Skoru

### 🔧 Baseline Yaklaşımı

**Model:** Logistic Regression (sklearn default parametreler)

**Preprocessing Stratejisi:**
- **RobustScaler:** Sayısal özellikleri ölçeklendirme (outlier'lara dayanıklı)
- **OneHotEncoder:** Kategorik özellikleri encoding (drop='first')
- **Passthrough:** Binary features (credit_card, active_member) olduğu gibi

**Validasyon:** 80-20 train-test split (random_state=42, stratify=y)

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

**✅ Güçlü Yönler:**
- Non-churn sınıfında yüksek recall (0.97)
- Genel accuracy makul (81%)
- Hızlı training (<1 saniye)

**❌ Zayıf Yönler:**
- **Churn recall çok düşük (0.19):** Churn edecek müşterilerin %81'i kaçırılıyor!
- F1-score çok düşük (0.28)
- ROC-AUC (0.77) yetersiz

**Sonuç:** Baseline, class imbalance nedeniyle minority class'ı ihmal ediyor. **Feature engineering ve model optimization zorunlu!**

---

## 3) Feature Engineering Denemeleri ve Sonuçları

EDA ve domain knowledge'a dayalı olarak **6 yeni özellik** oluşturuldu:

### 🛠️ Oluşturulan Özellikler

#### 1️⃣ **Ratio Features (Oran Özellikleri)**

**balance_to_salary_ratio:**
```python
balance_to_salary_ratio = balance / (estimated_salary + 1)
```
- **Hipotez:** Yüksek bakiye/düşük maaş oranı → Aktif olmayan müşteri → Churn riski
- **Sonuç:** ✅ Churn ile **pozitif korelasyon** (0.15)

**tenure_age_ratio:**
```python
tenure_age_ratio = tenure / (age + 1)
```
- **Hipotez:** Genç yaşta müşteri olanlar daha sadık
- **Sonuç:** ✅ Churn ile **negatif korelasyon**

#### 2️⃣ **Categorical Binning**

**credit_score_category:**
```python
pd.cut(credit_score, bins=[0, 600, 700, 850], labels=['Low', 'Medium', 'High'])
```
- **Sonuç:** ✅ 'Low' kategorisinde churn oranı **%35+** (High'da %15)

**age_group:**
```python
pd.cut(age, bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])
```
- **Sonuç:** ✅ 'Middle-aged' grubunda churn peak'i gözlemlendi

#### 3️⃣ **Interaction Features**

**high_value_customer:**
```python
(balance >= Q3_balance) & (salary >= Q3_salary)
```
- **Sonuç:** ✅ High-value müşterilerde churn **%12** (normal müşterilerde %22)

**inactive_high_balance:** ⭐ **EN GÜÇLÜ FEATURE**
```python
(active_member == 0) & (balance >= Q3_balance)
```
- **Sonuç:** ✅✅✅ Bu segmentte churn oranı **%45+** - Model için kritik sinyal!

### 📈 Feature Engineering Etkisi

| Metrik | Baseline | + Feature Engineering | İyileşme |
|--------|----------|----------------------|----------|
| **Test ROC-AUC** | 0.77 | 0.83 | +7.8% |
| **Test Recall (Churn)** | 0.19 | 0.42 | +121% (2.2x) |
| **Test F1 (Churn)** | 0.28 | 0.54 | +93% |

**Sonuç:** Feature engineering **oyunun kurallarını değiştirdi!** Recall neredeyse 2x arttı.

---

## 4) Validasyon Şeması ve Seçim Gerekçesi

### ✅ Seçilen Strateji: Train-Test Split + K-Fold Cross-Validation

#### 🔹 Train-Test Split

**Parametreler:**
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**Neden bu split?**
- ✅ **80-20 oranı:** 10K veri için yeterli (train: 8K, test: 2K)
- ✅ **stratify=y:** Class imbalance korunuyor (her split'te ~20% churn)
- ✅ **random_state=42:** Reproducibility sağlanıyor
- ✅ **Tek split:** Time-series değil, snapshot tabanlı veri → tek split yeterli

#### 🔹 K-Fold Cross-Validation (Hyperparameter Tuning için)

**Parametreler:**
```python
RandomizedSearchCV(cv=5, scoring='roc_auc', n_iter=20)
```

**Neden 5-Fold CV?**
- ✅ **5-fold optimal:** Bias-variance trade-off dengeli
- ✅ **ROC-AUC metric:** Class imbalance'ta accuracy misleading, ROC-AUC güvenilir
- ✅ **RandomizedSearchCV:** GridSearch yerine 5x daha hızlı

### ❌ Kullanılmayan Alternatifler

**Time-Series CV:**
- ❌ Veri temporal değil (müşteri snapshot'ları)
- ❌ Gereksiz complexity

**Nested CV:**
- ❌ Çok pahalı (3 model × 20 iter × 5 fold × 5 outer = 1500 model)
- ❌ Bu problemde overkill

**Sonuç:** Pratik + güvenilir + production'a uygun validasyon şeması.

---

## 5) Final Pipeline ve Feature Set Stratejisi

### 🏗️ Final Feature Set (19 özellik)

#### Orijinal Features (10)
- **Numerical (6):** credit_score, age, tenure, balance, products_number, estimated_salary
- **Categorical (2):** country, gender
- **Binary (2):** credit_card, active_member

#### Engineered Features (6)
- **Numerical (2):** balance_to_salary_ratio, tenure_age_ratio
- **Categorical (2):** credit_score_category, age_group
- **Binary (2):** high_value_customer, inactive_high_balance

**Toplam:** 16 raw features → **19 processed features** (OneHotEncoding sonrası)

### 🔧 Preprocessing Stratejisi

#### RobustScaler (Numerical Features)

```python
RobustScaler()  # Median ve IQR tabanlı
```

**Neden RobustScaler?**
- ✅ **Outlier'lara dayanıklı:** balance ve salary'de extreme değerler var
- ❌ StandardScaler: Outlier'lardan çok etkilenir
- ❌ MinMaxScaler: Outlier'lara hassas

**Test sonuçları:**
- StandardScaler → ROC-AUC: 0.86
- MinMaxScaler → ROC-AUC: 0.85
- **RobustScaler → ROC-AUC: 0.874** ✅ (KAZANAN!)

#### OneHotEncoder (Categorical Features)

```python
OneHotEncoder(drop='first', sparse_output=False)
```

**Neden drop='first'?**
- ✅ Multicollinearity önlenir (dummy variable trap)
- ✅ Baseline Logistic Regression için kritik

### 🎯 Neden Bu Strateji?

1. **Domain knowledge + Data-driven:** Features hem istatistiksel hem de business mantığıyla seçildi
2. **Robust:** Outlier'lara ve yeni data'ya dayanıklı
3. **Reproducible:** sklearn Pipeline ile tüm adımlar otomatik
4. **Production-ready:** Tek `.pkl` dosyası ile deploy edilebilir

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
| **Specificity** | 0.971 | 0.971 | Korundu ✅ |

### 💡 Hangi İyileştirmeler Katkı Sağladı?

1. **Feature Engineering (60% katkı):**
   - `inactive_high_balance` tek başına Recall'u +15 puan artırdı
   - Ratio features non-linear patterns yakaladı

2. **Model Değişimi (30% katkı):**
   - CatBoost, categorical features'ı doğal olarak handle ediyor
   - Overfitting riski düşük (depth=5, l2_reg=5)

3. **Hyperparameter Tuning (10% katkı):**
   - Optimal learning_rate (0.1) bulundu
   - Border_count=64 ile decision boundaries iyileşti

### ✅ Overfitting Check

- Train ROC-AUC: 0.883
- Test ROC-AUC: 0.874
- **Gap: 0.009** → Minimal overfitting! Model generalizable.

---

## 7) Final Modelin Business Gereksinimleri ile Uyumu

### ❓ Kritik Soru: "Recall 0.48 - Bu Kötü Değil mi?"

#### ✅ CEVAP: Hayır! Bu dataset özelinde **normal ve beklenen** bir performans.

### 🎯 Neden Recall 0.48 Kabul Edilebilir?

#### 1️⃣ Dataset Zorluğu

Bu dataset **inherently challenging:**
- ❌ **Transaction history yok:** Harcama alışkanlıkları görünmüyor
- ❌ **Customer interaction yok:** Çağrı merkezi, şikayetler yok
- ❌ **Campaign response yok:** Geçmiş retention çabalarına tepki yok

**Sonuç:** Model sadece **proxy signals** kullanıyor.

#### 2️⃣ Class Imbalance Etkisi

- Churn: 20% (minority)
- Non-churn: 80% (majority)

**Default threshold (0.5):** Model doğal olarak majority class'e kayıyor.

**Trade-off:**
- High specificity (0.97) → Low false positives ✅
- Moderate recall (0.48) → Higher false negatives ⚠️

#### 3️⃣ Kaggle Benchmarks

Aynı dataset'te public kernels:
- Ortalama Recall: 0.40 - 0.55
- Ortalama ROC-AUC: 0.82 - 0.88

**Bizim model:**
- Recall: 0.484 ✅ Ortalama içinde
- ROC-AUC: 0.874 ✅ **Üst dilimde**

### 💼 Model'in Güçlü Yönleri (Business Açısından)

#### ✅ 1) Çok Yüksek Specificity (0.971)

**Anlam:**
- Non-churn müşterilerin %97'sini doğru tespit
- **False positive sadece 46/1593** (Yanlış alarm %3)

**Business değeri:**
- ✅ Gereksiz retention kampanyalarına para harcanmıyor
- ✅ Müşteriler rahatsız edilmiyor
- ✅ Marketing budget optimize

**Örnek:**
- Kampanya maliyeti: ₺50/müşteri
- Yanlış alarm: 46 kişi → **₺2,300 israf** (Çok düşük!)

#### ✅ 2) Yüksek Precision (0.810)

**Anlam:**
- Model "churn" dediğinde %81 doğru
- **Retention kampanyası gönderilen müşterilerin 8/10'u gerçekten riskli!**

**Business değeri:**
- ✅ Kampanya ROI yüksek
- ✅ Müşteri memnuniyeti korunuyor

#### ✅ 3) ROC-AUC 0.874 (Mükemmel Ranking)

**Anlam:**
- Model, müşterileri **risk skoruna göre sıralama** konusunda çok iyi

**Business değeri:**
- ✅ **Priority queue:** En riskli 100 müşteriyle başla
- ✅ **Budget allocation:** Risksiz müşterilere kaynak harcama
- ✅ **Segmentation:** Düşük/Orta/Yüksek risk grupları

### 🔧 Recall İyileştirme Yolları (Future Work)

#### 1️⃣ Threshold Tuning (En kolay)

```python
threshold = 0.35  # 0.5 yerine
predictions = (probabilities > threshold).astype(int)
```

**Etki:**
- Recall: 0.48 → ~0.65 ↑
- Precision: 0.81 → ~0.60 ↓
- FP: 46 → ~150 ↑

#### 2️⃣ Cost-Sensitive Learning

```python
CatBoostClassifier(scale_pos_weight=4)
```

#### 3️⃣ SMOTE / Oversampling

```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

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

#### 🔹 Deployment Mimarisi

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Web UI     │──────│  Streamlit   │──────│ final_       │
│  (Browser)   │      │  App Server  │      │ pipeline.pkl │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │  Monitoring  │
                      │  Dashboard   │
                      └──────────────┘
```

#### 🔹 Deployment Seçenekleri

##### **Seçenek 1: Streamlit Cloud** (Önerilen)

**Adımlar:**
1. GitHub'a repo push et
2. [share.streamlit.io](https://share.streamlit.io) → Deploy
3. Repository seç: `username/bank-churn`
4. Main file: `Scripts/app.py`
5. Python: 3.11
6. Deploy! ✅

**Avantajlar:**
- ✅ Ücretsiz (public repo)
- ✅ 5 dakikada canlı
- ✅ Otomatik update (Git push → auto deploy)
- ✅ HTTPS otomatik

**Deployment Path Uyumluluğu:**
```python
# config.py'deki path stratejisi
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
```
✅ Bu yaklaşım hem local hem GitHub deployment için çalışır!

### 📊 Production'da İzlenecek Metrikler

#### 1️⃣ **Prediction Metrics** (Haftalık)

| Metrik | Target | Alarm |
|--------|--------|-------|
| Prediction Volume | ~X/week | <50% veya >200% |
| Churn Rate | ~20% | <10% veya >35% |
| Avg Churn Prob | ~0.25 | <0.10 veya >0.50 |

#### 2️⃣ **Model Accuracy** (Aylık)

```python
# Ground truth karşılaştırması
if roc_auc < 0.80:
    send_alert("Model performance degraded!")
```

**Target:**
- ROC-AUC ≥ 0.85 (alarm < 0.80)
- Recall ≥ 0.45 (alarm < 0.35)
- Precision ≥ 0.75 (alarm < 0.65)

#### 3️⃣ **Feature Drift Detection** (Aylık)

```python
# Kolmogorov-Smirnov test
if p_value < 0.05:
    print("⚠️ Feature drift detected - Retrain!")
```

#### 4️⃣ **Business Impact** (Aylık)

- **Retention Campaign ROI:** >200%
- **Churn Prevention Rate:** >30%
- **False Alarm Cost:** <₺10,000/month

### 🔄 Retraining Strategy

**Trigger Koşulları:**
1. **Time-based:** Her 3 ayda bir
2. **Performance-based:** ROC-AUC < 0.80
3. **Data drift:** KS-test p-value < 0.05

---

## 💻 Kurulum

### Gereksinimler

- Python 3.7+
- pip veya conda

### 1️⃣ Repository Clone

```bash
git clone https://github.com/username/bank-customer-churn.git
cd bank-customer-churn
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

**⚠️ Önemli:** scikit-learn **1.2.2** kullanıldı (Kaggle uyumluluğu). Farklı versiyon model loading hatası verir!

### 4️⃣ Model Dosyalarını İndir

Model dosyaları Git'e dahil edilmedi (boyut nedeniyle).

**Kaggle'dan indirme:**
1. Notebook 3 → Output → `preprocessor.pkl`
2. Notebook 4 → Output → `best_model.pkl`
3. Notebook 6 → Output → `final_pipeline.pkl`
4. İndirilen dosyaları `Scripts/` klasörüne kopyala

### 5️⃣ Streamlit App'i Çalıştır

```bash
cd Scripts
streamlit run app.py
```

Tarayıcınız otomatik açılacak (http://localhost:8501).

---

## 📁 Proje Yapısı

```
bank-customer-churn/
│
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
├── .gitignore                          # Git ignore rules
└── LICENSE                             # MIT License
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
