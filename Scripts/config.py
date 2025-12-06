"""
Banka Müşteri Kaybı Tahmin Projesi - Konfigürasyon Dosyası

Bu dosya tüm konfigürasyon ayarlarını içerir:
- Kaggle path'leri
- Model ayarları
- Özellik mühendisliği parametreleri
- Kolon tanımlamaları
"""

import os

# ==================== PATH'LER ====================

# Kaggle input path (veri seti)
KAGGLE_INPUT_PATH = '/kaggle/input/bank-customer-churn-dataset/'
DATASET_FILENAME = 'Bank Customer Churn Prediction.csv'
DATASET_PATH = os.path.join(KAGGLE_INPUT_PATH, DATASET_FILENAME)

# Kaggle working directory (çıktı)
KAGGLE_WORKING_PATH = '/kaggle/working/'

# Model path'leri - Local için mevcut dizini kullan
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # Scripts klasörü
FINAL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'final_pipeline.pkl')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')

# Sonuç path'leri
RESULTS_PATH = os.path.join(MODEL_DIR, 'results.json')
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_params.json')
EVALUATION_REPORT_PATH = os.path.join(MODEL_DIR, 'evaluation_report.md')

# Özellik mühendisliği çıktıları
X_TRAIN_FE_PATH = os.path.join(MODEL_DIR, 'X_train_fe.npy')
X_TEST_FE_PATH = os.path.join(MODEL_DIR, 'X_test_fe.npy')
Y_TRAIN_PATH = os.path.join(MODEL_DIR, 'y_train.npy')
Y_TEST_PATH = os.path.join(MODEL_DIR, 'y_test.npy')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.txt')


# ==================== KOLON TANIMLARI ====================

# Orijinal özellikler
ORIGINAL_NUMERICAL_COLS = [
    'credit_score',
    'age',
    'tenure',
    'balance',
    'products_number',
    'estimated_salary'
]

ORIGINAL_CATEGORICAL_COLS = [
    'country',  # France, Germany, Spain
    'gender'    # Male, Female
]

ORIGINAL_BINARY_COLS = [
    'credit_card',      # 0/1
    'active_member'     # 0/1
]

# Yeni özellikler (özellik mühendisliği sırasında oluşturulan)
NEW_NUMERICAL_COLS = [
    'balance_to_salary_ratio',
    'tenure_age_ratio'
]

NEW_CATEGORICAL_COLS = [
    'credit_score_category',  # Low, Medium, High
    'age_group'               # Young, Middle-aged, Senior
]

NEW_BINARY_COLS = [
    'high_value_customer',
    'inactive_high_balance'
]

# Birleşik özellik listeleri
ALL_NUMERICAL_COLS = ORIGINAL_NUMERICAL_COLS + NEW_NUMERICAL_COLS
ALL_CATEGORICAL_COLS = ORIGINAL_CATEGORICAL_COLS + NEW_CATEGORICAL_COLS
ALL_BINARY_COLS = ORIGINAL_BINARY_COLS + NEW_BINARY_COLS

ALL_FEATURES = ALL_NUMERICAL_COLS + ALL_CATEGORICAL_COLS + ALL_BINARY_COLS

# Hedef değişken
TARGET_COL = 'churn'


# ==================== ÖZELLİK MÜHENDİSLİĞİ PARAMETRELERİ ====================

# Kredi skoru aralıkları ve etiketleri
CREDIT_SCORE_BINS = [0, 600, 700, 850]
CREDIT_SCORE_LABELS = ['Low', 'Medium', 'High']

# Yaş aralıkları ve etiketleri
AGE_BINS = [0, 30, 50, 100]
AGE_LABELS = ['Young', 'Middle-aged', 'Senior']

# Yüksek değerli müşteri eşikleri (quantile'lar)
BALANCE_QUANTILE = 0.75
SALARY_QUANTILE = 0.75


# ==================== MODEL AYARLARI ====================

# Train-test ayrımı
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5

# RandomizedSearchCV
N_ITER = 20
SCORING_METRIC = 'roc_auc'

# Model hiperparametre grid'leri
XGBOOST_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

LIGHTGBM_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 50, 70, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_samples': [10, 20, 30]
}

CATBOOST_PARAM_GRID = {
    'iterations': [100, 200, 300, 500],
    'depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 0.5, 1]
}


# ==================== ÖN İŞLEME AYARLARI ====================

# Sayısal özellikler için scaler tipi
SCALER_TYPE = 'robust'  # Seçenekler: 'standard', 'robust', 'minmax'

# OneHotEncoder ayarları
OHE_DROP = 'first'
OHE_HANDLE_UNKNOWN = 'ignore'


# ==================== STREAMLIT APP AYARLARI ====================

# App başlığı ve açıklama
APP_TITLE = "🏦 Banka Müşteri Kayıp Tahmini"
APP_DESCRIPTION = """
Bu uygulama, bir banka müşterisinin kaybedilme olasılığını tahmin eder.
Tahmin almak için yan çubuktan müşteri bilgilerini girin.
"""

# Girdi alanı konfigürasyonları
INPUT_FIELDS = {
    'credit_score': {
        'label': 'Kredi Skoru',
        'min': 300,
        'max': 850,
        'value': 650,
        'help': 'Müşteri kredi skoru (300-850)'
    },
    'age': {
        'label': 'Yaş',
        'min': 18,
        'max': 100,
        'value': 35,
        'help': 'Müşteri yaşı (yıl)'
    },
    'tenure': {
        'label': 'Müşteri Olma Süresi (yıl)',
        'min': 0,
        'max': 20,
        'value': 5,
        'help': 'Müşteri olarak geçirilen yıl sayısı'
    },
    'balance': {
        'label': 'Hesap Bakiyesi',
        'min': 0.0,
        'max': 300000.0,
        'value': 100000.0,
        'step': 1000.0,
        'help': 'Mevcut hesap bakiyesi'
    },
    'products_number': {
        'label': 'Ürün Sayısı',
        'min': 1,
        'max': 4,
        'value': 2,
        'help': 'Sahip olunan banka ürünü sayısı'
    },
    'estimated_salary': {
        'label': 'Tahmini Maaş',
        'min': 0.0,
        'max': 300000.0,
        'value': 75000.0,
        'step': 1000.0,
        'help': 'Tahmini yıllık maaş'
    },
    'country': {
        'label': 'Ülke',
        'options': ['France', 'Germany', 'Spain'],
        'help': 'Müşteri ülkesi'
    },
    'gender': {
        'label': 'Cinsiyet',
        'options': ['Male', 'Female'],
        'help': 'Müşteri cinsiyeti'
    },
    'credit_card': {
        'label': 'Kredi Kartı Var mı',
        'value': True,
        'help': 'Müşterinin kredi kartı var mı?'
    },
    'active_member': {
        'label': 'Aktif Üye',
        'value': True,
        'help': 'Müşteri aktif üye mi?'
    }
}

# Tahmin gösterim ayarları
CHURN_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7
LOW_RISK_THRESHOLD = 0.3


# ==================== YARDİMCI FONKSİYONLAR ====================

def get_feature_info():
    """
    Özellik tipi bilgisi içeren dictionary döndür
    """
    return {
        'numerical': ALL_NUMERICAL_COLS,
        'categorical': ALL_CATEGORICAL_COLS,
        'binary': ALL_BINARY_COLS,
        'all': ALL_FEATURES,
        'target': TARGET_COL
    }


def get_column_mapping():
    """
    Ön işleme için kolon tipi eşleştirmesi döndür
    """
    return {
        'original_numerical': ORIGINAL_NUMERICAL_COLS,
        'original_categorical': ORIGINAL_CATEGORICAL_COLS,
        'original_binary': ORIGINAL_BINARY_COLS,
        'new_numerical': NEW_NUMERICAL_COLS,
        'new_categorical': NEW_CATEGORICAL_COLS,
        'new_binary': NEW_BINARY_COLS
    }


def validate_input(input_dict):
    """
    Girdi verisi dictionary'sini doğrula
    """
    required_fields = ORIGINAL_NUMERICAL_COLS + ORIGINAL_CATEGORICAL_COLS + ORIGINAL_BINARY_COLS
    
    for field in required_fields:
        if field not in input_dict:
            raise ValueError(f"Eksik gerekli alan: {field}")
    
    return True


# ==================== GÖRÜNTÜLEME ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Banka Müşteri Kaybı Tahmini - Konfigürasyon")
    print("=" * 60)
    print(f"\nVeri Seti Path: {DATASET_PATH}")
    print(f"Model Çıktı Path: {MODEL_DIR}")
    print(f"\nOrijinal Özellikler:")
    print(f"  - Sayısal: {len(ORIGINAL_NUMERICAL_COLS)}")
    print(f"  - Kategorik: {len(ORIGINAL_CATEGORICAL_COLS)}")
    print(f"  - Binary: {len(ORIGINAL_BINARY_COLS)}")
    print(f"\nYeni Özellikler:")
    print(f"  - Sayısal: {len(NEW_NUMERICAL_COLS)}")
    print(f"  - Kategorik: {len(NEW_CATEGORICAL_COLS)}")
    print(f"  - Binary: {len(NEW_BINARY_COLS)}")
    print(f"\nToplam Özellik: {len(ALL_FEATURES)}")
    print(f"Hedef: {TARGET_COL}")
    print("=" * 60)
