"""
Banka Müşteri Kaybı Tahmini için Pipeline Modülü

Bu modül içerir:
- ChurnPredictionPipeline sınıfı (tam ön işleme + model pipeline)
- Özellik mühendisliği yardımcı fonksiyonları
- Model yükleme araçları
"""

import numpy as np
import pandas as pd
import pickle
from typing import Union, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Import configuration
from config import (
    ORIGINAL_NUMERICAL_COLS, ORIGINAL_CATEGORICAL_COLS, ORIGINAL_BINARY_COLS,
    NEW_NUMERICAL_COLS, NEW_CATEGORICAL_COLS, NEW_BINARY_COLS,
    ALL_NUMERICAL_COLS, ALL_CATEGORICAL_COLS, ALL_BINARY_COLS, ALL_FEATURES,
    CREDIT_SCORE_BINS, CREDIT_SCORE_LABELS, AGE_BINS, AGE_LABELS,
    BALANCE_QUANTILE, SALARY_QUANTILE,
    FINAL_PIPELINE_PATH
)


class ChurnPredictionPipeline:
    """
    Kaybı tahmini için tam pipeline: ham girdi → tahmin
    
    Pipeline adımları:
    1. Özellik Mühendisliği (yeni özellikler oluştur)
    2. Ön İşleme (scaling + encoding)
    3. Model Tahmini
    """
    
    def __init__(self, preprocessor, model):
        """
        Pipeline'ı preprocessor ve model ile başlat
        
        Args:
            preprocessor: ön işleme için sklearn ColumnTransformer
            model: eğitilmiş ML model (XGBoost/LightGBM/CatBoost)
        """
        self.preprocessor = preprocessor
        self.model = model
        
        # Özellik kolonları
        self.original_numerical = ORIGINAL_NUMERICAL_COLS
        self.original_categorical = ORIGINAL_CATEGORICAL_COLS
        self.original_binary = ORIGINAL_BINARY_COLS
        
        self.new_numerical = NEW_NUMERICAL_COLS
        self.new_categorical = NEW_CATEGORICAL_COLS
        self.new_binary = NEW_BINARY_COLS
        
        self.all_numerical = ALL_NUMERICAL_COLS
        self.all_categorical = ALL_CATEGORICAL_COLS
        self.all_binary = ALL_BINARY_COLS
        
        self.all_features = ALL_FEATURES
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yeni özellikler oluşturmak için özellik mühendisliği uygula
        
        Args:
            df: Orijinal özelliklerle DataFrame
            
        Returns:
            Orijinal + yeni özelliklerle DataFrame
        """
        df_fe = df.copy()
        
        # 1. Bakiye Maaş Oranı
        df_fe['balance_to_salary_ratio'] = df_fe['balance'] / (df_fe['estimated_salary'] + 1)
        
        # 2. Müşteri Olma Süresi Yaş Oranı
        df_fe['tenure_age_ratio'] = df_fe['tenure'] / (df_fe['age'] + 1)
        
        # 3. Kredi Skoru Kategorisi
        df_fe['credit_score_category'] = pd.cut(
            df_fe['credit_score'], 
            bins=CREDIT_SCORE_BINS, 
            labels=CREDIT_SCORE_LABELS
        )
        
        # 4. Yaş Grubu
        df_fe['age_group'] = pd.cut(
            df_fe['age'], 
            bins=AGE_BINS, 
            labels=AGE_LABELS
        )
        
        # 5. Yüksek Değerli Müşteri
        balance_threshold = df_fe['balance'].quantile(BALANCE_QUANTILE)
        salary_threshold = df_fe['estimated_salary'].quantile(SALARY_QUANTILE)
        df_fe['high_value_customer'] = (
            (df_fe['balance'] >= balance_threshold) & 
            (df_fe['estimated_salary'] >= salary_threshold)
        ).astype(int)
        
        # 6. Aktif Olmayan Yüksek Bakiye
        df_fe['inactive_high_balance'] = (
            (df_fe['active_member'] == 0) & 
            (df_fe['balance'] >= balance_threshold)
        ).astype(int)
        
        return df_fe
    
    def _prepare_input(self, X: Union[Dict, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Girdiyi DataFrame formatına dönüştür
        
        Args:
            X: Girdi verisi (dict, DataFrame, veya array)
            
        Returns:
            Uygun formatta DataFrame
        """
        if isinstance(X, dict):
            return pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.original_numerical + self.original_categorical + self.original_binary)
        elif isinstance(X, pd.DataFrame):
            return X
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")
    
    def predict(self, X: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ham girdiden tahmin yap
        
        Args:
            X: Ham girdi verisi
            
        Returns:
            Tahmin dizisi (0 veya 1)
        """
        # DataFrame'e dönüştür
        X_df = self._prepare_input(X)
        
        # Özellik mühendisliği uygula
        X_fe = self.apply_feature_engineering(X_df)
        
        # Özellikleri seç
        X_selected = X_fe[self.all_features]
        
        # Ön işle
        X_preprocessed = self.preprocessor.transform(X_selected)
        
        # Tahmin yap
        predictions = self.model.predict(X_preprocessed)
        
        return predictions
    
    def predict_proba(self, X: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ham girdiden tahmin olasılıkları al
        
        Args:
            X: Ham girdi verisi
            
        Returns:
            Olasılık dizisi [P(kaybedilmez), P(kaybedilir)]
        """
        # DataFrame'e dönüştür
        X_df = self._prepare_input(X)
        
        # Özellik mühendisliği uygula
        X_fe = self.apply_feature_engineering(X_df)
        
        # Özellikleri seç
        X_selected = X_fe[self.all_features]
        
        # Ön işle
        X_preprocessed = self.preprocessor.transform(X_selected)
        
        # Olasılıkları tahmin et
        probabilities = self.model.predict_proba(X_preprocessed)
        
        return probabilities
    
    def get_feature_names(self) -> List[str]:
        """
        Tüm özellik isimlerinin listesini al
        
        Returns:
            Özellik isimleri listesi
        """
        return self.all_features


def load_pipeline(pipeline_path: str = FINAL_PIPELINE_PATH):
    """
    Kaydedilmiş pipeline'ı pickle dosyasından yükle
    
    Args:
        pipeline_path: Kaydedilmiş pipeline path'i
        
    Returns:
        ChurnPredictionPipeline objesi
    """
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"✅ Pipeline loaded successfully from {pipeline_path}")
        return pipeline
    except FileNotFoundError:
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    except Exception as e:
        raise Exception(f"Error loading pipeline: {str(e)}")


def save_pipeline(pipeline, pipeline_path: str = FINAL_PIPELINE_PATH):
    """
    Pipeline'ı pickle dosyasına kaydet
    
    Args:
        pipeline: ChurnPredictionPipeline objesi
        pipeline_path: Pipeline kaydedilecek path
    """
    try:
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"✅ Pipeline saved successfully to {pipeline_path}")
    except Exception as e:
        raise Exception(f"Error saving pipeline: {str(e)}")


def create_sample_input() -> Dict:
    """
    Test için örnek girdi dictionary'si oluştur
    
    Returns:
        Örnek müşteri verisiyle dictionary
    """
    return {
        'credit_score': 650,
        'country': 'France',
        'gender': 'Male',
        'age': 35,
        'tenure': 5,
        'balance': 100000.0,
        'products_number': 2,
        'credit_card': 1,
        'active_member': 1,
        'estimated_salary': 75000.0
    }


def validate_input_data(input_data: Dict) -> bool:
    """
    Girdi verisinin tüm gerekli alanlara sahip olduğunu doğrula
    
    Args:
        input_data: Müşteri verisiyle dictionary
        
    Returns:
        Geçerliyse True, geçersizse ValueError fırlat
    """
    required_fields = ORIGINAL_NUMERICAL_COLS + ORIGINAL_CATEGORICAL_COLS + ORIGINAL_BINARY_COLS
    
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
    
    return True


def format_prediction_output(prediction: int, probability: np.ndarray) -> Dict:
    """
    Tahmin çıktısını görüntüleme için formatla
    
    Args:
        prediction: Tahmin edilen sınıf (0 veya 1)
        probability: Tahmin olasılıkları [P(0), P(1)]
        
    Returns:
        Formatlanmış çıktıyla dictionary
    """
    return {
        'prediction': int(prediction),
        'prediction_label': 'Churned' if prediction == 1 else 'Not Churned',
        'churn_probability': float(probability[1]),
        'not_churn_probability': float(probability[0]),
        'risk_level': get_risk_level(probability[1])
    }


def get_risk_level(churn_probability: float) -> str:
    """
    Kaybı olasılığına göre risk seviyesini belirle
    
    Args:
        churn_probability: Kaybı olasılığı (0-1)
        
    Returns:
        Risk seviyesi string
    """
    if churn_probability >= 0.7:
        return 'Yüksek Risk'
    elif churn_probability >= 0.5:
        return 'Orta Risk'
    elif churn_probability >= 0.3:
        return 'Düşük-Orta Risk'
    else:
        return 'Düşük Risk'


# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Modülü - Banka Müşteri Kaybı Tahmini")
    print("=" * 60)
    
    # Örnek girdi ile test
    print("\n📝 Örnek girdi oluşturuluyor...")
    sample_input = create_sample_input()
    print(f"\nÖrnek Girdi:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    # Girdiyi doğrula
    print("\n✅ Girdi doğrulanıyor...")
    try:
        validate_input_data(sample_input)
        print("Girdi geçerli!")
    except ValueError as e:
        print(f"❌ Doğrulama hatası: {e}")
    
    print("\n" + "=" * 60)
    print("Not: Pipeline'ı kullanmak için load_pipeline() ile yükleyin")
    print("=" * 60)
