"""
Banka Müşteri Kaybı Tahmini için Çıkarım (Inference) Modülü

Bu modül şunları sağlar:
- Final pipeline'ı yükleme
- Yeni veri üzerinde tahmin yapma
- Sonuçları formatlama ve döndürme
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union, List
import sys
import warnings

warnings.filterwarnings('ignore')

# Import from local modules
from config import FINAL_PIPELINE_PATH
from pipeline import (
    load_pipeline, 
    validate_input_data, 
    format_prediction_output,
    create_sample_input
)


class ChurnPredictor:
    """
    Kaybı tahmini çıkarımı için wrapper sınıf
    """
    
    def __init__(self, pipeline_path: str = FINAL_PIPELINE_PATH):
        """
        Pipeline yükleyerek predictor'ı başlat
        
        Args:
            pipeline_path: Kaydedilmiş pipeline pickle dosya path'i
        """
        self.pipeline = load_pipeline(pipeline_path)
        self.pipeline_path = pipeline_path
        print(f"✅ ChurnPredictor initialized with pipeline from {pipeline_path}")
    
    def predict_single(self, input_data: Dict) -> Dict:
        """
        Tek bir müşteri için kaybı tahmini yap
        
        Args:
            input_data: Müşteri verisiyle dictionary
            
        Returns:
            Tahmin sonuçlarıyla dictionary
        """
        # Girdiyi doğrula
        validate_input_data(input_data)
        
        # Tahmin yap
        prediction = self.pipeline.predict(input_data)[0]
        probability = self.pipeline.predict_proba(input_data)[0]
        
        # Çıktıyı formatla
        result = format_prediction_output(prediction, probability)
        result['input_data'] = input_data
        
        return result
    
    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        """
        Çok sayıda müşteri için kaybı tahmini yap
        
        Args:
            input_list: Müşteri verisiyle dictionary'lerin listesi
            
        Returns:
            Tahmin sonuçlarıyla dictionary'lerin listesi
        """
        results = []
        
        for i, input_data in enumerate(input_list):
            try:
                result = self.predict_single(input_data)
                result['index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'input_data': input_data
                })
        
        return results
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Müşterilerden oluşan DataFrame için kaybı tahmini yap
        
        Args:
            df: Müşteri verisiyle DataFrame
            
        Returns:
            Tahminler eklenmiş DataFrame
        """
        # Tahminleri yap
        predictions = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)
        
        # DataFrame'e ekle
        df_result = df.copy()
        df_result['prediction'] = predictions
        df_result['churn_probability'] = probabilities[:, 1]
        df_result['prediction_label'] = df_result['prediction'].map({
            0: 'Not Churned', 
            1: 'Churned'
        })
        
        return df_result


def predict_from_json(json_input: Union[str, Dict]) -> Dict:
    """
    JSON girdisinden tahmin yap
    
    Args:
        json_input: JSON string veya dictionary
        
    Returns:
        Tahmin sonuçlarıyla dictionary
    """
    # String ise JSON'u parse et
    if isinstance(json_input, str):
        input_data = json.loads(json_input)
    else:
        input_data = json_input
    
    # Predictor oluştur ve tahmin yap
    predictor = ChurnPredictor()
    result = predictor.predict_single(input_data)
    
    return result


def predict_from_csv(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    CSV dosyasından tahminler yap
    
    Args:
        csv_path: Girdi CSV dosya path'i
        output_path: Sonuç CSV'yi kaydetmek için opsiyonel path
        
    Returns:
        Tahminlerle DataFrame
    """
    # Veriyi yükle
    df = pd.read_csv(csv_path)
    
    # Predictor oluştur ve tahmin yap
    predictor = ChurnPredictor()
    df_result = predictor.predict_from_dataframe(df)
    
    # Output path verilmişse kaydet
    if output_path:
        df_result.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}")
    
    return df_result


def cli_inference():
    """
    Çıkarım için komut satırı arayüzü
    
    Kullanım:
        python inference.py --input '{"credit_score": 650, ...}'
        python inference.py --file input.csv --output output.csv
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Banka Müşteri Kaybı Tahmini')
    parser.add_argument('--input', type=str, help='JSON girdi string')
    parser.add_argument('--file', type=str, help='Girdi CSV dosya path')
    parser.add_argument('--output', type=str, help='Çıktı CSV dosya path')
    parser.add_argument('--sample', action='store_true', help='Örnek girdi ile çalıştır')
    
    args = parser.parse_args()
    
    if args.sample:
        # Örnek girdi kullan
        print("\n📝 Örnek girdi ile tahmin yapılıyor...")
        sample_input = create_sample_input()
        print(f"\nGirdi:")
        print(json.dumps(sample_input, indent=2))
        
        result = predict_from_json(sample_input)
        
        print(f"\n🎯 Tahmin Sonuçları:")
        print(f"  Tahmin: {result['prediction_label']}")
        print(f"  Kaybı Olasılığı: {result['churn_probability']:.4f}")
        print(f"  Risk Seviyesi: {result['risk_level']}")
    
    elif args.input:
        # JSON girdisi
        print(f"\n📝 JSON girdisi ile tahmin yapılıyor...")
        result = predict_from_json(args.input)
        
        print(f"\n🎯 Tahmin Sonuçları:")
        print(json.dumps(result, indent=2))
    
    elif args.file:
        # CSV dosya girdisi
        print(f"\n📝 CSV dosyasından tahminler yapılıyor: {args.file}")
        df_result = predict_from_csv(args.file, args.output)
        
        print(f"\n✅ {len(df_result)} kayıt işlendi")
        print(f"\nTahmin Özeti:")
        print(df_result['prediction_label'].value_counts())
        print(f"\nOrtalama Kaybı Olasılığı: {df_result['churn_probability'].mean():.4f}")
    
    else:
        parser.print_help()


def interactive_inference():
    """
    Tahmin testleri için interaktif mod
    """
    print("=" * 60)
    print("Banka Müşteri Kaybı Tahmini - İnteraktif Mod")
    print("=" * 60)
    
    # Predictor yükle
    predictor = ChurnPredictor()
    
    while True:
        print("\n" + "-" * 60)
        print("Seçenekler:")
        print("  1. Örnek girdi kullan")
        print("  2. Özel girdi gir")
        print("  3. Çıkış")
        print("-" * 60)
        
        choice = input("\nSeçim yap (1-3): ").strip()
        
        if choice == '1':
            # Örnek girdi
            input_data = create_sample_input()
            print(f"\nÖrnek girdi kullanılıyor:")
            print(json.dumps(input_data, indent=2))
            
            result = predictor.predict_single(input_data)
            
            print(f"\n🎯 Tahmin Sonuçları:")
            print(f"  Tahmin: {result['prediction_label']}")
            print(f"  Kaybı Olasılığı: {result['churn_probability']:.4f}")
            print(f"  Kaybetmeme Olasılığı: {result['not_churn_probability']:.4f}")
            print(f"  Risk Seviyesi: {result['risk_level']}")
        
        elif choice == '2':
            # Özel girdi
            print("\nMüşteri detaylarını girin:")
            try:
                input_data = {
                    'credit_score': int(input("  Kredi Skoru (300-850): ")),
                    'country': input("  Ülke (France/Germany/Spain): "),
                    'gender': input("  Cinsiyet (Male/Female): "),
                    'age': int(input("  Yaş: ")),
                    'tenure': int(input("  Müşteri Olma Süresi (yıl): ")),
                    'balance': float(input("  Bakiye: ")),
                    'products_number': int(input("  Ürün Sayısı: ")),
                    'credit_card': int(input("  Kredi Kartı Var mı (0/1): ")),
                    'active_member': int(input("  Aktif Üye (0/1): ")),
                    'estimated_salary': float(input("  Tahmini Maaş: "))
                }
                
                result = predictor.predict_single(input_data)
                
                print(f"\n🎯 Tahmin Sonuçları:")
                print(f"  Tahmin: {result['prediction_label']}")
                print(f"  Kaybı Olasılığı: {result['churn_probability']:.4f}")
                print(f"  Kaybetmeme Olasılığı: {result['not_churn_probability']:.4f}")
                print(f"  Risk Seviyesi: {result['risk_level']}")
            
            except Exception as e:
                print(f"\n❌ Hata: {str(e)}")
        
        elif choice == '3':
            print("\nÇıkılıyor...")
            break
        
        else:
            print("\n❌ Geçersiz seçim. Lütfen 1, 2, veya 3 girin.")


# ==================== MAIN ====================

if __name__ == "__main__":
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1:
        cli_inference()
    else:
        interactive_inference()
