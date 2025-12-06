"""
Banka Müşteri Kaybı Tahmini için Streamlit Uygulaması

Bu, müşteri kaybını tahmin etmek için web tabanlı bir arayüzdür.
Kullanıcılar müşteri detaylarını girebilir ve gerçek zamanlı tahminler alabilir.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

# Import local modules
from config import (
    APP_TITLE, APP_DESCRIPTION, INPUT_FIELDS,
    CHURN_THRESHOLD, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD,
    FINAL_PIPELINE_PATH
)
from pipeline import ChurnPredictionPipeline, load_pipeline, validate_input_data
from inference import ChurnPredictor


# ==================== SAYFA YAPILANDIRMASI ====================

st.set_page_config(
    page_title="Banka Müşteri Kayıp Tahmini",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== ÖZEL CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ==================== YARDİMCI FONKSİYONLAR ====================

@st.cache_resource
def load_model_pipeline():
    """
    Eğitilmiş pipeline'ı yükle (cache'lenir)
    """
    try:
        predictor = ChurnPredictor(FINAL_PIPELINE_PATH)
        return predictor
    except Exception as e:
        st.error(f"Model yüklenirken hata: {str(e)}")
        st.info("Lütfen model dosyasının doğru yolda olduğundan emin olun.")
        return None


def get_risk_color(probability):
    """
    Kaybı olasılığına göre renk al
    """
    if probability >= HIGH_RISK_THRESHOLD:
        return "high-risk", "🔴", "#f44336"
    elif probability >= CHURN_THRESHOLD:
        return "medium-risk", "🟠", "#ff9800"
    elif probability >= LOW_RISK_THRESHOLD:
        return "low-risk", "🟡", "#ffc107"
    else:
        return "low-risk", "🟢", "#4caf50"


def display_prediction_result(result):
    """
    Tahmin sonuçlarını güzel bir formatta göster
    """
    prediction_label = result['prediction_label']
    churn_prob = result['churn_probability']
    risk_level = result['risk_level']
    
    risk_class, risk_icon, risk_color = get_risk_color(churn_prob)
    
    # Ana tahmin kutusu
    st.markdown(f"""
    <div class="prediction-box {risk_class}">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{risk_icon}</div>
        <div class="metric-value">{prediction_label}</div>
        <div class="metric-label">Tahmin Sonucu</div>
        <hr style="margin: 1.5rem 0;">
        <div class="metric-value" style="color: {risk_color};">{churn_prob:.1%}</div>
        <div class="metric-label">Kayıp Olasılığı</div>
        <div style="margin-top: 1rem; font-size: 1.1rem; font-weight: bold; color: {risk_color};">
            {risk_level}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Ek metrikler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Kalma Olasılığı",
            value=f"{result['not_churn_probability']:.1%}"
        )
    
    with col2:
        st.metric(
            label="Risk Seviyesi",
            value=risk_level
        )
    
    with col3:
        confidence = max(result['churn_probability'], result['not_churn_probability'])
        st.metric(
            label="Güven",
            value=f"{confidence:.1%}"
        )


# ==================== ANA UYGULAMA ====================

def main():
    """
    Ana Streamlit uygulaması
    """
    
    # Başlık
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)
    
    # Model yükle
    predictor = load_model_pipeline()
    
    if predictor is None:
        st.stop()
    
    # Sidebar - Girdi Formu
    st.sidebar.header("📝 Müşteri Bilgileri")
    st.sidebar.markdown("---")
    
    # Girdileri topla
    input_data = {}
    
    # Sayısal girdiler
    st.sidebar.subheader("📊 Sayısal Bilgiler")
    
    input_data['credit_score'] = st.sidebar.slider(
        INPUT_FIELDS['credit_score']['label'],
        min_value=INPUT_FIELDS['credit_score']['min'],
        max_value=INPUT_FIELDS['credit_score']['max'],
        value=INPUT_FIELDS['credit_score']['value'],
        help=INPUT_FIELDS['credit_score']['help']
    )
    
    input_data['age'] = st.sidebar.slider(
        INPUT_FIELDS['age']['label'],
        min_value=INPUT_FIELDS['age']['min'],
        max_value=INPUT_FIELDS['age']['max'],
        value=INPUT_FIELDS['age']['value'],
        help=INPUT_FIELDS['age']['help']
    )
    
    input_data['tenure'] = st.sidebar.slider(
        INPUT_FIELDS['tenure']['label'],
        min_value=INPUT_FIELDS['tenure']['min'],
        max_value=INPUT_FIELDS['tenure']['max'],
        value=INPUT_FIELDS['tenure']['value'],
        help=INPUT_FIELDS['tenure']['help']
    )
    
    input_data['balance'] = st.sidebar.number_input(
        INPUT_FIELDS['balance']['label'],
        min_value=INPUT_FIELDS['balance']['min'],
        max_value=INPUT_FIELDS['balance']['max'],
        value=INPUT_FIELDS['balance']['value'],
        step=INPUT_FIELDS['balance']['step'],
        help=INPUT_FIELDS['balance']['help']
    )
    
    input_data['products_number'] = st.sidebar.slider(
        INPUT_FIELDS['products_number']['label'],
        min_value=INPUT_FIELDS['products_number']['min'],
        max_value=INPUT_FIELDS['products_number']['max'],
        value=INPUT_FIELDS['products_number']['value'],
        help=INPUT_FIELDS['products_number']['help']
    )
    
    input_data['estimated_salary'] = st.sidebar.number_input(
        INPUT_FIELDS['estimated_salary']['label'],
        min_value=INPUT_FIELDS['estimated_salary']['min'],
        max_value=INPUT_FIELDS['estimated_salary']['max'],
        value=INPUT_FIELDS['estimated_salary']['value'],
        step=INPUT_FIELDS['estimated_salary']['step'],
        help=INPUT_FIELDS['estimated_salary']['help']
    )
    
    # Kategorik girdiler
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Kategorik Bilgiler")
    
    input_data['country'] = st.sidebar.selectbox(
        INPUT_FIELDS['country']['label'],
        options=INPUT_FIELDS['country']['options'],
        help=INPUT_FIELDS['country']['help']
    )
    
    input_data['gender'] = st.sidebar.selectbox(
        INPUT_FIELDS['gender']['label'],
        options=INPUT_FIELDS['gender']['options'],
        help=INPUT_FIELDS['gender']['help']
    )
    
    # Binary girdiler
    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ İkili Bilgiler")
    
    input_data['credit_card'] = int(st.sidebar.checkbox(
        INPUT_FIELDS['credit_card']['label'],
        value=INPUT_FIELDS['credit_card']['value'],
        help=INPUT_FIELDS['credit_card']['help']
    ))
    
    input_data['active_member'] = int(st.sidebar.checkbox(
        INPUT_FIELDS['active_member']['label'],
        value=INPUT_FIELDS['active_member']['value'],
        help=INPUT_FIELDS['active_member']['help']
    ))
    
    # Tahmin butonu
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🎯 Kayıp Tahmini Yap", type="primary", use_container_width=True)
    
    # Ana içerik alanı
    if predict_button:
        with st.spinner("Tahmin yapılıyor..."):
            try:
                # Tahmin yap
                result = predictor.predict_single(input_data)
                
                # Sonuçları göster
                st.success("✅ Tahmin tamamlandı!")
                
                # Tahmin sonucu
                st.markdown("### 🎯 Tahmin Sonucu")
                display_prediction_result(result)
                
                # Girdi özeti
                with st.expander("📋 Girdi Verilerini Gör"):
                    st.json(input_data)
                
                # Tam sonuç
                with st.expander("🔍 Detaylı Sonuçları Gör"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"❌ Tahmin hatası: {str(e)}")
    
    else:
        # Başlangıç durumu - talimatları göster
        st.info("👈 Lütfen yan çubukta müşteri detaylarını girin ve 'Kayıp Tahmini Yap' butonuna tıklayın")
        
        # Örnek bilgi göster
        st.markdown("### 📊 Nasıl Çalışır")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1️⃣ Veri Girin
            Yan çubuktaki formu kullanarak müşteri bilgilerini sağlayın.
            """)
        
        with col2:
            st.markdown("""
            #### 2️⃣ Tahmin Alın
            Analiz için 'Kayıp Tahmini Yap' butonuna tıklayın.
            """)
        
        with col3:
            st.markdown("""
            #### 3️⃣ Sonuçları Görün
            Kayıp olasılığı ve risk seviyesini anlık görün.
            """)
        
        st.markdown("---")
        
        # Özellik önemi bilgisi
        st.markdown("### 📈 Düşünülen Temel Faktörler")
        st.markdown("""
        Model şunlar dahil birden fazla faktörü analiz eder:
        - 💳 **Kredi Skoru**: Müşteri kredi güvenilirliği
        - 👤 **Yaş & Müşteri Olma Süresi**: Müşteri demografisi ve bağlılık
        - 💰 **Bakiye & Maaş**: Finansal göstergeler
        - 🏢 **Ürünler**: Banka ürünü sayısı
        - 🌍 **Konum**: Ülke ve bölgesel faktörler
        - ✅ **Etkileşim**: Kredi kartı ve aktif üyelik durumu
        """)
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>Banka Müşteri Kayıp Tahmin Sistemi | Makine Öğrenmesi Tabanlı</small>
    </div>
    """, unsafe_allow_html=True)


# ==================== UYGULAMAYI ÇALIŞTIR ====================

if __name__ == "__main__":
    main()
