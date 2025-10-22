# app.py
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Water Leak Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .header-title {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .leak-detected {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .no-leak {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00aa00;
    }
    </style>
    """, unsafe_allow_html=True)

# Load preprocessing objects (EXACTLY as saved in Cell 4)
@st.cache_resource
def load_preprocessing_objects():
    """Load scaler, label encoder, and feature selector"""
    models_path = Path("results/models")
    
    try:
        scaler = joblib.load(models_path / "scaler.pkl")
        label_encoder = joblib.load(models_path / "label_encoder.pkl")
        feature_selector_info = joblib.load(models_path / "feature_selector.pkl")
        
        return scaler, label_encoder, feature_selector_info
    except Exception as e:
        st.error(f"Error loading preprocessing objects: {e}")
        return None, None, None

# Load models (EXACTLY as saved in Cell 4)
@st.cache_resource
def load_all_models():
    """Load all trained models"""
    models_path = Path("results/models")
    
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'Gradient Boosting': 'gradient_boosting_model.pkl',
        'SVM': 'svm_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'LightGBM': 'lightgbm_model.pkl',
        'Neural Network': 'neural_network_model.pkl'
    }
    
    loaded_models = {}
    failed = []
    
    for model_name, filename in model_files.items():
        filepath = models_path / filename
        if filepath.exists():
            try:
                model = joblib.load(filepath)
                loaded_models[model_name] = model
            except Exception as e:
                failed.append(f"{model_name}: {str(e)[:50]}")
        else:
            failed.append(f"{model_name}: File not found")
    
    if failed:
        st.warning(f"⚠️ Failed to load: {', '.join(failed)}")
    
    return loaded_models

# Load best model info
@st.cache_resource
def load_best_model_info():
    """Load best model details from JSON"""
    try:
        import json
        with open("results/models/best_model_details.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load best model info")
        return None

# Get sample audio files
@st.cache_resource
def get_sample_files():
    """Get all audio files from Samples folder"""
    samples_path = Path("Samples")
    
    if not samples_path.exists():
        return []
    
    # Get all audio files with common extensions
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(samples_path.glob(f'*{ext}'))
        audio_files.extend(samples_path.glob(f'*{ext.upper()}'))
    
    # Sort by name
    audio_files = sorted(list(set(audio_files)))  # Remove duplicates and sort
    
    return audio_files

# Feature extraction (EXACTLY from Cell 3)
def extract_features(y, sr=22050):
    """
    Extract comprehensive features from audio - IDENTICAL to Cell 3
    """
    try:
        features = {}
        
        # 1. Basic Time-Domain Features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['mean_amplitude'] = np.mean(np.abs(y))
        features['std_amplitude'] = np.std(y)
        features['max_amplitude'] = np.max(np.abs(y))
        features['min_amplitude'] = np.min(y)
        features['rms_energy'] = np.sqrt(np.mean(y**2))
        features['peak_amplitude'] = np.max(np.abs(y))
        
        # 2. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_median'] = np.median(zcr)
        features['zcr_max'] = np.max(zcr)
        features['zcr_min'] = np.min(zcr)
        
        # 3. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_median'] = np.median(spectral_centroids)
        features['spectral_centroid_max'] = np.max(spectral_centroids)
        features['spectral_centroid_min'] = np.min(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['spectral_rolloff_median'] = np.median(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        features['spectral_bandwidth_median'] = np.median(spectral_bandwidth)
        
        # 4. Spectral Flatness
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spec_flatness)
        features['spectral_flatness_std'] = np.std(spec_flatness)
        features['spectral_flatness_median'] = np.median(spec_flatness)
        
        # 5. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i+1}_median'] = np.median(mfccs[i])
            features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i+1}_min'] = np.min(mfccs[i])
        
        # 6. Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(5):
            features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta_{i+1}_std'] = np.std(mfcc_delta[i])
        
        # 7. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        features['chroma_median'] = np.median(chroma)
        
        for i in range(12):
            features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
        
        # 8. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        features['spectral_contrast_mean'] = np.mean(contrast)
        features['spectral_contrast_std'] = np.std(contrast)
        features['spectral_contrast_median'] = np.median(contrast)
        
        for i in range(7):
            features[f'spectral_contrast_band_{i+1}_mean'] = np.mean(contrast[i])
        
        # 9. Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        features['tonnetz_median'] = np.median(tonnetz)
        
        # 10. Tempo and Beat
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        features['beat_count'] = len(beats)
        
        # 11. Energy and Dynamics
        features['energy'] = np.sum(y**2)
        features['energy_entropy'] = -np.sum((y**2 + 1e-10) * np.log(y**2 + 1e-10))
        
        # 12. Harmonic and Percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmonic_energy'] = np.sum(y_harmonic**2)
        features['percussive_energy'] = np.sum(y_percussive**2)
        features['harmonic_percussive_ratio'] = features['harmonic_energy'] / (features['percussive_energy'] + 1e-10)
        
        # 13. Low-level frequency features
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        low_freq_mask = freqs < 500
        mid_freq_mask = (freqs >= 500) & (freqs < 2000)
        high_freq_mask = freqs >= 2000
        
        features['low_freq_energy'] = np.sum(stft[low_freq_mask, :]**2)
        features['mid_freq_energy'] = np.sum(stft[mid_freq_mask, :]**2)
        features['high_freq_energy'] = np.sum(stft[high_freq_mask, :]**2)
        
        total_energy = features['low_freq_energy'] + features['mid_freq_energy'] + features['high_freq_energy']
        features['low_freq_ratio'] = features['low_freq_energy'] / (total_energy + 1e-10)
        features['mid_freq_ratio'] = features['mid_freq_energy'] / (total_energy + 1e-10)
        features['high_freq_ratio'] = features['high_freq_energy'] / (total_energy + 1e-10)
        
        # 14. Statistical moments
        features['skewness'] = float(pd.Series(y).skew())
        features['kurtosis'] = float(pd.Series(y).kurtosis())
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Prepare features for prediction (EXACTLY from Cell 4)
def prepare_features_for_prediction(features, feature_selector_info, scaler):
    """
    Prepare extracted features for model prediction
    Uses feature_selector_info from Cell 4
    """
    try:
        # Create DataFrame with all extracted features
        df = pd.DataFrame([features])
        
        # Get selected features list from feature_selector_info
        selected_features = feature_selector_info['selected_features']
        
        # Select only the features used during training
        X = df[selected_features].values
        
        # Scale features using the fitted scaler
        X_scaled = scaler.transform(X)
        
        return X_scaled, selected_features  # Return selected features list too
        
    except KeyError as e:
        st.error(f"❌ Missing feature: {e}")
        return None, None
    except ValueError as e:
        st.error(f"❌ Shape mismatch: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error preparing features: {e}")
        return None, None

# Perform analysis
def perform_analysis(audio_file_path, selected_model, models, scaler, label_encoder, feature_selector_info):
    """
    Perform complete analysis on audio file
    """
    try:
        # Step 1: Load audio
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading audio file...")
        progress_bar.progress(20)
        
        audio_data, sr = librosa.load(audio_file_path, sr=22050)
        st.success(f"✓ Audio loaded (Sample Rate: {sr} Hz, Duration: {len(audio_data)/sr:.2f}s)")
        
        # Step 2: Extract features
        status_text.text("Extracting audio features...")
        progress_bar.progress(40)
        
        features = extract_features(audio_data, sr=sr)
        if features is None:
            st.error("Failed to extract features")
            return None
        
        st.success("✓ Features extracted successfully")
        
        # Step 3: Prepare features
        status_text.text("Preparing features for prediction...")
        progress_bar.progress(60)
        
        X_prepared, selected_features = prepare_features_for_prediction(features, feature_selector_info, scaler)
        if X_prepared is None:
            return None
        
        st.success("✓ Features prepared and scaled")
        
        # Step 4: Make prediction
        status_text.text("Running inference...")
        progress_bar.progress(80)
        
        model = models[selected_model]
        prediction = model.predict(X_prepared)[0]
        
        # prediction is 0 for 'Leak', 1 for 'NoLeak'
        is_leak = prediction == 0
        
        # Get probability
        leak_probability = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_prepared)[0]
            leak_probability = probabilities[0]
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return {
            'is_leak': is_leak,
            'leak_probability': leak_probability,
            'prediction': prediction,
            'audio_data': audio_data,
            'sr': sr,
            'features': features,
            'selected_features': selected_features
        }
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Display results
def display_results(result, selected_model, models):
    """
    Display analysis results
    """
    if result is None:
        return
    
    is_leak = result['is_leak']
    leak_probability = result['leak_probability']
    audio_data = result['audio_data']
    sr = result['sr']
    features = result['features']
    selected_features = result['selected_features']
    
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write(f"**Raw Prediction:** {result['prediction']}")
        if hasattr(models[selected_model], 'predict_proba'):
            st.info("✓ Model supports probability predictions")
    
    prediction_label = "🚨 LEAK DETECTED" if is_leak else "✅ NO LEAK"
    
    if is_leak:
        st.markdown(f'<div class="leak-detected"><h2>{prediction_label}</h2></div>', 
                  unsafe_allow_html=True)
        
        if leak_probability is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Leak Probability", f"{leak_probability*100:.2f}%")
            with col2:
                confidence = "🔴 HIGH" if leak_probability > 0.8 else "🟡 MEDIUM"
                st.metric("Confidence", confidence)
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=leak_probability*100,
                title={'text': "Leak Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 33], 'color': "#d4f4dd"},
                        {'range': [33, 67], 'color': "#fff3cd"},
                        {'range': [67, 100], 'color': "#ffcccc"}
                    ]
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f'<div class="no-leak"><h2>{prediction_label}</h2></div>', 
                  unsafe_allow_html=True)
        
        if leak_probability is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Leak Probability", f"{leak_probability*100:.2f}%")
            with col2:
                confidence = "🟢 HIGH" if leak_probability < 0.2 else "🟡 MEDIUM"
                st.metric("Confidence", confidence)
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=leak_probability*100,
                title={'text': "Leak Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 33], 'color': "#d4f4dd"},
                        {'range': [33, 67], 'color': "#fff3cd"},
                        {'range': [67, 100], 'color': "#ffcccc"}
                    ]
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional info
    st.divider()
    st.subheader("ℹ️ Additional Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Used", selected_model)
    with col2:
        st.metric("Duration", f"{len(audio_data)/sr:.2f}s")
    with col3:
        st.metric("Sample Rate", f"{sr} Hz")
    with col4:
        st.metric("Samples", f"{len(audio_data):,}")
    
    # Feature details
    with st.expander("📈 Extracted Features (30 Selected)"):
        # Get values for selected features
        selected_features_values = {
            feature_name: features[feature_name] 
            for feature_name in selected_features
        }
        
        features_display = pd.DataFrame(
            list(selected_features_values.items()),
            columns=['Feature', 'Value']
        )
        
        features_display = features_display.reset_index(drop=True)
        features_display.index = features_display.index + 1
        
        st.dataframe(features_display, use_container_width=True)
        
        st.info(f"📊 Total Selected Features: {len(selected_features)}")

# Main app
def main():
    st.markdown('<div class="header-title">💧 Water Leak Detection System</div>', unsafe_allow_html=True)
    
    # Load all necessary objects
    models = load_all_models()
    scaler, label_encoder, feature_selector_info = load_preprocessing_objects()
    sample_files = get_sample_files()
    best_info = load_best_model_info()
    
    # Check if everything loaded
    if not models:
        st.error("❌ No models loaded. Please run training script first.")
        return
    
    if scaler is None or feature_selector_info is None:
        st.error("❌ Preprocessing objects not found. Please run training script first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        st.subheader("Select Model")
        selected_model = st.selectbox(
            "Choose a model for inference:",
            list(models.keys())
        )
        
        st.divider()
        st.subheader("📋 Instructions")
        st.info("""
        1. Select a model above
        2. Either upload an audio file or use a sample
        3. Click "Analyze Audio"
        4. View results and confidence scores
        """)
        
        if sample_files:
            st.divider()
            st.subheader("📊 Sample Files Count")
            st.info(f"Available samples: **{len(sample_files)}** files")
    
    # Main content
    st.subheader("🎯 Audio Source")
    
    # Tab selection for upload or sample
    tab1, tab2 = st.tabs(["📤 Upload Audio", "📁 Use Sample"])
    
    audio_file_path = None
    audio_display_name = None
    
    with tab1:
        st.subheader("Upload Your Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            key="uploaded_file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                audio_file_path = tmp_file.name
                audio_display_name = uploaded_file.name
            
            # Audio preview
            st.subheader("🔊 Audio Preview")
            st.audio(uploaded_file)
            
            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            with col2:
                st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
            with col3:
                st.metric("Model", selected_model)
    
    with tab2:
        if sample_files:
            st.subheader("Select a Sample Audio File")
            
            # Create display names
            sample_display_names = [f.name for f in sample_files]
            
            selected_sample_idx = st.selectbox(
                "Available Samples:",
                range(len(sample_files)),
                format_func=lambda idx: sample_display_names[idx],
                key="sample_select"
            )
            
            audio_file_path = str(sample_files[selected_sample_idx])
            audio_display_name = sample_display_names[selected_sample_idx]
            
            # Audio preview
            st.subheader("🔊 Audio Preview")
            try:
                st.audio(audio_file_path)
            except:
                st.warning("Could not load audio preview")
            
            # File info
            file_size = Path(audio_file_path).stat().st_size
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{file_size / 1024:.2f} KB")
            with col2:
                st.metric("File Type", Path(audio_file_path).suffix[1:].upper())
            with col3:
                st.metric("Model", selected_model)
        else:
            st.warning("⚠️ No sample files found in 'Samples' folder")
            st.info("Please add audio files to the 'Samples' folder to use this feature")
    
    st.divider()
    
    # Analyze button
    if st.button("🔍 Analyze Audio", use_container_width=True, type="primary"):
        if audio_file_path:
            result = perform_analysis(
                audio_file_path,
                selected_model,
                models,
                scaler,
                label_encoder,
                feature_selector_info
            )
            
            if result:
                display_results(result, selected_model, models)
        else:
            st.error("❌ Please select or upload an audio file first")

if __name__ == "__main__":
    main()