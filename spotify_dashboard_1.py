import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import numpy as np

# --- Konfigurasi Halaman & Styling Kustom ---
st.set_page_config(
    page_title="🎧 Analisis Musik Terbaik Spotify 2023",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Masukkan CSS kustom
st.markdown("""
<style>
    /* Styling utama */
    .main {
        background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
    }
    
    /* Kartu metrik yang terinspirasi dari Spotify */
    [data-testid="stMetric"] {
        background: rgba(30, 30, 46, 0.7);
        border: 1px solid #4f4f6b;
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetric"] > div:nth-child(2) > div {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1DB954;
        text-shadow: 0 0 10px rgba(29, 185, 84, 0.3);
    }
    
    [data-testid="stMetric"] > label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #b3b3b3;
    }
    
    /* Styling tab kustom */
    .stTabs [role="tablist"] {
        gap: 10px;
    }
    
    .stTabs [role="tab"] {
        padding: 12px 20px;
        border-radius: 8px 8px 0 0;
        background: #2a2a39;
        color: #b3b3b3;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        background: #1DB954;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Fungsi Memuat Data ---
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path, encoding='latin-1')
    except FileNotFoundError:
        st.error(f"File tidak ditemukan di path: {path}. Pastikan 'spotify-2023.csv' berada di direktori yang sama.")
        return None
        
    # <-- DIPERBAIKI: Logika pembersihan nama kolom yang baru dan lebih aman
    # 1. Buat semua nama kolom menjadi huruf kecil dan ganti spasi
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # 2. Buat kamus untuk mengganti nama kolom yang bermasalah secara eksplisit
    rename_map = {
        'artist(s)_name': 'artist_name',
        'danceability_%': 'danceability',
        'valence_%': 'valence',
        'energy_%': 'energy',
        'acousticness_%': 'acousticness',
        'instrumentalness_%': 'instrumentalness',
        'liveness_%': 'liveness',
        'speechiness_%': 'speechiness'
    }
    
    # 3. Ganti nama kolom menggunakan kamus
    df.rename(columns=rename_map, inplace=True)
    
    # Menangani nilai non-numerik di kolom 'streams'
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df.dropna(subset=['streams'], inplace=True)
    df['streams'] = df['streams'].astype('int64')
    
    # Membuat kolom tanggal gabungan untuk analisis deret waktu yang lebih baik
    df['released_date'] = pd.to_datetime(df[['released_year', 'released_month', 'released_day']].astype(str).agg('-'.join, axis=1), errors='coerce')
    
    # Menghapus duplikat
    df.drop_duplicates(inplace=True)
    
    # Membuat kolom dekade untuk analisis
    df['decade'] = (df['released_year'] // 10) * 10
    
    return df

# --- Memuat Data ---
df = load_data('spotify-2023.csv')

if df is None:
    st.stop()

# --- Filter di Sidebar ---
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_White.png", width=200)
    st.title("🎛️ Panel Kontrol")
    st.markdown("Kustomisasi pengalaman eksplorasi musik Anda")
    
    st.subheader("Pilihan Artis")
    selected_artists_count = st.slider(
        'Jumlah Artis Teratas', 5, 30, 15,
        help="Pilih berapa banyak artis teratas yang akan ditampilkan dalam visualisasi"
    )
    
    st.subheader("Rentang Tahun Rilis")
    min_year, max_year = int(df['released_year'].min()), int(df['released_year'].max())
    year_range = st.slider(
        'Pilih Rentang Tahun Rilis', min_year, max_year, (min_year, max_year),
        help="Filter lagu berdasarkan tahun rilisnya"
    )
    
    st.subheader("Karakteristik Audio")
    bpm_range = st.slider(
        'Rentang BPM', 
        int(df['bpm'].min()), int(df['bpm'].max()), 
        (int(df['bpm'].min()), int(df['bpm'].max()))
    )
    
    col1, col2 = st.columns(2)
    with col1:
        key_options = sorted(df['key'].dropna().unique().tolist())
        all_keys = ['Semua'] + key_options
        selected_key = st.selectbox('Kunci Musik', all_keys, index=0)

    with col2:
        mode_options = sorted(df['mode'].dropna().unique().tolist())
        all_modes = ['Semua'] + mode_options
        selected_mode = st.selectbox('Mode', all_modes, index=0)
    
    with st.expander("Filter Lanjutan"):
        danceability = st.slider('Danceability (%)', 0, 100, (0, 100))
        energy = st.slider('Energy (%)', 0, 100, (0, 100))
        valence = st.slider('Positivity (Valence) (%)', 0, 100, (0, 100))
    
    st.markdown("""
    <a href="https://open.spotify.com" target="_blank" style="text-decoration: none;">
        <div style="background-color: #1DB954; color: white; padding: 10px; border-radius: 25px; text-align: center; font-weight: bold; margin-top: 20px;">
            🎵 Buka Spotify
        </div>
    </a>
    """, unsafe_allow_html=True)

# Terapkan filter dasar
filtered_df = df[
    (df['released_year'] >= year_range[0]) & 
    (df['released_year'] <= year_range[1]) &
    (df['bpm'] >= bpm_range[0]) & 
    (df['bpm'] <= bpm_range[1])
]

if selected_key != 'Semua':
    filtered_df = filtered_df[filtered_df['key'] == selected_key]
if selected_mode != 'Semua':
    filtered_df = filtered_df[filtered_df['mode'] == selected_mode]

# Terapkan filter lanjutan
filtered_df = filtered_df[
    (filtered_df['danceability'] >= danceability[0]) & 
    (filtered_df['danceability'] <= danceability[1]) &
    (filtered_df['energy'] >= energy[0]) & 
    (filtered_df['energy'] <= energy[1]) &
    (filtered_df['valence'] >= valence[0]) & 
    (filtered_df['valence'] <= valence[1])
]

# --- Halaman Utama ---
st.title("🎧 Analisis Musik Terbaik Spotify 2023")
st.markdown("""
<div style="background: rgba(29, 185, 84, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #1DB954;">
    <p style="color: #b3b3b3; margin: 0;">Jelajahi lagu, artis, dan tren musik terpopuler dari dataset Spotify 2023. 
    Temukan pola tersembunyi, bandingkan fitur audio, dan temukan lagu favorit Anda berikutnya!</p>
</div>
""", unsafe_allow_html=True)

# Cegah error jika dataframe kosong setelah filter
if filtered_df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih. Silakan sesuaikan kembali filter Anda.")
    st.stop()

# --- Baris Metrik ---
st.subheader("📊 Metrik Kunci")
total_tracks = len(filtered_df)
total_artists = filtered_df['artist_name'].nunique()
total_streams = filtered_df['streams'].sum()

# <-- DIPERBAIKI: Menghapus metrik durasi dan menyesuaikan kolom
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Lagu", f"{total_tracks:,}", help="Jumlah lagu dalam pilihan saat ini")
with col2:
    st.metric("Artis Unik", f"{total_artists:,}", help="Jumlah artis unik dalam pilihan saat ini")
with col3:
    st.metric("Total Streaming", f"{total_streams / 1_000_000_000:.2f} Miliar", help="Jumlah semua streaming dalam pilihan saat ini")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Gambaran Umum", "🎤 Artis & Lagu", "🎧 Analisis Audio", "📅 Tren dari Waktu ke Waktu"])

with tab1:
    st.header("Lanskap Musik Global")
    # ... (kode di tab ini tidak perlu diubah)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("🌐 Distribusi Kebangsaan Artis")
        st.info("Fitur ini memerlukan data negara di dalam dataset untuk dapat berfungsi secara akurat.")
        # Data placeholder
        country_data = {'location': ["USA", "GBR", "CAN", "AUS", "DEU", "FRA", "BRA", "MEX", "IND", "KOR"],
                        'artists': [10, 8, 6, 5, 4, 4, 3, 3, 2, 2]}
        country_df = pd.DataFrame(country_data)
        fig_map = px.choropleth(country_df, locations="location", locationmode="ISO-3",
                                color="artists",
                                color_continuous_scale=px.colors.sequential.Viridis,
                                title="Negara Asal Artis (Data Sampel)")
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.subheader("📱 Popularitas Platform")
        st.info("Data ini adalah ilustrasi dan tidak berasal dari dataset.")
        platforms = {'Spotify': 45, 'Apple Music': 25, 'YouTube Music': 15, 'Amazon Music': 10, 'Lainnya': 5}
        fig_platform = px.pie(names=list(platforms.keys()), values=list(platforms.values()),
                              hole=0.4, color_discrete_sequence=px.colors.sequential.Viridis)
        fig_platform.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_platform, use_container_width=True)
    
    st.subheader("🔠 Kata Paling Umum dalam Nama Lagu")
    text = " ".join(track for track in filtered_df['track_name'])
    wordcloud = WordCloud(width=800, height=400, background_color='#121212', colormap='viridis').generate(text)
    
    fig_wc, ax = plt.subplots(figsize=(10, 5), facecolor='#121212')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)


with tab2:
    st.header("Analisis Artis & Lagu")
    st.subheader("🎤 Performa Artis Teratas")
    
    df_artists = filtered_df.copy()
    df_artists['artist_name'] = df_artists['artist_name'].str.split(',_')
    df_artists = df_artists.explode('artist_name')
    
    col1, col2 = st.columns(2)
    with col1:
        top_artists_count = df_artists['artist_name'].value_counts().nlargest(selected_artists_count)
        fig_artists_count = px.bar(top_artists_count, y=top_artists_count.index, x=top_artists_count.values,
                                   orientation='h', title=f'{selected_artists_count} Artis Teratas Berdasarkan Jumlah Lagu',
                                   labels={'x': 'Jumlah Lagu', 'y': 'Artis'},
                                   color=top_artists_count.values,
                                   color_continuous_scale=px.colors.sequential.Viridis)
        fig_artists_count.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_artists_count, use_container_width=True)
    
    with col2:
        top_artists_streams = df_artists.groupby('artist_name')['streams'].sum().nlargest(selected_artists_count)
        fig_artists_streams = px.bar(top_artists_streams, y=top_artists_streams.index, x=top_artists_streams.values,
                                     orientation='h', title=f'{selected_artists_count} Artis Teratas Berdasarkan Total Streaming',
                                     labels={'x': 'Total Streaming (Miliar)', 'y': 'Artis'},
                                     color=top_artists_streams.values,
                                     color_continuous_scale=px.colors.sequential.Plasma)
        fig_artists_streams.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_artists_streams, use_container_width=True)
    
    st.subheader("🎵 Analisis Lagu Teratas")
    
    col1, col2 = st.columns(2)
    with col1:
        top_tracks = filtered_df.sort_values(by='streams', ascending=False).head(10)
        fig_top_tracks = px.bar(top_tracks, x='streams', y='track_name', orientation='h',
                                color='artist_name', title='10 Lagu Teratas Berdasarkan Streaming',
                                labels={'streams': 'Streaming (Miliar)', 'track_name': 'Nama Lagu'},
                                color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_top_tracks.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_tracks, use_container_width=True)
    
    with col2:
        # <-- DIPERBAIKI: Mengganti grafik durasi dengan danceability vs streams
        st.subheader("💃 Danceability vs. Popularitas")
        fig_dance = px.scatter(filtered_df.nlargest(100, 'streams'), 
                               x='danceability', y='streams',
                               color='energy', size='streams',
                               title='Danceability vs Popularitas (100 Lagu Teratas)',
                               labels={'danceability': 'Danceability (%)', 'streams': 'Streaming'},
                               hover_data=['track_name', 'artist_name'])
        st.plotly_chart(fig_dance, use_container_width=True)

with tab3:
    st.header("Analisis Audio Mendalam")
    st.subheader("🎼 Karakteristik Musikal")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎹 Distribusi Kunci & Mode")
        key_mode_df = filtered_df.groupby(['key', 'mode']).size().reset_index(name='counts')
        if not key_mode_df.empty:
            fig_sunburst = px.sunburst(key_mode_df, path=['key', 'mode'], values='counts',
                                       color='key', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk Kunci & Mode pada filter ini.")
    
    with col2:
        st.markdown("#### 🏃 Distribusi BPM (Beats Per Minute)")
        fig_bpm = px.histogram(filtered_df, x='bpm', nbins=30, 
                               title='Distribusi Tempo Lagu',
                               color_discrete_sequence=['#1DB954'])
        st.plotly_chart(fig_bpm, use_container_width=True)
    
    st.subheader("🎚 Profil Fitur Audio")
    audio_features = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    
    avg_features = filtered_df[audio_features].mean().reset_index()
    avg_features.columns = ['feature', 'value']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_features['value'], theta=avg_features['feature'], fill='toself',
        name='Rata-rata Pilihan', line=dict(color='#1DB954', width=2)
    ))
    
    overall_avg = df[audio_features].mean().values
    fig_radar.add_trace(go.Scatterpolar(
        r=overall_avg, theta=avg_features['feature'],
        name='Rata-rata Keseluruhan', line=dict(color='#b3b3b3', width=2)
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.2)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        ),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.subheader("🔄 Analisis Korelasi Fitur")
    corr_df = filtered_df[audio_features + ['streams', 'bpm']].corr()
    fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale=px.colors.diverging.RdYlGn,
                         zmin=-1, zmax=1, title="Matriks Korelasi Fitur Audio")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.header("Tren Musik dari Waktu ke Waktu")
    st.subheader("📅 Tren Rilisan")
    
    col1, col2 = st.columns(2)
    with col1:
        songs_per_year = filtered_df['released_year'].value_counts().sort_index()
        fig_year = px.area(songs_per_year, x=songs_per_year.index, y=songs_per_year.values,
                           title='Lagu yang Dirilis per Tahun',
                           labels={'x': 'Tahun', 'y': 'Jumlah Lagu'},
                           color_discrete_sequence=['#1DB954'])
        st.plotly_chart(fig_year, use_container_width=True)
    
    with col2:
        songs_per_month = filtered_df['released_month'].value_counts().sort_index()
        fig_month = px.bar(songs_per_month, x=songs_per_month.index, y=songs_per_month.values,
                           title='Lagu yang Dirilis per Bulan',
                           labels={'x': 'Bulan', 'y': 'Jumlah Lagu'},
                           color=songs_per_month.values,
                           color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_month, use_container_width=True)
    
    st.subheader("🎵 Evolusi Fitur dari Waktu ke Waktu")
    decade_features = filtered_df.groupby('decade')[audio_features].mean().reset_index()
    fig_decade = px.line(decade_features, x='decade', y=audio_features,
                         title='Tren Fitur Audio per Dekade',
                         labels={'value': 'Nilai Fitur (%)', 'variable': 'Fitur'},
                         color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_decade, use_container_width=True)
    
    st.subheader("🥁 Tren Tempo dari Waktu ke Waktu")
    bpm_trend = filtered_df.groupby('released_year')['bpm'].mean().reset_index()
    fig_bpm_trend = px.line(bpm_trend, x='released_year', y='bpm',
                            title='Rata-rata BPM dari Waktu ke Waktu',
                            labels={'released_year': 'Tahun', 'bpm': 'Beats Per Minute'},
                            color_discrete_sequence=['#1DB954'])
    st.plotly_chart(fig_bpm_trend, use_container_width=True)

# --- Penjelajah Data ---
st.sidebar.header("🔍 Penjelajah Data")
if st.sidebar.checkbox("Tampilkan Data Mentah"):
    st.subheader("📋 Pratinjau Dataset yang Difilter")
    st.dataframe(filtered_df.head(100))
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Unduh Data yang Difilter", data=csv,
        file_name=f"spotify_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# --- Footer ---
st.markdown("""
<hr style="border: 1px solid rgba(255, 255, 255, 0.1); margin: 30px 0;">
<div style="text-align: center; color: #b3b3b3; font-size: 0.9em;">
    <p>🎵 Dasbor Analisis Musik Spotify 2023</p>
    <p>Sumber data: Spotify | Dibuat dengan Streamlit dan Plotly</p>
</div>
""", unsafe_allow_html=True)