import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Беттің баптаулары
st.set_page_config(page_title="AI Quake KZ", page_icon="🇰🇿", layout="wide")

st.title("AI Quake KZ 🇰🇿")
st.subheader("Жер сілкінісінің қауіптілік деңгейін талдау және ескерту веб-платформасы")
st.markdown("---")

# 1. ДЕРЕКТЕРДІ ИНТЕРНЕТТЕН ЖИНАУ ЖӘНЕ ӨҢДЕУ
@st.cache_data(ttl=600)
def load_and_clean_data():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
    df = pd.read_csv(url)
    
    df = df.dropna(subset=["latitude", "longitude", "depth", "mag", "place"])
    
    # Қазақстан аумағын сүзгілеу
    df_kz = df[(df["latitude"] >= 38) & (df["latitude"] <= 56) & 
              (df["longitude"] >= 45) & (df["longitude"] <= 88)].copy()
              
    def label_danger(mag):
        if mag < 4.0: return 0
        elif mag < 5.0: return 1
        else: return 2
        
    df_kz["danger"] = df_kz["mag"].apply(label_danger)
    return df_kz

try:
    df_kz = load_and_clean_data()
    
    if df_kz.empty:
        st.warning("Қазақстан аймағында соңғы айда деректер табылған жоқ.")
    else:
        # 2. RANDOM FOREST МОДЕЛІН ҮЙРЕТУ
        X = df_kz[["latitude", "longitude", "depth"]]
        y = df_kz["danger"]
        
        if len(df_kz) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            ml_model.fit(X_train, y_train)
            accuracy = ml_model.score(X_test, y_test) if len(X_test) > 0 else 1.0
        else:
            accuracy = 1.0

        # 3. ЕСКЕРТУ ЖҮЙЕСІ ЖӘНЕ СӨЗДІК ТАЛДАУ ПАНЕЛІ
        max_mag = df_kz["mag"].max()
        
        st.sidebar.header("🤖 Жүйелік талдау панелі")
        st.sidebar.markdown("**Әзірлеуші:** Алимбаева Жансая")
        st.sidebar.markdown(f"**Random Forest Дәлдігі:** `{accuracy:.2f}`")
        st.sidebar.markdown("---")
        
        if max_mag >= 5.0:
            st.error(f"🚨 ҚАУІПТІ СІЛКІНІС АНЫҚТАЛДЫ! Ең жоғары магнитуда: {max_mag}")
            st.sidebar.error("Мәртебе: Қауіп жоғары!")
        elif max_mag >= 4.0:
            st.warning(f"⚠️ НАЗАР АУДАРЫҢЫЗ: Орташа деңгейлі сейсмикалық белсенділік. Ең жоғары магнитуда: {max_mag}")
            st.sidebar.warning("Мәртебе: Орташа қауіп")
        else:
            st.success(f"🟢 Қазақстан аймағында жағдай тұрақты. Ең жоғары магнитуда: {max_mag}")
            st.sidebar.success("Мәртебе: Қауіп жоқ")

        # ВЕБ-САЙТТЫҢ ВКЛАДКАЛАРЫ
        tab1, tab2, tab3 = st.tabs(["🗺 Интерактивті Карта", "📊 Графиктер мен Талдау", "📋 Деректер қоры"])
        
        with tab1:
            st.markdown("### 🗺 Жер сілкіністерінің гео-картасы")
            m = folium.Map(location=[43.25, 76.95], zoom_start=5)
            
            for _, row in df_kz.iterrows():
                mag = row["mag"]
                color = "green" if mag < 4.0 else "orange" if mag < 5.0 else "red"
                
                popup_html = f"<b>Орыны:</b> {row['place']}<br><b>Магнитуда:</b> {mag}<br><b>Тереңдік:</b> {row['depth']} км"
                
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=int(mag) * 2 + 2,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(m)
                
            st_folium(m, width="100%", height=500, returned_objects=[])

        with tab2:
            st.markdown("### 📊 Аймақтар бойынша интерактивті талдау")
            st.info("💡 Кеңес: График элементтерінің үстіне тышқанды апарсаңыз, аймақ атауы көрінеді.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📈 Жер сілкіністерінің магнитудалық таралуы")
                chart1 = alt.Chart(df_kz).mark_bar(color="skyblue").encode(
                    x=alt.X('mag:Q', title='Магнитуда (Күші)'),
                    y=alt.Y('count():Q', title='Сілкініс саны'),
                    tooltip=[
                        alt.Tooltip('place:N', title='Болған орны'),
                        alt.Tooltip('mag:Q', title='Магнитудасы'),
                        alt.Tooltip('time:N', title='Уақыты')
                    ]
                ).properties(height=400)
                st.altair_chart(chart1, use_container_width=True)
                
            with col2:
                st.markdown("##### 🎯 Тереңдік пен Күштің байланысы")
                chart2 = alt.Chart(df_kz).mark_circle(size=100).encode(
                    x=alt.X('depth:Q', title='Ошақ тереңдігі (км)'),
                    y=alt.Y('mag:Q', title='Магнитудасы'),
                    color=alt.Color('mag:Q', scale=alt.Scale(scheme='orangered'), title='Күші'),
                    tooltip=[
                        alt.Tooltip('place:N', title='Нақты аймақ/Ел'),
                        alt.Tooltip('mag:Q', title='Магнитудасы'),
                        alt.Tooltip('depth:Q', title='Тереңдігі (км)')
                    ]
                ).properties(height=400).interactive()
                st.altair_chart(chart2, use_container_width=True)

        with tab3:
            st.markdown("### 🗂 Өңделген деректер кестесі")
            df_kz_kz = df_kz[["time", "place", "latitude", "longitude", "depth", "mag", "danger"]].copy()
            df_kz_kz.columns = [
                "Болған уақыты", 
                "Болған аймағы / Орны",
                "Географиялық ендік (Latitude)", 
                "Географиялық бойлық (Longitude)", 
                "Ошақ тереңдігі (км)", 
                "Магнитудасы (Күші)", 
                "Қауіптілік коды (0-Төмен, 1-Орта, 2-Жоғары)"
            ]
            st.dataframe(df_kz_kz, use_container_width=True)

except Exception as e:
    st.error("🌐 Интернет байланысын тексеріп, бетті қайта жаңартыңыз (F5).")
