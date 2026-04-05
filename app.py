import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os 
import time
# Importamos la lógica de tu Predict.py
from Predict import predecir, features

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="IA Diagnóstico ICFES & Vocacional Pro", page_icon="🎓", layout="wide")

# Estilo personalizado profesional preservado v1.0
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-top: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; background-color: #1f77b4; color: white; height: 3.5em; font-weight: bold; border-radius: 10px; transition: 0.3s; }
    .stButton>button:hover { background-color: #155a8a; border-color: #155a8a; }
    .carrera-box { background-color: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #e0e0e0; text-align: center; font-size: 0.9em; font-weight: bold; color: #1f77b4; min-height: 60px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .perfil-header { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 8px solid; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE RUTAS DINÁMICAS (Soporte Nube/Local) ---
# Obtenemos la carpeta actual donde reside app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nombre_archivo = "programas_academicos.csv"
ruta_dinamica = os.path.join(BASE_DIR, nombre_archivo)

df_catalogo = None

def cargar_csv_seguro(ruta):
    try:
        # Intentamos latin-1 para compatibilidad con Excel/ANSI de Windows
        return pd.read_csv(ruta, encoding='latin-1')
    except Exception:
        try:
            # Si falla, intentamos utf-8 estándar de la nube
            return pd.read_csv(ruta, encoding='utf-8')
        except Exception as e:
            st.sidebar.error(f"Error al leer la base de datos: {e}")
            return None

# Lógica de detección de base de datos
if os.path.exists(ruta_dinamica):
    df_catalogo = cargar_csv_seguro(ruta_dinamica)
    if df_catalogo is not None:
        st.sidebar.success("✅ Base de Datos Conectada")
else:
    st.sidebar.warning("⚠️ Usando motor de respaldo (CSV no detectado)")

# --- 3. LÓGICA DE CÁLCULO ---

def aplicar_variabilidad(notas):
    return [max(0, min(100, n + np.random.normal(0, 1.8))) for n in notas]

def calcular_global(notas):
    # Fórmula oficial ICFES (Pesos: 3, 3, 3, 3, 1)
    puntaje = ((notas[4] * 3) + (notas[1] * 3) + (notas[2] * 3) + (notas[3] * 3) + (notas[0] * 1)) / 13 * 5
    return int(max(0, min(500, puntaje)))

def obtener_recomendaciones_realistas(notas_dict, global_score):
    ordenadas = sorted(notas_dict.items(), key=lambda x: x[1], reverse=True)
    fuerte_1, fuerte_2 = ordenadas[0][0], ordenadas[1][0]
    
    if global_score >= 360: rango = 'Top'
    elif global_score >= 280: rango = 'Medio'
    else: rango = 'Tec'

    sugerencias = []
    if df_catalogo is not None:
        try:
            filtro = (df_catalogo['Area_Principal'].isin([fuerte_1, fuerte_2])) & (df_catalogo['Nivel_Puntaje'] == rango)
            sugerencias = df_catalogo[filtro]['Carrera'].sample(frac=1).tolist()[:6]
        except: pass

    if not sugerencias:
        respaldo = {
            'Matemáticas': {'Top': ["Ingeniería IA", "Física"], 'Medio': ["Ing. Civil"], 'Tec': ["Tec. Software"]},
            'C. Naturales': {'Top': ["Medicina", "Ing. Biomédica"], 'Medio': ["Ambiental"], 'Tec': ["Regencia Farmacia"]},
            'Lectura Crítica': {'Top': ["Derecho", "Filosofía"], 'Medio': ["Psicología"], 'Tec': ["Gestión Doc."]},
            'Sociales': {'Top': ["Relaciones Int.", "Historia"], 'Medio': ["Lic. Sociales"], 'Tec': ["Gestión Com."]},
            'Inglés': {'Top': ["Negocios Int.", "Lenguas Modernas"], 'Medio': ["Hotelería"], 'Tec': ["Asistente Bilingüe"]}
        }
        sugerencias = list(set(respaldo[fuerte_1][rango] + respaldo[fuerte_2][rango]))

    perfiles = {
        'Matemáticas': ("Analítico - Exacto", "#1E88E5"),
        'C. Naturales': ("Científico - Investigador", "#00897B"),
        'Lectura Crítica': ("Humanista - Crítico", "#D32F2F"),
        'Sociales': ("Social - Ciudadano", "#7B1FA2"),
        'Inglés': ("Global - Comunicativo", "#FF8F00")
    }
    return perfiles[fuerte_1][0], perfiles[fuerte_1][1], sugerencias, fuerte_1, rango

# --- 4. INTERFAZ DE USUARIO ---
st.title("🎓 Consultor IA: Diagnóstico de Alto Rendimiento & Vocacional")
st.write("Predicción multivariable basada en Redes Neuronales y Big Data.")
st.divider()

col_input, col_output = st.columns([1, 1.6], gap="large")

with col_input:
    st.subheader("📝 Perfil Integral del Estudiante")
    
    st.write("**Entorno Familiar:**")
    edu_opciones = ["Ninguno", "Primaria", "Secundaria", "Técnica/Tecnológica", "Universitaria", "Postgrado"]
    edu_padre = st.selectbox("Nivel Educativo Padre:", edu_opciones)
    edu_madre = st.selectbox("Nivel Educativo Madre:", edu_opciones)
    libros = st.selectbox("Cantidad de Libros en Casa:", ["0 a 10", "11 a 25", "26 a 100", "Más de 100"])

    st.markdown("---")
    st.subheader("📊 Variables Socioeconómicas")
    estrato = st.select_slider("Estrato Socioeconómico:", options=[1, 2, 3, 4, 5, 6], value=2)
    genero = st.radio("Género:", ["Femenino", "Masculino"], horizontal=True)
    
    st.write("**Equipamiento del Hogar:**")
    c1, c2 = st.columns(2)
    with c1:
        tiene_pc = st.checkbox("Tiene Computador")
        tiene_lavadora = st.checkbox("Tiene Lavadora")
    with c2:
        tiene_internet = st.checkbox("Tiene Internet", value=False)
        tiene_auto = st.checkbox("Tiene Automóvil")

    st.markdown("---")
    st.write("**Institución y Ubicación:**")
    deptos = sorted([f.replace("ESTU_DEPTO_RESIDE_", "") for f in features if "ESTU_DEPTO_RESIDE_" in f])
    depto_sel = st.selectbox("Departamento de Residencia:", deptos)
    nat_cole = st.selectbox("Naturaleza del Colegio:", ["Oficial", "No Oficial (Privado)"])
    jornada = st.selectbox("Jornada Escolar:", ["Completa / Única", "Mañana", "Tarde", "Noche"])
    es_bilingue = st.checkbox("Colegio Bilingüe")

    if st.button("GENERAR DIAGNÓSTICO E INFORME VOCACIONAL"):
        try:
            input_data = {f: 0 for f in features}
            input_data.update({
                "EDAD": 17, "ANIO": 2023, "FAMI_ESTRATOVIV": estrato, "DENSIDADHOGAR": 1.0,
                "FAMI_TIENEINTERNET": 1 if tiene_internet else 0,
                "FAMI_TIENECOMPUTADOR": 1 if tiene_pc else 0,
                "FAMI_TIENEAUTOMOVIL": 1 if tiene_auto else 0,
                "FAMI_TIENELAVADORA": 1 if tiene_lavadora else 0,
                "COLE_BILINGUE": 1 if es_bilingue else 0
            })
            input_data["ESTU_GENERO_M"] = 1 if genero == "Masculino" else 0
            
            if f"FAMI_EDUCACIONPADRE_{edu_padre}" in input_data: input_data[f"FAMI_EDUCACIONPADRE_{edu_padre}"] = 1
            if f"FAMI_EDUCACIONMADRE_{edu_madre}" in input_data: input_data[f"FAMI_EDUCACIONMADRE_{edu_madre}"] = 1
            if f"ESTU_DEPTO_RESIDE_{depto_sel}" in input_data: input_data[f"ESTU_DEPTO_RESIDE_{depto_sel}"] = 1
            
            if nat_cole == "No Oficial (Privado)":
                input_data["COLE_CARACTER_TÉCNICO/ACADÉMICO"] = 1
                input_data["COLE_NATURALEZA_OFICIAL"] = 0
            else:
                input_data["COLE_NATURALEZA_OFICIAL"] = 1
            
            j_map = {"Completa / Única": "COLE_JORNADA_UNICA", "Mañana": "COLE_JORNADA_MAÑANA", "Tarde": "COLE_JORNADA_TARDE", "Noche": "COLE_JORNADA_NOCHE"}
            if j_map[jornada] in input_data: input_data[j_map[jornada]] = 1

            # PROCESO IA
            with st.spinner("La IA está calculando los resultados..."):
                notas_raw = predecir(input_data)
                notas_reales = aplicar_variabilidad(notas_raw)
                puntaje_global = calcular_global(notas_reales)
                
                materias = ['Inglés', 'Matemáticas', 'Sociales', 'C. Naturales', 'Lectura Crítica']
                notas_dict = dict(zip(materias, notas_reales))
                df_res = pd.DataFrame({'Materia': materias, 'Puntaje': notas_reales})

            with col_output:
                st.subheader(f"📊 Puntaje Global Estimado: {puntaje_global} / 500")
                p_nom, p_col, sugerencias, fuerte, nivel_r = obtener_recomendaciones_realistas(notas_dict, puntaje_global)
                
                st.markdown(f"<div class='perfil-header' style='background:{p_col}22; border-color:{p_col};'> <h3 style='margin:0; color:{p_col};'>Perfil {p_nom}</h3> <p style='margin:5px 0 0 0; color:#333;'>Nivel de proyección académica: <b>{nivel_r}</b></p> </div>", unsafe_allow_html=True)

                chart = alt.Chart(df_res).mark_bar(color=p_col).encode(
                    x=alt.X('Materia:N', sort=None, title='Materias evaluadas'),
                    y=alt.Y('Puntaje:Q', scale=alt.Scale(domain=[0, 100]), title='Puntaje (0-100)'),
                    tooltip=['Materia', 'Puntaje']
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)

                m_cols = st.columns(5)
                for i, m in enumerate(materias):
                    m_cols[i].metric(m, f"{int(notas_reales[i])}")

                st.write("---")
                st.markdown(f"#### 🎓 Abanico de Carreras Sugeridas ({nivel_r})")
                
                c_grid = st.columns(3)
                for i, carrera in enumerate(sugerencias):
                    with c_grid[i % 3]:
                        st.markdown(f"<div class='carrera-box'>{carrera}</div>", unsafe_allow_html=True)
                
                if puntaje_global > 380: st.balloons()
                
        except Exception as e:
            st.error(f"Error técnico durante la predicción: {e}")

# --- SIDEBAR (COMPLEMENTO TÉCNICO IA) ---
st.sidebar.title("🛠️ Panel Técnico de IA")
with st.sidebar.expander("Ver Arquitectura del Modelo"):
    st.write("**Tipo:** Red Neuronal Artificial (ANN)")
    st.write("**Entrenamiento:** Backpropagation")
    st.write("**Capas:** Entrada + 3 Ocultas + Salida")
    st.write("**Función Activación:** ReLU / Sigmoid")

st.sidebar.info("💡 **Dato de Fiabilidad:** El modelo analiza patrones socioeconómicos para ajustar los pesos de éxito académico.")
st.sidebar.caption("Proyecto ICFES v1.0 - 2026")
