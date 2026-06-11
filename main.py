from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Definir rutas absolutas explícitas para Local (Windows) y Producción (Render/Linux)
RUTA_LOCAL = "componentes_binarios.pkl"
RUTA_RENDER = "/opt/render/project/src/api-ia-python/componentes_binarios.pkl"

# Elige automáticamente cuál ruta usar dependiendo de dónde se está ejecutando el código
if os.path.exists(RUTA_RENDER):
    componentes_path = RUTA_RENDER
else:
    componentes_path = RUTA_LOCAL

# Cargar el diccionario usando la ruta correcta
componentes = joblib.load(componentes_path)

# Extraer los elementos del diccionario
modelo = componentes['modelo_binario']
scaler = componentes['escalador_binario']
variables_lasso = componentes['variables_lasso_bin']

class ExoplanetInput(BaseModel):
    koi_fpflag_nt: int
    koi_fpflag_ss: int
    koi_fpflag_co: int
    koi_fpflag_ec: int
    koi_period: float
    koi_time0bk: float
    koi_time0bk_err1: float
    koi_time0bk_err2 : float
    koi_impact: float
    koi_impact_err1: float
    koi_impact_err2: float
    koi_duration : float
    koi_duration_err1: float
    koi_duration_err2: float
    koi_depth: float
    koi_depth_err1: float
    koi_depth_err2: float
    koi_prad: float
    koi_prad_err2: float
    koi_teq: float
    koi_insol:  float
    koi_insol_err2: float
    koi_model_snr: float
    koi_tce_plnt_num: float
    koi_steff: float
    koi_steff_err1: float
    koi_slogg: float
    koi_slogg_err2: float
    koi_srad_err1: float
    ra: float
    dec : float
    koi_kepmag: float

@app.post("/predict")
def predict_exoplanet(data: ExoplanetInput):
    df_input = pd.DataFrame([data.dict()])
    columnas_originales = scaler.feature_names_in_
    df_reindexed = df_input.reindex(columns=columnas_originales, fill_value=0)

    # Procesar con los componentes cargados
    X_escalado = scaler.transform(df_reindexed)
    df_escalado = pd.DataFrame(X_escalado, columns=columnas_originales)
    df_final_lasso = df_escalado[variables_lasso]
    
    if hasattr(modelo, 'feature_name_'):
        df_final_lasso = df_final_lasso[modelo.feature_name_]
        
    clase_predicha = float(modelo.predict(df_final_lasso)[0])
    probabilidades = modelo.predict_proba(df_final_lasso)[0]
    prob_porcentaje = float(probabilidades[1]) * 100
    
    return {
        "prediction": clase_predicha,
        "probability": round(prob_porcentaje, 2)
    }
