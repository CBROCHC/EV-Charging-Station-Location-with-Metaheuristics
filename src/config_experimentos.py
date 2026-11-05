"""
config_experimentos.py
======================

Todo lo que define el diseño experimental en un único sitio: semillas,
presupuesto de cómputo, configuración por defecto del AG, niveles de cada
factor del AG (apartado 2) y niveles de cada parámetro del PROBLEMA (apartado 3).
"""

import json
import os

# --- Semillas del AG: 15 repeticiones por configuración. El problema/instancia usa
# siempre SEMILLA_INSTANCIA (ver instancias.py), de modo que lo único que varía entre
# repeticiones es el AG.
SEMILLAS_AG = list(range(101, 116))

# --- Presupuesto de cómputo: nº de SOLUCIONES GENERADAS por ejecución
# (mismo presupuesto para todas las configuraciones de una fase -> comparación justa).
EVALUACIONES_FASE = {
    "A": 10_000,   # factores del AG, un factor cada vez, en la instancia de referencia
    "B": 20_000,   # configuración final en todas las instancias (escala y diversidad)
    "C": 10_000,   # factores clave del AG repetidos en otras instancias
    "D": 10_000,   # sensibilidad a los parámetros del problema (configuración final del AG)
}

# --- Configuración por defecto del AG (nivel base de cada factor).
DEFECTO = {
    "tam_poblacion": 100,
    "tam_torneo": 3,
    "prob_cruce": 0.9,
    "tipo_cruce": "subestacion",
    "factor_mut_gen": 1.0,              # prob_mutacion_gen = factor / |J|
    "prob_mutacion_intercambio": 0.7,
    "max_intercambios": 3,
    "num_elite": 4,
    "num_semillas_greedy": 2,
}

# --- Niveles de cada factor del AG (incluyen el valor por defecto).
NIVELES = {
    "tam_poblacion": [50, 100, 200],
    "tam_torneo": [2, 3, 5, 10],
    "prob_cruce": [0.6, 0.8, 0.9, 1.0],
    "tipo_cruce": ["uniforme", "subestacion"],
    "factor_mut_gen": [0.5, 1.0, 2.0, 4.0],
    "prob_mutacion_intercambio": [0.0, 0.3, 0.7, 1.0],
    "max_intercambios": [1, 3, 5],
    "num_elite": [1, 2, 4, 10],
    "num_semillas_greedy": [0, 2, 10],
}

# --- Subconjunto de factores del AG que se repite en otras instancias (fase C).
FACTORES_FASE_C = ["tam_poblacion", "tam_torneo", "prob_mutacion_intercambio", "tipo_cruce"]

# --- Parámetros del PROBLEMA (fase D = bloque 3 del plan). Valores por defecto = los de
# problema.generar_parametros_problema; "escala_radios" multiplica los radios 120/220 m y
# "factor_subestaciones" multiplica el nº de subestaciones K de la instancia.
PROBLEMA_DEFECTO = {
    "fraccion_presupuesto": 0.2,
    "ratio_demanda_capacidad": 0.8,
    "escala_radios": 1.0,
    "fraccion_hotspots": 0.03,
    "factor_subestaciones": 1.0,
}
NIVELES_PROBLEMA = {
    "fraccion_presupuesto": [0.1, 0.2, 0.3],
    "ratio_demanda_capacidad": [0.5, 0.8, 1.2],
    "escala_radios": [0.6, 1.0, 1.5],
    "fraccion_hotspots": [0.01, 0.03, 0.10],
    "factor_subestaciones": [0.3, 1.0, 2.0],
}

# --- Instancias por defecto de cada fase.
INSTANCIA_REFERENCIA = "R-centro"
INSTANCIAS_FASE_C = ["R-pueblo", "R-grid", "S-250"]


def configuracion_final():
    """Configuración del AG elegida tras la fase A. Si existe `config_final.json`
    (solo hace falta incluir los parámetros que cambian respecto a DEFECTO),
    se aplica sobre DEFECTO; si no, las fases B y D usan DEFECTO."""
    cfg = dict(DEFECTO)
    if os.path.exists("config_final.json"):
        with open("config_final.json", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg
