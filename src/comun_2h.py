"""
comun_2h.py
============

Piezas compartidas por run_ag_2h.py, run_es_2h.py, comparar_2h.py y
experimentos_es.py (barrido):

  1. construir_problema_2h(): construye la MISMA instancia que main.py (mismos
     parámetros: se leen de main.py), pero guardando/cargando el grafo real en
     `grafos/instancia_2h.graphml`. Así el AG y el ES (que corren como dos
     trabajos distintos del servidor) usan EXACTAMENTE el mismo grafo, y no
     dependen de que dos descargas independientes de OpenStreetMap coincidan.

     Para crear el fichero UNA vez (en tu PC o en el nodo de login, con Internet):
         python comun_2h.py

  2. Checkpoint: callback que va guardando en disco el progreso (por si el
     trabajo se corta antes de terminar las 2 h).

  3. guardar_resultado(): escritura atómica del pickle de resultados.
"""

import os
import pickle
import time

import numpy as np

import main as cfg_ag
from problema import generar_grafo_sintetico, generar_parametros_problema

RUTA_GRAFO = os.path.join("grafos", "instancia_2h.graphml")
CARPETA_SALIDA = os.path.join("salidas", "comparacion_2h")


def construir_problema_2h():
    """Misma instancia que main.py / main_es.py (parámetros leídos de main.py),
    con el grafo real cacheado en disco (ver docstring del módulo)."""
    c = cfg_ag
    comunes = dict(
        fraccion_presupuesto=c.FRACCION_PRESUPUESTO,
        fraccion_hotspots=c.FRACCION_HOTSPOTS,
        radio_lenta=c.RADIO_LENTA_M,
        radio_rapida=c.RADIO_RAPIDA_M,
        ratio_demanda_capacidad=c.RATIO_DEMANDA_CAPACIDAD,
        rango_cap_subestaciones=c.RANGO_CAP_SUBESTACIONES,
    )
    if c.USAR_GRAFO_REAL:
        from grafo_real import cargar_grafo_real, cargar_grafo_guardado, guardar_grafo_real
        if os.path.exists(RUTA_GRAFO):
            grafo = cargar_grafo_guardado(RUTA_GRAFO)
        else:
            print(f"[comun_2h] {RUTA_GRAFO} no existe: se descarga de OpenStreetMap y se guarda.")
            grafo = cargar_grafo_real(punto=c.GRAFO_REAL_PUNTO, distancia_metros=c.GRAFO_REAL_DISTANCIA_M)
            guardar_grafo_real(grafo, RUTA_GRAFO)
        return generar_parametros_problema(
            grafo,
            n_candidatos=c.GRAFO_REAL_N_CANDIDATOS,
            n_subestaciones=c.GRAFO_REAL_N_SUBESTACIONES,
            semilla=c.SEMILLA,
            **comunes,
        )
    grafo = generar_grafo_sintetico(
        n_nodos=c.SINTETICO_N_NODOS, escala_metros=c.SINTETICO_ESCALA_METROS, semilla=c.SEMILLA
    )
    return generar_parametros_problema(
        grafo,
        n_candidatos=c.SINTETICO_N_CANDIDATOS,
        n_subestaciones=c.SINTETICO_N_SUBESTACIONES,
        semilla=c.SEMILLA,
        **comunes,
    )


def guardar_resultado(ruta: str, datos: dict) -> None:
    """Escritura atómica: escribe a un temporal y lo renombra, para que un
    corte durante la escritura nunca deje un pickle a medias."""
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    tmp = ruta + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(datos, f)
    os.replace(tmp, ruta)


class Checkpoint:
    """Callback(evaluaciones, tiempo_s, mejor_fitness, mejor_individuo) que
    acumula la curva "mejor fitness" y, cada `cada_s` segundos, vuelca el
    progreso a `ruta` con completo=False. El script principal lo sobrescribe
    al terminar con completo=True y el historial completo."""

    def __init__(self, ruta: str, meta: dict, cada_s: float = 900.0):
        self.ruta, self.meta, self.cada_s = ruta, dict(meta), cada_s
        self.tiempo, self.evaluaciones, self.mejor = [], [], []
        self.mejor_individuo = None
        self._ultimo = time.time()

    def __call__(self, evaluaciones, tiempo_s, mejor_fitness, mejor_individuo):
        self.evaluaciones.append(int(evaluaciones))
        self.tiempo.append(float(tiempo_s))
        self.mejor.append(float(mejor_fitness))
        self.mejor_individuo = np.array(mejor_individuo, copy=True)
        if time.time() - self._ultimo >= self.cada_s:
            self.volcar()

    def volcar(self):
        d = dict(self.meta)
        d.update(
            tiempo=np.array(self.tiempo), evaluaciones=np.array(self.evaluaciones),
            mejor=np.array(self.mejor), mejor_individuo=self.mejor_individuo,
            mejor_fitness=self.mejor[-1] if self.mejor else float("nan"), completo=False,
        )
        guardar_resultado(self.ruta, d)
        self._ultimo = time.time()


if __name__ == "__main__":
    # Prepara (una vez) el grafo cacheado y muestra el resumen de la instancia.
    p = construir_problema_2h()
    print("Instancia:", p.resumen())
    print("Grafo guardado en:", RUTA_GRAFO)
