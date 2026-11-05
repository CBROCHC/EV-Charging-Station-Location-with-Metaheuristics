"""
experimentos_instancias_es.py
=============================

Ejecuta el ES con la configuración final sobre las 8 instancias del estudio
del AG (4 sintéticas S-100/250/500/1000 y 4 reales R-pueblo/grid/centro/
ciudad) y genera, análogo a la Tabla 4 del AG:

  salidas/instancias_es/tabla_es.csv   resumen por instancia
  salidas/instancias_es/tabla_es.tex   filas LaTeX listas para pegar en la memoria
  salidas/instancias_es/convergencia_instancias_es.png   (análoga a la Fig. 5 del AG)

Comparación en IGUALDAD DE TIEMPO: a cada instancia se le da al ES
FACTOR_TIEMPO x el tiempo que tardó el AG en esa instancia (columna "Tiempo (s)"
de la Tabla 4 del AG; ver TIEMPO_AG_S). La temperatura se calcula a partir del
tiempo consumido (T0 -> T_min exactamente en ese tiempo).

IMPORTANTE - AJUSTAR A TUS INSTANCIAS DEL AG
--------------------------------------------
Las definiciones de INSTANCIAS siguen lo descrito en la memoria (|J| = 35 % de
|V|, presupuesto = 20 % del coste de abrir todos los candidatos como rápida,
radios 120/220 m), pero el nº de subestaciones K, la escala de los grafos
sintéticos, las coordenadas de las zonas reales y las semillas de generación NO
se pueden deducir de la memoria: edita el diccionario INSTANCIAS (y
SEMILLA_PROBLEMA) para que coincidan EXACTAMENTE con las que usaste en el AG.
Los grafos reales se descargan de OpenStreetMap (requiere Internet y osmnx) la
primera vez y se guardan en `grafos/<nombre>.graphml` para reutilizarlos.

Ejecutar con:  python3 experimentos_instancias_es.py
"""

import csv
import dataclasses
import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problema import generar_grafo_sintetico, generar_parametros_problema
from distancias import calcular_distancias
from codificacion import (
    decodificar_individuo, coste_total, uso_por_subestacion, generar_individuo_greedy,
)
from fitness import evaluar_individuo
from algoritmo_es import ConfiguracionES, ejecutar_enfriamiento_simulado

# ============================================================================
# PARÁMETROS
# ============================================================================
SEMILLAS = [1, 2, 3, 4, 5]          # semillas del ES por instancia
SEMILLA_PROBLEMA = 7                # semilla de generación de cada instancia (AJUSTAR)
FACTOR_TIEMPO = 1.0                 # tiempo del ES = FACTOR_TIEMPO x tiempo del AG
CARPETA_SALIDA = os.path.join("salidas", "instancias_es")
CARPETA_GRAFOS = "grafos"

# Configuración FINAL del ES (T0 elegida en la calibración; el resto, valores por defecto)
CONFIG_ES_FINAL = ConfiguracionES(
    T0=0.1, T_min=1e-5, esquema_enfriamiento="geometrico", L=1,
    criterio_parada="tiempo", max_evaluaciones=None,
    tiempo_max_segundos=1.0,                                # valor provisional: se fija por instancia
    probs_operadores=(0.25, 0.35, 0.15, 0.25), max_movimientos=1, k_vecinos_cercanos=5,
    solucion_inicial="aleatoria", recalentado=False, verbose=False,
)

# Tiempo (s) que tardó el AG en cada instancia (columna "Tiempo (s)" de la Tabla 4)
TIEMPO_AG_S = {
    "S-100": 45, "R-pueblo": 124, "S-250": 134, "R-grid": 126,
    "S-500": 378, "R-centro": 520, "S-1000": 852, "R-ciudad": 2382,
}

# --- Definición de las instancias (AJUSTAR: K, escalas, coordenadas) ---------
# Sintéticas: misma densidad de nodos y grado medio constante ->
#   escala_m = 2600*sqrt(n/500) y radio_conexion = 0.1*sqrt(500/n)  (S-500 = la de main.py)
def _sintetica(n, K):
    return dict(tipo="sintetico", n_nodos=n, K=K,
                escala_m=2600.0 * math.sqrt(n / 500.0),
                radio_conexion=0.1 * math.sqrt(500.0 / n))


INSTANCIAS: Dict[str, dict] = {
    "S-100": _sintetica(100, K=4),
    "S-250": _sintetica(250, K=6),
    "S-500": _sintetica(500, K=8),
    "S-1000": _sintetica(1000, K=14),
    "R-pueblo": dict(tipo="real", punto=(40.1403, -3.4217), dist_m=600, K=6),     # Chinchón
    "R-grid": dict(tipo="real", punto=(41.3917, 2.1649), dist_m=800, K=6),        # Eixample, Barcelona
    "R-centro": dict(tipo="real", punto=(40.4167, -3.7037), dist_m=1000, K=10),   # Madrid, Puerta del Sol
    "R-ciudad": dict(tipo="real", punto=(39.4699, -0.3763), dist_m=2000, K=20),   # Valencia centro
}
ORDEN = ["S-100", "R-pueblo", "S-250", "R-grid", "S-500", "R-centro", "S-1000", "R-ciudad"]


# ============================================================================
def construir_instancia(nombre: str):
    d = INSTANCIAS[nombre]
    if d["tipo"] == "sintetico":
        grafo = generar_grafo_sintetico(n_nodos=d["n_nodos"], radio_conexion=d["radio_conexion"],
                                        escala_metros=d["escala_m"], semilla=SEMILLA_PROBLEMA)
    else:
        from grafo_real import cargar_grafo_real, cargar_grafo_guardado, guardar_grafo_real
        ruta = os.path.join(CARPETA_GRAFOS, f"{nombre}.graphml")
        if os.path.exists(ruta):
            grafo = cargar_grafo_guardado(ruta)
        else:
            grafo = cargar_grafo_real(punto=d["punto"], distancia_metros=d["dist_m"])
            guardar_grafo_real(grafo, ruta)
    n_cand = int(round(0.35 * grafo.number_of_nodes()))          # |J| = 35 % de |V|
    return generar_parametros_problema(
        grafo, n_candidatos=n_cand, n_subestaciones=d["K"],
        fraccion_presupuesto=0.2, radio_lenta=120.0, radio_rapida=220.0,
        semilla=SEMILLA_PROBLEMA,
    )


def _pct_subestaciones(x, problema) -> float:
    uso = uso_por_subestacion(x, problema)
    pcts = [100 * uso[k] / problema.capacidad_subestacion[k]
            for k in uso if problema.capacidad_subestacion[k] > 0]
    return float(np.mean(pcts)) if pcts else 0.0


def ejecutar_instancia(nombre: str) -> dict:
    problema = construir_instancia(nombre)
    dist = calcular_distancias(problema)
    D = float(sum(problema.demanda.values()))
    tiempo = FACTOR_TIEMPO * TIEMPO_AG_S[nombre]
    print(f"\n=== {nombre}: {problema.resumen()}  | tiempo ES = {tiempo:.0f} s")

    # Greedy (referencia de la tabla del AG)
    ind_g = generar_individuo_greedy(problema, np.random.default_rng(SEMILLA_PROBLEMA))
    f_greedy, _ = evaluar_individuo(ind_g, problema, dist, reparar=False)

    fits, evals, tiempos, lentas, rapidas, presup, subest, curvas = [], [], [], [], [], [], [], []
    for s in SEMILLAS:
        cfg = dataclasses.replace(CONFIG_ES_FINAL, tiempo_max_segundos=tiempo, semilla=s)
        r = ejecutar_enfriamiento_simulado(problema, dist, cfg)
        x = decodificar_individuo(r.mejor_individuo, problema)
        fits.append(100 * r.mejor_fitness / D); evals.append(r.evaluaciones_totales)
        tiempos.append(r.tiempo_total)
        lentas.append(int((r.mejor_individuo == 1).sum())); rapidas.append(int((r.mejor_individuo == 2).sum()))
        presup.append(100 * coste_total(x, problema) / problema.presupuesto)
        subest.append(_pct_subestaciones(x, problema))
        curvas.append((np.asarray(r.historial_evaluaciones), 100 * np.asarray(r.historial_mejor) / D))
        print(f"   semilla {s}: {fits[-1]:.2f} %  ({r.evaluaciones_totales} eval, {r.tiempo_total:.0f} s)")

    return dict(
        instancia=nombre, V=problema.grafo.number_of_nodes(), J=len(problema.candidatos),
        K=len(problema.capacidad_subestacion), demanda_total=D,
        greedy_pct=100 * f_greedy / D, es_media=float(np.mean(fits)), es_std=float(np.std(fits)),
        es_mejor=float(np.max(fits)), evaluaciones=float(np.mean(evals)), tiempo_s=float(np.mean(tiempos)),
        lentas=float(np.mean(lentas)), rapidas=float(np.mean(rapidas)),
        presup_pct=float(np.mean(presup)), subest_pct=float(np.mean(subest)), curvas=curvas,
    )


def _grafica_convergencia(resultados: List[dict], ruta: str):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, prefijo, titulo in ((axs[0], "S-", "Instancias sintéticas"), (axs[1], "R-", "Instancias reales")):
        for r in resultados:
            if not r["instancia"].startswith(prefijo):
                continue
            x_max = max(c[0][-1] for c in r["curvas"])
            rej = np.linspace(0, x_max, 200)
            M = []
            for ev, mej in r["curvas"]:
                idx = np.clip(np.searchsorted(ev, rej, side="right") - 1, 0, len(mej) - 1)
                M.append(mej[idx])
            m, s = np.mean(M, axis=0), np.std(M, axis=0)
            ax.plot(rej, m, label=f"{r['instancia']} (|V|={r['V']})")
            ax.fill_between(rej, m - s, m + s, alpha=0.15)
        ax.set_title(titulo); ax.set_xlabel("Soluciones generadas"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axs[0].set_ylabel("Mejor demanda atendida (% del total)")
    fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    resultados = [ejecutar_instancia(n) for n in ORDEN]

    campos = ["instancia", "V", "J", "K", "demanda_total", "greedy_pct", "es_media", "es_std", "es_mejor",
              "evaluaciones", "tiempo_s", "lentas", "rapidas", "presup_pct", "subest_pct"]
    with open(os.path.join(CARPETA_SALIDA, "tabla_es.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader(); w.writerows(resultados)

    # Filas LaTeX con el formato de la Tabla 4 del AG (coma decimal)
    c = lambda v, n=1: f"{v:.{n}f}".replace(".", "{,}")
    with open(os.path.join(CARPETA_SALIDA, "tabla_es.tex"), "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(f"{r['instancia']} & {r['V']} & {c(r['greedy_pct'])} & ${c(r['es_media'])} \\pm {c(r['es_std'])}$ & "
                    f"{r['evaluaciones']:.0f} & {r['tiempo_s']:.0f} & {r['lentas']:.0f}/{r['rapidas']:.0f} & "
                    f"{r['presup_pct']:.0f} & {r['subest_pct']:.0f} \\\\\n")
    _grafica_convergencia(resultados, os.path.join(CARPETA_SALIDA, "convergencia_instancias_es.png"))

    print("\n" + "-" * 100)
    print(f"{'Instancia':<10}{'|V|':>6}{'Greedy%':>9}{'ES% (media±std)':>20}{'Evals':>9}{'t(s)':>7}{'L/R':>9}{'Pres%':>7}{'Sub%':>6}")
    for r in resultados:
        print(f"{r['instancia']:<10}{r['V']:>6}{r['greedy_pct']:>9.1f}{r['es_media']:>13.1f} ± {r['es_std']:<4.1f}"
              f"{r['evaluaciones']:>9.0f}{r['tiempo_s']:>7.0f}{r['lentas']:>5.0f}/{r['rapidas']:<3.0f}"
              f"{r['presup_pct']:>7.0f}{r['subest_pct']:>6.0f}")
    print(f"\nResultados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
