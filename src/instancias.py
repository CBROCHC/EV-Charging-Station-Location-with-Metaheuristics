"""
instancias.py
=============

Casos de prueba del apartado 1 (escala y diversidad).

  Serie de ESCALA     : S-100, S-250, S-500, S-1000 (sintéticos, misma densidad de
                        nodos por km²: al crecer |V| "crece la ciudad") y R-centro / R-ciudad.
  Serie de DIVERSIDAD : R-pueblo, R-grid y R-centro (|V| parecido, topología distinta).

Los grafos REALES se generan en cada ejecución descargándolos de OpenStreetMap con
osmnx (grafo_real.cargar_grafo_real), exactamente como en main.py. Por eso:
  * el servidor necesita salida a Internet y `pip install osmnx`;
  * osmnx guarda las respuestas en una caché (carpeta cache_osmnx), de modo que
    las siguientes ejecuciones desde el mismo directorio reutilizan los mismos datos;
  * el grafo se ordena de forma canónica (nodos y aristas por id) para que la instancia
    no dependa del orden en que OSM devuelva los elementos.
"""

import math
import os
from functools import lru_cache

import networkx as nx
import numpy as np

from problema import generar_grafo_sintetico, generar_parametros_problema
from distancias import calcular_distancias
from codificacion import generar_individuo_greedy
from fitness import evaluar_individuo

SEMILLA_INSTANCIA = 7
FRACCION_CANDIDATOS = 0.35     # |J| ≈ 35 % de |V| (si la instancia no fija n_cand)
RADIO_LENTA_BASE, RADIO_RAPIDA_BASE = 120.0, 220.0

INSTANCIAS = {
    # --- sintéticos: n_cand y n_sub explícitos
    "S-100":  dict(tipo="sintetico", n_nodos=100,  n_cand=35,  n_sub=3),
    "S-250":  dict(tipo="sintetico", n_nodos=250,  n_cand=90,  n_sub=5),
    "S-500":  dict(tipo="sintetico", n_nodos=500,  n_cand=175, n_sub=8),
    "S-1000": dict(tipo="sintetico", n_nodos=1000, n_cand=350, n_sub=12),
    # --- reales (punto = (lat, lon); editar aquí si quieres otras zonas)
    "R-pueblo": dict(tipo="real", punto=(40.1408, -3.4225), distancia_m=600),    # Chinchón (Madrid)
    "R-grid":   dict(tipo="real", punto=(41.3917, 2.1649),  distancia_m=800),    # Eixample (Barcelona)
    "R-centro": dict(tipo="real", punto=(40.4167, -3.7037), distancia_m=1000,    # Madrid, Sol
                     n_cand=250, n_sub=10),
    "R-ciudad": dict(tipo="real", punto=(39.4699, -0.3763), distancia_m=2000),   # Valencia centro
}
ORDEN = list(INSTANCIAS)  # de menor a mayor escala aproximada


def _k_por_regla(n_cand):
    return max(3, round(0.7 * math.sqrt(n_cand)))


def _canonico(G):
    """Copia del grafo con nodos y aristas añadidos en orden de id (determinista)."""
    H = nx.Graph()
    H.graph.update(G.graph)
    for n in sorted(G.nodes()):
        H.add_node(n, **G.nodes[n])
    for u, v in sorted(tuple(sorted(e)) for e in G.edges()):
        H.add_edge(u, v, **G.edges[u, v])
    return H


@lru_cache(maxsize=None)
def obtener_grafo(id_):
    """Genera (sintético) o descarga (real) el grafo de la instancia. Cacheado por proceso."""
    spec = INSTANCIAS[id_]
    if spec["tipo"] == "sintetico":
        n = spec["n_nodos"]
        # Grado medio constante y densidad de nodos constante (la "ciudad" crece con n).
        return generar_grafo_sintetico(
            n_nodos=n,
            radio_conexion=math.sqrt(9.3 / (math.pi * n)),
            escala_metros=2000.0 * math.sqrt(n / 700.0),
            semilla=SEMILLA_INSTANCIA,
        )
    try:  # caché de osmnx en una carpeta fija (si la versión de osmnx lo permite)
        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.cache_folder = os.environ.get("OSMNX_CACHE", "cache_osmnx")
    except Exception:
        pass
    from grafo_real import cargar_grafo_real
    return _canonico(cargar_grafo_real(punto=spec["punto"], distancia_metros=spec["distancia_m"]))


@lru_cache(maxsize=None)
def construir_instancia(id_, overrides=()):
    """Devuelve (problema, distancias, info).

    `overrides` es una tupla ordenada de pares (clave, valor) —hashable para la caché— con
    cambios sobre los parámetros del problema (fase D): fraccion_presupuesto,
    ratio_demanda_capacidad, fraccion_hotspots, escala_radios, factor_subestaciones.
    Determinista (SEMILLA_INSTANCIA). Cacheada por proceso."""
    ov = dict(overrides)
    spec = INSTANCIAS[id_]
    G = obtener_grafo(id_)
    if spec["tipo"] == "sintetico":
        n_cand, n_sub0 = spec["n_cand"], spec["n_sub"]
    else:
        n_cand = spec.get("n_cand") or max(10, round(FRACCION_CANDIDATOS * G.number_of_nodes()))
        n_sub0 = spec.get("n_sub") or _k_por_regla(n_cand)
    n_sub = max(2, round(n_sub0 * ov.get("factor_subestaciones", 1.0)))

    kwargs = {k: ov[k] for k in ("fraccion_presupuesto", "ratio_demanda_capacidad", "fraccion_hotspots")
              if k in ov}
    if "escala_radios" in ov:
        kwargs["radio_lenta"] = RADIO_LENTA_BASE * ov["escala_radios"]
        kwargs["radio_rapida"] = RADIO_RAPIDA_BASE * ov["escala_radios"]

    P = generar_parametros_problema(G, n_cand, n_sub, semilla=SEMILLA_INSTANCIA, **kwargs)
    D = calcular_distancias(P)

    # Referencia "greedy": 5 construcciones (con ruido) -> media y máximo.
    rng = np.random.default_rng(SEMILLA_INSTANCIA)
    fits = [evaluar_individuo(generar_individuo_greedy(P, rng), P, D, reparar=False)[0]
            for _ in range(5)]
    info = dict(
        id=id_, tipo=spec["tipo"], overrides=dict(ov),
        n_nodos=G.number_of_nodes(), n_aristas=G.number_of_edges(),
        n_candidatos=len(P.candidatos), n_subestaciones=len(P.capacidad_subestacion),
        presupuesto=float(P.presupuesto), demanda_total=float(sum(P.demanda.values())),
        greedy_medio=float(np.mean(fits)), greedy_max=float(np.max(fits)),
    )
    return P, D, info
