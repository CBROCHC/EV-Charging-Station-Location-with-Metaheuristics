"""
visualizacion_es.py
===================

Gráficas de ANÁLISIS del enfriamiento simulado. Todas reciben un
`ResultadoES` (algoritmo_es.py) y usan como eje X el nº acumulado de
evaluaciones (= soluciones generadas), igual que `fitness_vs_evaluaciones.png`
del AG, de modo que las figuras de ES y AG se pueden poner lado a lado.

Para dibujar la solución sobre el grafo o el mapa real se siguen usando
`dibujar_grafo_solucion` y `dibujar_mapa_real` de visualizacion.py (solo
necesitan el individuo), y `generar_explorador_generaciones` también vale
(sus textos hablan de "generaciones" pero cada fotograma es una muestra del
historial del ES).
"""

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problema import ProblemaCarga
from codificacion import decodificar_individuo, coste_total, uso_por_subestacion
from algoritmo_es import ResultadoES

COLOR_LENTA = "#F2A93B"
COLOR_RAPIDA = "#D6432D"


def _guardar(fig, ruta: str) -> str:
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def dibujar_convergencia_es(resultado: ResultadoES, ruta_salida: str = "convergencia_es.png") -> str:
    """Mejor fitness histórico y fitness de la solución ACTUAL (que oscila a
    alta temperatura porque se aceptan empeoramientos) vs evaluaciones."""
    ev = resultado.historial_evaluaciones
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev, resultado.historial_actual, color="#8FA8C4", linewidth=1, label="Solución actual")
    ax.plot(ev, resultado.historial_mejor, color=COLOR_RAPIDA, linewidth=2, label="Mejor histórico")
    ax.set_xlabel("Nº acumulado de evaluaciones de fitness (soluciones generadas)")
    ax.set_ylabel("Fitness (demanda atendida)")
    ax.set_title("Convergencia del enfriamiento simulado")
    ax.legend()
    ax.grid(alpha=0.3)
    return _guardar(fig, ruta_salida)


def dibujar_temperatura(resultado: ResultadoES, ruta_salida: str = "temperatura.png") -> str:
    ev = resultado.historial_evaluaciones
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev, resultado.historial_temperatura, color="#2E5E8C")
    ax.set_yscale("log")
    ax.set_xlabel("Nº acumulado de evaluaciones")
    ax.set_ylabel("Temperatura T (escala log)")
    ax.set_title("Esquema de enfriamiento")
    ax.grid(alpha=0.3, which="both")
    return _guardar(fig, ruta_salida)


def dibujar_tasa_aceptacion(resultado: ResultadoES, ruta_salida: str = "tasa_aceptacion.png") -> str:
    """Tasa de aceptación por ventana: global y solo de vecinos PEORES. Es la
    gráfica clave para calibrar T0 (al inicio la de peores debería ser alta y
    caer hacia 0 al final)."""
    ev = resultado.historial_evaluaciones
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev, resultado.historial_tasa_aceptacion, color="#8FA8C4", label="Todos los vecinos")
    ax.plot(ev, resultado.historial_tasa_aceptacion_peores, color=COLOR_RAPIDA, label="Solo vecinos peores")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Nº acumulado de evaluaciones")
    ax.set_ylabel("Tasa de aceptación (por ventana)")
    ax.set_title("Tasa de aceptación")
    ax.legend()
    ax.grid(alpha=0.3)
    return _guardar(fig, ruta_salida)


def dibujar_evolucion_estaciones_es(
    problema: ProblemaCarga, resultado: ResultadoES, ruta_salida: str = "evolucion_estaciones_es.png"
) -> str:
    ev = resultado.historial_evaluaciones
    n_len = np.array([int(np.sum(i == 1)) for i in resultado.historial_individuos])
    n_rap = np.array([int(np.sum(i == 2)) for i in resultado.historial_individuos])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev, n_len, label="Estaciones lentas", color=COLOR_LENTA)
    ax.plot(ev, n_rap, label="Estaciones rápidas", color=COLOR_RAPIDA)
    ax.plot(ev, n_len + n_rap, label="Total instaladas", color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Nº acumulado de evaluaciones")
    ax.set_ylabel("Nº de estaciones (mejor solución)")
    ax.set_title("Evolución de la composición de la solución")
    ax.legend()
    ax.grid(alpha=0.3)
    return _guardar(fig, ruta_salida)


def dibujar_uso_recursos_es(
    problema: ProblemaCarga, resultado: ResultadoES, ruta_salida: str = "uso_recursos_es.png"
) -> str:
    pct_pres, pct_sub = [], []
    for ind in resultado.historial_individuos:
        x = decodificar_individuo(ind, problema)
        pct_pres.append(100 * coste_total(x, problema) / problema.presupuesto)
        uso = uso_por_subestacion(x, problema)
        pcts = [100 * uso[k] / problema.capacidad_subestacion[k]
                for k in uso if problema.capacidad_subestacion[k] > 0]
        pct_sub.append(float(np.mean(pcts)) if pcts else 0.0)
    ev = resultado.historial_evaluaciones
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev, pct_pres, label="% presupuesto usado", color="#2E8B57")
    ax.plot(ev, pct_sub, label="% capacidad media de subestación usada", color="#B8860B")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Nº acumulado de evaluaciones")
    ax.set_ylabel("% usado")
    ax.set_title("Uso de recursos de la mejor solución")
    ax.legend()
    ax.grid(alpha=0.3)
    return _guardar(fig, ruta_salida)


def dibujar_tiempo_computo_es(resultado: ResultadoES, ruta_salida: str = "tiempo_computo_es.png") -> str:
    """Mejor fitness vs tiempo de cómputo acumulado (s)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(resultado.historial_tiempo, resultado.historial_mejor, color=COLOR_RAPIDA)
    ax.set_xlabel("Tiempo de cómputo acumulado (s)")
    ax.set_ylabel("Mejor fitness")
    ax.set_title("Fitness vs tiempo de cómputo")
    ax.grid(alpha=0.3)
    return _guardar(fig, ruta_salida)


def dibujar_fitness_actual_vs_temperatura(
    resultado: ResultadoES, ruta_salida: str = "fitness_vs_temperatura.png"
) -> str:
    """Fitness de la solución actual frente a T (eje invertido: se enfría hacia
    la derecha). Muestra a qué temperatura 'cristaliza' la búsqueda."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(resultado.historial_temperatura, resultado.historial_actual, color="#2E5E8C", linewidth=1)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Temperatura T (escala log, enfriando →)")
    ax.set_ylabel("Fitness de la solución actual")
    ax.set_title("Fitness vs temperatura")
    ax.grid(alpha=0.3, which="both")
    return _guardar(fig, ruta_salida)


def generar_todas_las_graficas_es(
    problema: ProblemaCarga, resultado: ResultadoES, carpeta_salida: str = "graficas_es"
) -> Dict[str, str]:
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta = lambda n: os.path.join(carpeta_salida, n)
    return {
        "convergencia": dibujar_convergencia_es(resultado, ruta("convergencia_es.png")),
        "temperatura": dibujar_temperatura(resultado, ruta("temperatura.png")),
        "tasa_aceptacion": dibujar_tasa_aceptacion(resultado, ruta("tasa_aceptacion.png")),
        "fitness_vs_temperatura": dibujar_fitness_actual_vs_temperatura(resultado, ruta("fitness_vs_temperatura.png")),
        "evolucion_estaciones": dibujar_evolucion_estaciones_es(problema, resultado, ruta("evolucion_estaciones_es.png")),
        "uso_recursos": dibujar_uso_recursos_es(problema, resultado, ruta("uso_recursos_es.png")),
        "tiempo_computo": dibujar_tiempo_computo_es(resultado, ruta("tiempo_computo_es.png")),
    }
