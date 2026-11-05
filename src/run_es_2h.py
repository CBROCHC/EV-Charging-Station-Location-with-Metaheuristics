"""
run_es_2h.py
=============

Enfriamiento simulado con parada por TIEMPO (por defecto 2 h). Con parada por
tiempo la temperatura baja de T0 a T_min en función de la fracción de tiempo
consumida (T0 -> T_min exactamente al acabar), así que `alpha` NO se usa.

Δf va en unidades absolutas de fitness (normalizacion="ninguna"), igual que en el
barrido de T0 de experimentos_es.py: pon aquí el T0 ganador del barrido.
La vecindad y las probabilidades de operadores se toman de CONFIG_ES (main_es.py).
La solución inicial es aleatoria (el AG no interviene: son dos ejecuciones independientes).

Guarda en salidas/comparacion_2h/es_2h.pkl (checkpoint cada 15 min y resultado final).

Uso:  python -u run_es_2h.py --T0 100 --T_min 0.01 --horas 2
"""

import argparse
import dataclasses
import os

import numpy as np

from comun_2h import construir_problema_2h, guardar_resultado, Checkpoint, CARPETA_SALIDA
from distancias import calcular_distancias
from main_es import CONFIG_ES
from algoritmo_es import ejecutar_enfriamiento_simulado
from codificacion import decodificar_individuo, coste_total, es_factible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T0", type=float, required=True, help="temperatura inicial (unidades de fitness)")
    ap.add_argument("--T_min", type=float, default=0.01)
    ap.add_argument("--esquema", default="geometrico", choices=["geometrico", "lineal", "inverso"])
    ap.add_argument("--inicial", default="aleatoria", choices=["aleatoria", "voraz"])
    ap.add_argument("--horas", type=float, default=2.0)
    ap.add_argument("--semilla", type=int, default=3)
    ap.add_argument("--checkpoint-min", type=float, default=15.0)
    ap.add_argument("--salida", default=os.path.join(CARPETA_SALIDA, "es_2h.pkl"))
    a = ap.parse_args()

    problema = construir_problema_2h()
    print("Problema:", problema.resumen(), flush=True)
    dist = calcular_distancias(problema)
    D = float(sum(problema.demanda.values()))

    cfg = dataclasses.replace(
        CONFIG_ES,
        T0=a.T0, T_min=a.T_min, esquema_enfriamiento=a.esquema,
        alpha=0.99, delta_lineal=None, beta=None,
        normalizacion="ninguna", solucion_inicial=a.inicial,
        criterio_parada="tiempo", tiempo_max_segundos=a.horas * 3600.0, max_evaluaciones=None,
        recalentado=False,
        intervalo_historial=200, intervalo_seguimiento=20000,
        verbose=True, semilla=a.semilla,
    )
    meta = dict(algoritmo="ES", demanda_total=D, horas=a.horas, semilla=a.semilla,
                config=dataclasses.asdict(cfg))
    ck = Checkpoint(a.salida, meta, cada_s=a.checkpoint_min * 60.0)

    r = ejecutar_enfriamiento_simulado(problema, dist, cfg, callback=ck)

    datos = dict(meta)
    datos.update(
        tiempo=np.array(r.historial_tiempo), evaluaciones=np.array(r.historial_evaluaciones),
        evaluaciones_reales=np.array(r.historial_evaluaciones_reales),
        mejor=np.array(r.historial_mejor), actual=np.array(r.historial_actual),
        temperatura=np.array(r.historial_temperatura),
        mejor_individuo=r.mejor_individuo, mejor_fitness=float(r.mejor_fitness), completo=True,
    )
    guardar_resultado(a.salida, datos)

    x = decodificar_individuo(r.mejor_individuo, problema)
    print("\n--- Resultado ES ---")
    print(f"Demanda atendida: {r.mejor_fitness:.2f} / {D:.2f} ({100 * r.mejor_fitness / D:.2f} %)")
    print(f"Coste: {coste_total(x, problema):.1f} / {problema.presupuesto:.1f}  factible={es_factible(x, problema)}")
    print(f"Evaluaciones: {r.evaluaciones_totales}, T final: {r.T_final:.2e}, tiempo: {r.tiempo_total / 3600:.2f} h")
    print("Guardado en:", a.salida)


if __name__ == "__main__":
    main()
