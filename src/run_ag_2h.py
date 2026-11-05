"""
run_ag_2h.py
=============

AG con población grande (por defecto 1000) y parada por TIEMPO (por defecto 2 h).
El resto de hiperparámetros se toman de CONFIG_AG de main.py.

Guarda en salidas/comparacion_2h/ag_2h.pkl (checkpoint cada 15 min y resultado
final) la curva "mejor fitness vs tiempo / evaluaciones", que luego lee
comparar_2h.py.

Nota: la parada por tiempo se comprueba al empezar cada generación, así que el AG
puede pasarse del límite lo que tarde UNA generación (con población 1000, unos minutos).

Uso:  python -u run_ag_2h.py --horas 2 --poblacion 1000
"""

import argparse
import dataclasses
import os

import numpy as np

import main as cfg_ag
from comun_2h import construir_problema_2h, guardar_resultado, Checkpoint, CARPETA_SALIDA
from distancias import calcular_distancias
from algoritmo_genetico import ejecutar_algoritmo_genetico
from codificacion import decodificar_individuo, coste_total, es_factible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, default=2.0)
    ap.add_argument("--poblacion", type=int, default=1000)
    ap.add_argument("--semilla", type=int, default=3)
    ap.add_argument("--checkpoint-min", type=float, default=15.0)
    ap.add_argument("--salida", default=os.path.join(CARPETA_SALIDA, "ag_2h.pkl"))
    a = ap.parse_args()

    problema = construir_problema_2h()
    print("Problema:", problema.resumen(), flush=True)
    dist = calcular_distancias(problema)
    D = float(sum(problema.demanda.values()))

    cfg = dataclasses.replace(
        cfg_ag.CONFIG_AG,
        tam_poblacion=a.poblacion, criterio_parada="tiempo",
        tiempo_max_segundos=a.horas * 3600.0, semilla=a.semilla,
        verbose=True, intervalo_seguimiento=5,
    )
    meta = dict(algoritmo="AG", demanda_total=D, horas=a.horas, semilla=a.semilla,
                config=dataclasses.asdict(cfg))
    ck = Checkpoint(a.salida, meta, cada_s=a.checkpoint_min * 60.0)

    r = ejecutar_algoritmo_genetico(problema, dist, cfg, callback=ck)

    datos = dict(meta)
    datos.update(
        tiempo=np.array(r.historial_tiempo), evaluaciones=np.array(r.historial_evaluaciones),
        evaluaciones_reales=np.array(r.historial_evaluaciones_reales),
        mejor=np.array(r.historial_mejor), mejor_individuo=r.mejor_individuo,
        mejor_fitness=float(r.mejor_fitness), generaciones=r.generaciones_ejecutadas, completo=True,
    )
    guardar_resultado(a.salida, datos)

    x = decodificar_individuo(r.mejor_individuo, problema)
    print("\n--- Resultado AG ---")
    print(f"Demanda atendida: {r.mejor_fitness:.2f} / {D:.2f} ({100 * r.mejor_fitness / D:.2f} %)")
    print(f"Coste: {coste_total(x, problema):.1f} / {problema.presupuesto:.1f}  factible={es_factible(x, problema)}")
    print(f"Generaciones: {r.generaciones_ejecutadas}, evaluaciones: {r.historial_evaluaciones[-1]}, "
          f"tiempo: {r.historial_tiempo[-1] / 3600:.2f} h")
    print("Guardado en:", a.salida)


if __name__ == "__main__":
    main()
