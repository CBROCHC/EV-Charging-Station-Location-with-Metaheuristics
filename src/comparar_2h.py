"""
comparar_2h.py
===============

Lee ag_2h.pkl y es_2h.pkl (generados por run_ag_2h.py / run_es_2h.py) y dibuja:

  comparacion_2h_tiempo.png         mejor fitness (% de la demanda total) vs tiempo (h):
                                     vista completa + zoom
  comparacion_2h_evaluaciones.png   lo mismo vs nº acumulado de evaluaciones (eje log)
  resumen_2h.csv                    resumen final y valores en hitos de tiempo

Si algún pickle es un checkpoint (la ejecución no terminó), se dibuja igualmente y se
marca como "(parcial)".

Uso:  python comparar_2h.py [--carpeta salidas/comparacion_2h]
"""

import argparse
import csv
import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORES = {"AG": "#3B6FF2", "ES": "#D6432D"}
HITOS_H = [0.5, 1, 1.5, 2]


def _cargar(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


def _valor_en(d, horas):
    """Mejor fitness (%) que había al cabo de `horas` (None si no se llegó a ese tiempo)."""
    t = d["tiempo"]
    if horas * 3600.0 > t[-1] + 1e-9:
        return None
    idx = max(0, np.searchsorted(t, horas * 3600.0, side="right") - 1)
    return 100.0 * d["mejor"][idx] / d["demanda_total"]


def _etiqueta(d):
    parcial = "" if d.get("completo", True) else " (parcial)"
    return f"{d['algoritmo']}{parcial}: {100 * d['mejor_fitness'] / d['demanda_total']:.2f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=os.path.join("salidas", "comparacion_2h"))
    ap.add_argument("--ag", default=None)
    ap.add_argument("--es", default=None)
    a = ap.parse_args()
    datos = {
        "AG": _cargar(a.ag or os.path.join(a.carpeta, "ag_2h.pkl")),
        "ES": _cargar(a.es or os.path.join(a.carpeta, "es_2h.pkl")),
    }
    t_max_h = max(d["tiempo"][-1] for d in datos.values()) / 3600.0

    # --- Figura 1: vs tiempo (completa + zoom) ---
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    t0_zoom = 0.1 * t_max_h
    vals_zoom = []
    for nombre, d in datos.items():
        y = 100.0 * np.asarray(d["mejor"]) / d["demanda_total"]
        x = np.asarray(d["tiempo"]) / 3600.0
        for ax in axs:
            ax.plot(x, y, drawstyle="steps-post", color=COLORES[nombre], lw=1.6, label=_etiqueta(d))
        vals_zoom += [y[max(0, np.searchsorted(x, t0_zoom, side="right") - 1)], y[-1]]
    margen = 0.05 * (max(vals_zoom) - min(vals_zoom)) + 1e-3
    axs[1].set_xlim(t0_zoom, t_max_h * 1.01)
    axs[1].set_ylim(min(vals_zoom) - margen, max(vals_zoom) + margen)
    axs[0].set_title("Mejor demanda atendida vs tiempo de cómputo")
    axs[1].set_title("Zoom (a partir del 10 % del tiempo)")
    for ax in axs:
        ax.set_xlabel("Tiempo de cómputo (h)")
        ax.set_ylabel("Mejor demanda atendida (% del total)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.carpeta, "comparacion_2h_tiempo.png"), dpi=150)
    plt.close(fig)

    # --- Figura 2: vs evaluaciones (eje x log) ---
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for nombre, d in datos.items():
        ev = np.maximum(np.asarray(d["evaluaciones"], dtype=float), 1.0)
        ax.plot(ev, 100.0 * np.asarray(d["mejor"]) / d["demanda_total"],
                drawstyle="steps-post", color=COLORES[nombre], lw=1.6, label=_etiqueta(d))
    ax.set_xscale("log")
    ax.set_xlabel("Nº acumulado de evaluaciones (soluciones generadas, escala log)")
    ax.set_ylabel("Mejor demanda atendida (% del total)")
    ax.set_title("AG vs ES: fitness vs soluciones generadas")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.carpeta, "comparacion_2h_evaluaciones.png"), dpi=150)
    plt.close(fig)

    # --- Resumen CSV + consola ---
    filas = []
    for nombre, d in datos.items():
        f = dict(
            algoritmo=nombre, completo=bool(d.get("completo", True)),
            fitness=float(d["mejor_fitness"]), pct_demanda=100.0 * d["mejor_fitness"] / d["demanda_total"],
            evaluaciones=int(d["evaluaciones"][-1]),
            evaluaciones_reales=int(d["evaluaciones_reales"][-1]) if "evaluaciones_reales" in d else "",
            tiempo_h=float(d["tiempo"][-1] / 3600.0),
        )
        for h in HITOS_H:
            v = _valor_en(d, h)
            f[f"pct_{h}h"] = "" if v is None else round(v, 3)
        filas.append(f)
    ruta_csv = os.path.join(a.carpeta, "resumen_2h.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f)
    print("Figuras y CSV en:", a.carpeta)


if __name__ == "__main__":
    main()
