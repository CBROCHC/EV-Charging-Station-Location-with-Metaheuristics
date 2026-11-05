"""
experimentos_es.py
==================

Experimentos para la sección "Evaluación" de la memoria. Cada experimento
repite varias semillas, guarda un CSV resumen y una gráfica en
`salidas/experimentos_es/`.

  1. barrido_T0_alpha      -> calibrado EXPERIMENTAL de T0 (y del ritmo de enfriamiento).
  2. comparar_esquemas     -> geométrico / lineal / inverso / escalonado (mismo T0->T_min).
  3. comparar_inicializaciones -> aleatoria vs voraz vs solución del AG.
  4. comparar_operadores   -> cada operador solo vs mezcla; y max_movimientos / k.
  5. comparar_ag_vs_es     -> AG vs ES, a IGUAL nº de evaluaciones y a IGUAL tiempo.

Qué experimentos se ejecutan, con qué presupuesto y cuántas semillas se
decide en el bloque PARÁMETROS de abajo. Con la instancia de main.py cada
evaluación cuesta ~0.03-0.08 s (flujo máximo), así que reduce
MAX_EVALUACIONES / SEMILLAS para pruebas rápidas.

Ejecutar con:  python3 experimentos_es.py
"""

import csv
import dataclasses
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main as cfg_ag
from main_es import CONFIG_ES
from comun_2h import construir_problema_2h as construir_problema   # instancia fija (grafo cacheado)
from distancias import calcular_distancias
from algoritmo_genetico import ejecutar_algoritmo_genetico
from algoritmo_es import ConfiguracionES, ejecutar_enfriamiento_simulado

# ============================================================================
# PARÁMETROS
# ============================================================================
# Se pueden pasar por línea de comandos: python experimentos_es.py barrido
EXPERIMENTOS = ["barrido"]        # opciones: "barrido", "esquemas", "inicializaciones", "operadores", "ag_vs_es"

MAX_EVALUACIONES = 10000          # presupuesto por ejecución
SEMILLAS = [1, 2, 3, 4, 5]        # repeticiones por configuración
TIEMPO_MAX_S = 300.0              # presupuesto de tiempo para ag_vs_es "igual tiempo"
AG_TIEMPO_INICIO_S = 60.0         # tiempo (s) que gasta el AG para dar la solución inicial
AG_VS_ES_MODOS = ["tiempo"]       # "tiempo" y/o "evaluaciones" para el experimento ag_vs_es

# --- Barrido de T0 x alpha (enfriamiento geométrico: T <- alpha * T en CADA evaluación, L=1) ---
# Δf va en UNIDADES ABSOLUTAS de fitness (normalizacion="ninguna"), por eso T0 = 10...10000.
# Equivalencia con la T0 normalizada de main_es.py:  T0_abs = T0_norm * demanda_total.
BARRIDO_T0 = [10, 50, 100, 200, 10000]
BARRIDO_ALPHA = [0.8, 0.9, 0.95, 0.99, 0.999]
BARRIDO_T_MIN = 0.01                    # suelo de temperatura (unidades de fitness)
BARRIDO_EVALUACIONES = 10000            # evaluaciones por ejecución
# Procesos en paralelo (cada par T0-alpha-semilla es una ejecución independiente). En SLURM se
# toma de --cpus-per-task; en local, de los núcleos de la máquina.
N_PROCESOS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

CARPETA = os.path.join("salidas", "experimentos_es")


# ============================================================================
# Utilidades
# ============================================================================
def _curva_escalon(x_hist, y_hist, rejilla):
    """Valor de una serie 'en escalón' (mejor-hasta-ahora) en cada punto de la rejilla."""
    idx = np.clip(np.searchsorted(x_hist, rejilla, side="right") - 1, 0, len(y_hist) - 1)
    return np.asarray(y_hist, dtype=float)[idx]


def _ejecutar_es(problema, dist, cfg: ConfiguracionES, semilla: int, inicial=None, previas=0):
    cfg = dataclasses.replace(cfg, semilla=semilla, verbose=False)
    return ejecutar_enfriamiento_simulado(problema, dist, cfg, solucion_inicial=inicial,
                                          evaluaciones_previas=previas)


def _ejecutar_ag(problema, dist, semilla: int, **kw):
    cfg = dataclasses.replace(cfg_ag.CONFIG_AG, semilla=semilla, verbose=False, **kw)
    return ejecutar_algoritmo_genetico(problema, dist, cfg)


def _resumen(nombre: str, finales: List[float], evals: List[int], tiempos: List[float]) -> dict:
    return dict(
        configuracion=nombre,
        media=float(np.mean(finales)), std=float(np.std(finales)),
        mejor=float(np.max(finales)), peor=float(np.min(finales)),
        evaluaciones=float(np.mean(evals)), tiempo_s=float(np.mean(tiempos)),
    )


def _guardar_csv(filas: List[dict], nombre: str) -> str:
    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, nombre)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return ruta


def _imprimir_tabla(filas: List[dict]):
    print(f"  {'configuración':<32}{'media':>10}{'std':>8}{'mejor':>10}{'peor':>10}{'evals':>9}{'t(s)':>8}")
    for r in filas:
        print(f"  {r['configuracion']:<32}{r['media']:>10.2f}{r['std']:>8.2f}"
              f"{r['mejor']:>10.2f}{r['peor']:>10.2f}{r['evaluaciones']:>9.0f}{r['tiempo_s']:>8.1f}")


def _dibujar_curvas(curvas: Dict[str, np.ndarray], rejilla, xlabel: str, titulo: str, ruta: str):
    """curvas[nombre] = matriz (n_semillas, len(rejilla)); se dibuja media ± std."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, M in curvas.items():
        m, s = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        ax.plot(rejilla, m, label=nombre)
        ax.fill_between(rejilla, m - s, m + s, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mejor fitness (media ± std sobre semillas)")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def comparativa_es(problema, dist, configs: Dict[str, ConfiguracionES], nombre: str, titulo: str):
    """Ejecuta cada configuración con todas las semillas; devuelve el resumen,
    y guarda CSV + curvas fitness-vs-evaluaciones."""
    rejilla = np.linspace(0, MAX_EVALUACIONES, 200)
    curvas, filas = {}, []
    for etiqueta, cfg in configs.items():
        finales, evals, tiempos, filas_curva = [], [], [], []
        for s in SEMILLAS:
            r = _ejecutar_es(problema, dist, cfg, s)
            finales.append(r.mejor_fitness); evals.append(r.evaluaciones_totales)
            tiempos.append(r.tiempo_total)
            filas_curva.append(_curva_escalon(r.historial_evaluaciones, r.historial_mejor, rejilla))
        curvas[etiqueta] = np.array(filas_curva)
        filas.append(_resumen(etiqueta, finales, evals, tiempos))
        print(f"  {etiqueta}: media={filas[-1]['media']:.2f} ± {filas[-1]['std']:.2f}")
    _guardar_csv(filas, f"{nombre}.csv")
    _dibujar_curvas(curvas, rejilla, "Nº acumulado de evaluaciones", titulo,
                    os.path.join(CARPETA, f"{nombre}.png"))
    print(f"\n[{nombre}]"); _imprimir_tabla(filas)
    return filas


def _base() -> ConfiguracionES:
    """CONFIG_ES de main_es.py como base común (criterio por evaluaciones)."""
    return dataclasses.replace(CONFIG_ES, criterio_parada="evaluaciones",
                               max_evaluaciones=MAX_EVALUACIONES, verbose=False,
                               solucion_inicial="aleatoria", recalentado=False)


# ============================================================================
# 1) Calibrado de T0 x alpha (en paralelo, con curvas de evolución del fitness)
# ============================================================================
_CTX = {}


def _init_worker(problema, dist):
    _CTX["problema"], _CTX["dist"] = problema, dist


def _media_sin_nan(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float("nan")


def _tarea_barrido(a_i: int, t_i: int, semilla: int, cfg: ConfiguracionES):
    """Una ejecución del ES (se lanza en un proceso hijo). Devuelve las curvas ya
    remuestreadas sobre una rejilla común de evaluaciones."""
    r = _ejecutar_es(_CTX["problema"], _CTX["dist"], cfg, semilla)
    rej = np.linspace(0, BARRIDO_EVALUACIONES, 200)
    n_ini = max(2, len(r.historial_tasa_aceptacion_peores) // 10)
    return a_i, t_i, semilla, dict(
        final=r.mejor_fitness, T_final=r.T_final, evals=r.evaluaciones_totales, tiempo=r.tiempo_total,
        mejor=_curva_escalon(r.historial_evaluaciones, r.historial_mejor, rej),
        actual=_curva_escalon(r.historial_evaluaciones, r.historial_actual, rej),
        temp=_curva_escalon(r.historial_evaluaciones, r.historial_temperatura, rej),
        acept_peores_ini=_media_sin_nan(r.historial_tasa_aceptacion_peores[1:n_ini]),
    )


def _ylim_zoom(medias: List[np.ndarray], rej: np.ndarray):
    """Rango y que ignora el arranque (primer 5 %) para que se vean las diferencias finas."""
    mask = rej >= 0.05 * rej[-1]
    lo = min(float(np.nanmin(m[mask])) for m in medias)
    hi = max(float(np.nanmax(m)) for m in medias)
    pad = 0.05 * (hi - lo) + 1e-9
    return lo - pad, hi + pad


def _graficar_barrido(datos: dict, carpeta: str = None):
    """Genera las figuras del barrido a partir del diccionario `datos` (el mismo
    que se guarda en barrido_curvas.pkl)."""
    carpeta = carpeta or CARPETA
    os.makedirs(carpeta, exist_ok=True)
    T0s, alphas, rej = datos["T0s"], datos["alphas"], np.asarray(datos["rejilla"])
    nA, nT = len(alphas), len(T0s)
    mejor = {k: np.asarray(v["mejor"]) for k, v in datos["curvas"].items()}
    actual = {k: np.asarray(v["actual"]) for k, v in datos["curvas"].items()}
    temp = {k: np.asarray(v["temp"]) for k, v in datos["curvas"].items()}
    media = {k: np.nanmean(M, axis=0) for k, M in mejor.items()}
    std = {k: np.nanstd(M, axis=0) for k, M in mejor.items()}
    finales = np.array([[media[(a, t)][-1] for t in range(nT)] for a in range(nA)])

    # (a) Mapa de calor: fitness medio final
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(finales, aspect="auto", cmap="viridis")
    ax.set_xticks(range(nT)); ax.set_xticklabels([f"{t:g}" for t in T0s])
    ax.set_yticks(range(nA)); ax.set_yticklabels([f"{a:g}" for a in alphas])
    for a in range(nA):
        for t in range(nT):
            ax.text(t, a, f"{finales[a, t]:.0f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("T0 (Δf absoluto)"); ax.set_ylabel("alpha")
    ax.set_title("Fitness medio final según T0 y alpha")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "barrido_T0_alpha.png"), dpi=150); plt.close(fig)

    # (b) Rejilla alpha x T0: una gráfica por prueba (mejor ± std, solución actual y temperatura)
    ylo, yhi = _ylim_zoom(list(media.values()) + [np.nanmean(M, axis=0) for M in actual.values()], rej)
    fig, axs = plt.subplots(nA, nT, figsize=(3.3 * nT, 2.7 * nA), sharex=True, sharey=True, squeeze=False)
    for a in range(nA):
        for t in range(nT):
            ax = axs[a][t]
            ax.plot(rej, np.nanmean(actual[(a, t)], axis=0), color="#8FA8C4", lw=0.8, label="Solución actual")
            ax.plot(rej, media[(a, t)], color="#D6432D", lw=1.6, label="Mejor (media)")
            ax.fill_between(rej, media[(a, t)] - std[(a, t)], media[(a, t)] + std[(a, t)],
                            color="#D6432D", alpha=0.15)
            ax2 = ax.twinx()
            ax2.plot(rej, np.nanmean(temp[(a, t)], axis=0), color="gray", ls="--", lw=0.8)
            ax2.set_yscale("log")
            ax2.set_ylim(BARRIDO_T_MIN * 0.5, max(T0s) * 2)
            ax2.tick_params(labelright=(t == nT - 1), labelsize=6)
            ax.set_title(f"α={alphas[a]:g}, T0={T0s[t]:g}  →  {media[(a, t)][-1]:.0f}", fontsize=8)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            if a == 0 and t == 0:
                ax.legend(fontsize=6, loc="lower right")
    ax_lo, ax_hi = ylo, yhi
    axs[0][0].set_ylim(ax_lo, ax_hi)
    for t in range(nT):
        axs[-1][t].set_xlabel("Evaluaciones", fontsize=8)
    for a in range(nA):
        axs[a][0].set_ylabel("Fitness", fontsize=8)
    fig.suptitle("Evolución del fitness por prueba (gris discontinuo: temperatura, escala log, eje derecho)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "barrido_rejilla.png"), dpi=130); plt.close(fig)

    # (c) Comparativas: por alpha (líneas = T0) y por T0 (líneas = alpha)
    cmap = plt.cm.viridis
    for nombre, n_pan, n_lin, etiq_pan, etiq_lin, valores_pan, valores_lin, clave in (
        ("barrido_por_alpha.png", nA, nT, "α", "T0", alphas, T0s, lambda p, l: (p, l)),
        ("barrido_por_T0.png", nT, nA, "T0", "α", T0s, alphas, lambda p, l: (l, p)),
    ):
        fig, axs = plt.subplots(1, n_pan, figsize=(3.6 * n_pan, 3.8), sharey=True, squeeze=False)
        for p in range(n_pan):
            ax = axs[0][p]
            for l in range(n_lin):
                k = clave(p, l)
                ax.plot(rej, media[k], color=cmap(l / max(1, n_lin - 1)), lw=1.4,
                        label=f"{etiq_lin}={valores_lin[l]:g}")
                ax.fill_between(rej, media[k] - std[k], media[k] + std[k],
                                color=cmap(l / max(1, n_lin - 1)), alpha=0.1)
            ax.set_title(f"{etiq_pan}={valores_pan[p]:g}", fontsize=9)
            ax.set_xlabel("Evaluaciones"); ax.grid(alpha=0.3)
            if p == 0:
                ax.set_ylabel("Mejor fitness (media ± std)")
                ax.legend(fontsize=7, loc="lower right")
        axs[0][0].set_ylim(*_ylim_zoom(list(media.values()), rej))
        fig.tight_layout(); fig.savefig(os.path.join(carpeta, nombre), dpi=150); plt.close(fig)


def regraficar_barrido():
    """Rehace las figuras desde barrido_curvas.pkl sin volver a ejecutar nada."""
    with open(os.path.join(CARPETA, "barrido_curvas.pkl"), "rb") as f:
        _graficar_barrido(pickle.load(f))


def barrido_T0_alpha(problema, dist):
    D = float(sum(problema.demanda.values()))
    base = dataclasses.replace(
        _base(), T0=max(BARRIDO_T0), normalizacion="ninguna", T_min=BARRIDO_T_MIN,
        esquema_enfriamiento="geometrico",
        max_evaluaciones=BARRIDO_EVALUACIONES, L=1,
    )
    tareas = []
    for a_i, alpha in enumerate(BARRIDO_ALPHA):
        for t_i, T0 in enumerate(BARRIDO_T0):
            cfg = dataclasses.replace(base, T0=T0, alpha=alpha)
            tareas += [(a_i, t_i, s, cfg) for s in SEMILLAS]
    print(f"  demanda total = {D:.1f}  (T0_normalizada = T0 / {D:.1f})")
    print(f"  {len(tareas)} ejecuciones ({len(BARRIDO_ALPHA)} alphas x {len(BARRIDO_T0)} T0 x {len(SEMILLAS)} semillas) "
          f"con {min(N_PROCESOS, len(tareas))} procesos")

    res = {}
    with ProcessPoolExecutor(max_workers=min(N_PROCESOS, len(tareas)),
                             initializer=_init_worker, initargs=(problema, dist)) as ex:
        futuros = [ex.submit(_tarea_barrido, *t) for t in tareas]
        for k, fut in enumerate(as_completed(futuros), 1):
            a_i, t_i, s, d = fut.result()
            res[(a_i, t_i, s)] = d
            print(f"  [{k}/{len(tareas)}] alpha={BARRIDO_ALPHA[a_i]:g} T0={BARRIDO_T0[t_i]:g} semilla={s}: "
                  f"{d['final']:.2f}  ({d['tiempo']:.0f} s)", flush=True)

    filas, curvas = [], {}
    for a_i, alpha in enumerate(BARRIDO_ALPHA):
        for t_i, T0 in enumerate(BARRIDO_T0):
            rs = [res[(a_i, t_i, s)] for s in SEMILLAS]
            finales = [r["final"] for r in rs]
            curvas[(a_i, t_i)] = {c: np.array([r[c] for r in rs]) for c in ("mejor", "actual", "temp")}
            filas.append(dict(
                T0=T0, alpha=alpha, media=float(np.mean(finales)), std=float(np.std(finales)),
                mejor=float(np.max(finales)), peor=float(np.min(finales)),
                pct_demanda=100 * float(np.mean(finales)) / D,
                T_final=float(np.mean([r["T_final"] for r in rs])),
                acept_peores_ini=_media_sin_nan([r["acept_peores_ini"] for r in rs]),
                tiempo_s=float(np.mean([r["tiempo"] for r in rs])),
            ))
            print(f"  alpha={alpha:g} T0={T0:g}: {filas[-1]['media']:.2f} ± {filas[-1]['std']:.2f}  "
                  f"T_final={filas[-1]['T_final']:.3g}  acept.peores(inicio)={filas[-1]['acept_peores_ini']:.2f}")
    _guardar_csv(filas, "barrido_T0_alpha.csv")

    datos = dict(T0s=BARRIDO_T0, alphas=BARRIDO_ALPHA, rejilla=np.linspace(0, BARRIDO_EVALUACIONES, 200),
                 curvas=curvas, demanda_total=D)
    with open(os.path.join(CARPETA, "barrido_curvas.pkl"), "wb") as f:
        pickle.dump(datos, f)
    _graficar_barrido(datos)
    return filas


# ============================================================================
# 2) Esquemas, 3) inicializaciones, 4) operadores
# ============================================================================
def comparar_esquemas(problema, dist):
    base = _base()
    configs = {e: dataclasses.replace(base, esquema_enfriamiento=e, alpha=None, delta_lineal=None, beta=None)
               for e in ("geometrico", "lineal", "inverso", "escalonado")}
    return comparativa_es(problema, dist, configs, "esquemas_enfriamiento",
                          "Esquemas de enfriamiento (mismo T0 -> T_min)")


def comparar_inicializaciones(problema, dist):
    """aleatoria / voraz: ES con todo el presupuesto. 'ag': el AG gasta
    AG_TIEMPO_INICIO_S segundos y el ES recibe el resto de evaluaciones; la curva incluye las
    evaluaciones del AG (coste honesto de la solución inicial)."""
    base = _base()
    rejilla = np.linspace(0, MAX_EVALUACIONES, 200)
    curvas, filas = {}, []
    for etiqueta in ("aleatoria", "voraz", "ag"):
        finales, evals, tiempos, filas_curva = [], [], [], []
        for s in SEMILLAS:
            if etiqueta == "ag":
                ra = _ejecutar_ag(problema, dist, s, criterio_parada="tiempo",
                                  tiempo_max_segundos=AG_TIEMPO_INICIO_S)
                prev = ra.historial_evaluaciones[-1]
                cfg = dataclasses.replace(base, solucion_inicial="ag", max_evaluaciones=MAX_EVALUACIONES - prev)
                r = _ejecutar_es(problema, dist, cfg, s, inicial=ra.mejor_individuo, previas=prev)
                ev = np.concatenate([ra.historial_evaluaciones, np.asarray(r.historial_evaluaciones) + prev])
                mej = np.concatenate([ra.historial_mejor, r.historial_mejor])
                tiempo = r.tiempo_total + ra.historial_tiempo[-1]
            else:
                r = _ejecutar_es(problema, dist, dataclasses.replace(base, solucion_inicial=etiqueta), s)
                ev, mej, tiempo = np.asarray(r.historial_evaluaciones), np.asarray(r.historial_mejor), r.tiempo_total
            mej = np.maximum.accumulate(mej)
            finales.append(r.mejor_fitness); evals.append(ev[-1]); tiempos.append(tiempo)
            filas_curva.append(_curva_escalon(ev, mej, rejilla))
        curvas[etiqueta] = np.array(filas_curva)
        filas.append(_resumen(f"inicial={etiqueta}", finales, evals, tiempos))
        print(f"  inicial={etiqueta}: media={filas[-1]['media']:.2f}")
    _guardar_csv(filas, "inicializaciones.csv")
    _dibujar_curvas(curvas, rejilla, "Nº acumulado de evaluaciones (incluye las del AG)",
                    "Solución inicial: aleatoria vs voraz vs AG", os.path.join(CARPETA, "inicializaciones.png"))
    print("\n[inicializaciones]"); _imprimir_tabla(filas)
    return filas


def comparar_operadores(problema, dist):
    base = _base()
    solo = lambda i: tuple(1.0 if k == i else 0.0 for k in range(4))
    configs = {
        "mezcla (por defecto)": base,
        "solo estado": dataclasses.replace(base, probs_operadores=solo(0)),
        "solo swap": dataclasses.replace(base, probs_operadores=solo(1)),
        "solo tipo": dataclasses.replace(base, probs_operadores=solo(2)),
        "solo desplazamiento": dataclasses.replace(base, probs_operadores=solo(3)),
        "sin desplazamiento": dataclasses.replace(base, probs_operadores=(0.3, 0.45, 0.25, 0.0)),
        "max_movimientos=3": dataclasses.replace(base, max_movimientos=3),
        "k_cercanos=2": dataclasses.replace(base, k_vecinos_cercanos=2),
        "k_cercanos=10": dataclasses.replace(base, k_vecinos_cercanos=10),
    }
    return comparativa_es(problema, dist, configs, "operadores", "Operadores de vecindad")


# ============================================================================
# 5) AG vs ES
# ============================================================================
def comparar_ag_vs_es(problema, dist):
    base = _base()
    out = {}
    for modo in AG_VS_ES_MODOS:
        if modo == "evaluaciones":
            rejilla, clave_x = np.linspace(0, MAX_EVALUACIONES, 200), "historial_evaluaciones"
            kw_ag = dict(criterio_parada="evaluaciones", max_evaluaciones=MAX_EVALUACIONES)
            cfg_es = base
            xlabel, titulo = "Nº acumulado de evaluaciones", "AG vs ES a igual nº de evaluaciones"
        else:
            rejilla, clave_x = np.linspace(0, TIEMPO_MAX_S, 200), "historial_tiempo"
            kw_ag = dict(criterio_parada="tiempo", tiempo_max_segundos=TIEMPO_MAX_S)
            # Con parada por tiempo el ES enfría en función del tiempo consumido (T0 -> T_min
            # exactamente en TIEMPO_MAX_S), así que no hay que calcular alpha a mano.
            esq = base.esquema_enfriamiento if base.esquema_enfriamiento != "escalonado" else "geometrico"
            cfg_es = dataclasses.replace(base, criterio_parada="tiempo", tiempo_max_segundos=TIEMPO_MAX_S,
                                         max_evaluaciones=None, esquema_enfriamiento=esq,
                                         alpha=None, delta_lineal=None, beta=None, recalentado=False)
            xlabel, titulo = "Tiempo de cómputo (s)", "AG vs ES a igual tiempo de cómputo"
        curvas = {"AG": [], "ES": []}
        finales = {"AG": [], "ES": []}; evals = {"AG": [], "ES": []}; tiempos = {"AG": [], "ES": []}
        for s in SEMILLAS:
            ra = _ejecutar_ag(problema, dist, s, **kw_ag)
            re_ = _ejecutar_es(problema, dist, cfg_es, s)
            for nombre, r in (("AG", ra), ("ES", re_)):
                curvas[nombre].append(_curva_escalon(getattr(r, clave_x), r.historial_mejor, rejilla))
                finales[nombre].append(r.mejor_fitness)
                evals[nombre].append(r.historial_evaluaciones[-1]); tiempos[nombre].append(r.historial_tiempo[-1])
        filas = [_resumen(n, finales[n], evals[n], tiempos[n]) for n in ("AG", "ES")]
        _guardar_csv(filas, f"ag_vs_es_igual_{modo}.csv")
        _dibujar_curvas({n: np.array(curvas[n]) for n in curvas}, rejilla, xlabel, titulo,
                        os.path.join(CARPETA, f"ag_vs_es_igual_{modo}.png"))
        print(f"\n[ag_vs_es, igual {modo}]"); _imprimir_tabla(filas)
        out[modo] = filas
    return out


# ============================================================================
def main():
    problema = construir_problema()
    print("Problema:", problema.resumen())
    dist = calcular_distancias(problema)
    os.makedirs(CARPETA, exist_ok=True)
    tabla = dict(barrido=barrido_T0_alpha, esquemas=comparar_esquemas,
                 inicializaciones=comparar_inicializaciones, operadores=comparar_operadores,
                 ag_vs_es=comparar_ag_vs_es)
    for nombre in (sys.argv[1:] or EXPERIMENTOS):
        print(f"\n===== Experimento: {nombre} =====")
        tabla[nombre](problema, dist)
    print(f"\nResultados en: {CARPETA}")


if __name__ == "__main__":
    main()
