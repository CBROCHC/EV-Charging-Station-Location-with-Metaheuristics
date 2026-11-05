"""
experimentos.py
===============

Lanzador de experimentos (apartados 2 y 3 del plan de evaluación, sobre las
instancias del apartado 1). Cada ejecución del AG se guarda en su propio
JSON: si el trabajo se corta (límite de 24 h, cancelación...) basta con
relanzar y SOLO se hacen las que faltan.

Fases (15 semillas del AG por configuración)
--------------------------------------------
  A  Un factor del AG cada vez (OFAT) sobre la instancia de referencia:
     configuración por defecto + cada nivel de cada factor (23 configuraciones).
  B  Configuración final del AG (config_final.json) en TODAS las instancias
     (escala y diversidad): convergencia, tiempo y calidad frente a |V|.
  C  Los factores clave del AG (FACTORES_FASE_C) en otras instancias: ¿se
     transfieren las conclusiones de A?
  D  Sensibilidad a los parámetros del PROBLEMA (presupuesto, demanda, radios,
     focos de demanda, nº de subestaciones) en la instancia de referencia, con la
     configuración final del AG: impacto en la composición de las soluciones.

Los grafos reales se descargan de OpenStreetMap (osmnx) al empezar, UNA vez por
instancia en el proceso principal; los workers heredan la instancia ya construida.

Ejemplos
--------
  python experimentos.py --listar
  python experimentos.py --fase A --workers 16
  python experimentos.py --fase C --instancias R-pueblo R-grid --workers 16
  python experimentos.py --fase A --evals 600 --semillas 2 --instancia-ref S-100   # prueba rápida

Parada: criterio "evaluaciones" (soluciones generadas), mismo presupuesto para
todas las configuraciones de una fase.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # un hilo por proceso: el paralelismo son los procesos

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback

import numpy as np

import config_experimentos as cfgx
from instancias import INSTANCIAS, ORDEN, construir_instancia
from algoritmo_genetico import ejecutar_algoritmo_genetico, ConfiguracionAG
from codificacion import decodificar_individuo, coste_total, uso_por_subestacion, es_factible


def _clave_problema(problema):
    """Overrides del problema como tupla ordenada (hashable, para la caché de instancias)."""
    return tuple(sorted(problema.items()))


# ----------------------------------------------------------------------
def crear_config(params, n_cand, evals, semilla):
    """Traduce los parámetros del experimento a ConfiguracionAG."""
    return ConfiguracionAG(
        tam_poblacion=params["tam_poblacion"],
        criterio_parada="evaluaciones",
        max_evaluaciones=evals,
        num_semillas_greedy=min(params["num_semillas_greedy"], params["tam_poblacion"]),
        tipo_cruce=params["tipo_cruce"],
        prob_cruce=params["prob_cruce"],
        prob_mutacion_gen=params["factor_mut_gen"] / n_cand,
        tam_torneo=params["tam_torneo"],
        num_elite=params["num_elite"],
        prob_mutacion_intercambio=params["prob_mutacion_intercambio"],
        max_intercambios=params["max_intercambios"],
        usar_cache_fitness=True,
        verbose=False,
        semilla=semilla,
    )


def _guardar_json(ruta, datos):
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f)
    os.replace(tmp, ruta)   # escritura atómica: nunca queda un JSON a medias


def _evals_hasta(hist_mejor, hist_ev, hist_t, fraccion):
    """Soluciones generadas y tiempo hasta alcanzar `fraccion` del fitness final."""
    objetivo = fraccion * hist_mejor[-1]
    for i, v in enumerate(hist_mejor):
        if v >= objetivo - 1e-9:
            return hist_ev[i], hist_t[i]
    return hist_ev[-1], hist_t[-1]


# ----------------------------------------------------------------------
def ejecutar_tarea(t):
    """Ejecuta UNA repetición del AG y guarda su JSON. Devuelve un resumen."""
    ruta = t["ruta"]
    if os.path.exists(ruta):
        return f"[omitida] {t['id']}"
    try:
        P, D, info = construir_instancia(t["instancia"], _clave_problema(t["problema"]))
        config = crear_config(t["params"], len(P.candidatos), t["evals"], t["semilla"])
        r = ejecutar_algoritmo_genetico(P, D, config)

        x = decodificar_individuo(r.mejor_individuo, P)
        uso = uso_por_subestacion(x, P)
        usos = [uso[k] / P.capacidad_subestacion[k] for k in uso if P.capacidad_subestacion[k] > 0]
        ev95, t95 = _evals_hasta(r.historial_mejor, r.historial_evaluaciones, r.historial_tiempo, 0.95)
        ev99, t99 = _evals_hasta(r.historial_mejor, r.historial_evaluaciones, r.historial_tiempo, 0.99)
        datos = dict(
            id=t["id"], fase=t["fase"], instancia=t["instancia"], factor=t["factor"],
            nivel=str(t["nivel"]), semilla=t["semilla"], params=t["params"], problema=t["problema"],
            evals=t["evals"], info=info,
            metricas=dict(
                fitness=float(r.mejor_fitness),
                pct_demanda=100.0 * r.mejor_fitness / info["demanda_total"],
                generaciones=int(r.generaciones_ejecutadas),
                soluciones_generadas=int(r.historial_evaluaciones[-1]),
                evaluaciones_reales=int(r.historial_evaluaciones_reales[-1]),
                tiempo=float(r.historial_tiempo[-1]),
                evals_95=int(ev95), evals_99=int(ev99), tiempo_95=float(t95), tiempo_99=float(t99),
                hamming_final=float(r.historial_hamming[-1]),
                distintos_final=int(r.historial_genotipos_distintos[-1]),
                n_lentas=int(np.sum(r.mejor_individuo == 1)),
                n_rapidas=int(np.sum(r.mejor_individuo == 2)),
                pct_presupuesto=100.0 * coste_total(x, P) / P.presupuesto,
                pct_subestaciones=100.0 * float(np.mean(usos)) if usos else 0.0,
                factible=bool(es_factible(x, P)),
            ),
            hist=dict(
                evaluaciones=r.historial_evaluaciones,
                mejor=[round(v, 4) for v in r.historial_mejor],
                medio=[round(v, 4) for v in r.historial_medio],
                hamming=[round(v, 5) for v in r.historial_hamming],
                distintos=r.historial_genotipos_distintos,
                tiempo=[round(v, 3) for v in r.historial_tiempo],
            ),
            mejor_individuo=[int(g) for g in r.mejor_individuo],
        )
        _guardar_json(ruta, datos)
        m = datos["metricas"]
        return (f"[ok] {t['id']}  fitness={m['fitness']:.1f} ({m['pct_demanda']:.1f} %)  "
                f"gens={m['generaciones']}  t={m['tiempo']:.0f}s")
    except Exception:
        with open("errores_experimentos.log", "a", encoding="utf-8") as f:
            f.write(f"--- {t['id']}\n{traceback.format_exc()}\n")
        return f"[ERROR] {t['id']} (ver errores_experimentos.log)"


# ----------------------------------------------------------------------
def _tarea(fase, inst, factor, nivel, params, semilla, evals, salida, problema=None):
    id_ = f"{fase}__{inst}__{factor}={nivel}__s{semilla}"
    carpeta = os.path.join(salida, fase)
    os.makedirs(carpeta, exist_ok=True)
    return dict(id=id_, ruta=os.path.join(carpeta, id_ + ".json"), fase=fase, instancia=inst,
                factor=factor, nivel=nivel, params=params, problema=problema or {},
                semilla=semilla, evals=evals)


def _configs_ofat(niveles, defecto):
    """Valor por defecto + un cambio de nivel cada vez (los factores de `niveles`)."""
    cfgs = [("defecto", "-", {})]
    for f in niveles:
        for n in cfgx.NIVELES.get(f, cfgx.NIVELES_PROBLEMA.get(f)):
            if n != defecto[f]:
                cfgs.append((f, n, {f: n}))
    return cfgs


def generar_tareas(args):
    semillas = cfgx.SEMILLAS_AG[: args.semillas]
    evals = args.evals or cfgx.EVALUACIONES_FASE[args.fase]
    tareas = []
    if args.fase == "A":
        for factor, nivel, cambio in _configs_ofat(cfgx.NIVELES, cfgx.DEFECTO):
            for s in semillas:
                tareas.append(_tarea("A", args.instancia_ref, factor, nivel,
                                     {**cfgx.DEFECTO, **cambio}, s, evals, args.salida))
    elif args.fase == "B":
        final = cfgx.configuracion_final()
        for inst in args.instancias or ORDEN:
            for s in semillas:
                tareas.append(_tarea("B", inst, "final", "-", final, s, evals, args.salida))
    elif args.fase == "C":
        for inst in args.instancias or cfgx.INSTANCIAS_FASE_C:
            for factor, nivel, cambio in _configs_ofat(cfgx.FACTORES_FASE_C, cfgx.DEFECTO):
                for s in semillas:
                    tareas.append(_tarea("C", inst, factor, nivel,
                                         {**cfgx.DEFECTO, **cambio}, s, evals, args.salida))
    else:  # D
        final = cfgx.configuracion_final()
        for factor, nivel, cambio in _configs_ofat(cfgx.NIVELES_PROBLEMA, cfgx.PROBLEMA_DEFECTO):
            for s in semillas:
                tareas.append(_tarea("D", args.instancia_ref, factor, nivel, final, s, evals,
                                     args.salida, problema=cambio))
    return tareas


def precontruir_instancias(pendientes):
    """Construye (y, si son reales, DESCARGA) cada instancia una vez en el proceso principal.
    Con `fork`, los workers heredan la caché: no se vuelve a descargar nada y todos
    trabajan sobre exactamente el mismo problema. Devuelve las instancias que fallan."""
    claves = sorted({(t["instancia"], _clave_problema(t["problema"])) for t in pendientes},
                    key=lambda k: ORDEN.index(k[0]))
    fallidas = set()
    for inst, ov in claves:
        t0 = time.time()
        try:
            _, _, info = construir_instancia(inst, ov)
            extra = f" {dict(ov)}" if ov else ""
            print(f"  instancia {inst}{extra}: |V|={info['n_nodos']} |E|={info['n_aristas']} "
                  f"|J|={info['n_candidatos']} K={info['n_subestaciones']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            fallidas.add(inst)
            print(f"  AVISO: no se pudo construir {inst}: {type(e).__name__}: {e}\n"
                  f"         (si es un grafo real, comprueba la salida a Internet del servidor y "
                  f"que osmnx está instalado)", flush=True)
    return fallidas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", choices=["A", "B", "C", "D"])
    ap.add_argument("--instancia-ref", default=cfgx.INSTANCIA_REFERENCIA,
                    help="instancia de las fases A y D (por defecto %(default)s)")
    ap.add_argument("--instancias", nargs="+", help="instancias de las fases B/C")
    ap.add_argument("--semillas", type=int, default=15, help="nº de semillas del AG (1-15, por defecto 15)")
    ap.add_argument("--evals", type=int, help="soluciones generadas por ejecución (por defecto, según la fase)")
    ap.add_argument("--workers", type=int, default=1, help="procesos en paralelo (= CPUs pedidas a slurm)")
    ap.add_argument("--shard", default="0/1", help="i/n: ejecuta solo la parte i de n (para varios trabajos)")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--listar", action="store_true", help="solo muestra instancias y nº de ejecuciones")
    args = ap.parse_args()

    if args.listar or not args.fase:
        print("Instancias:")
        for i in ORDEN:
            print(f"  {i:9s} {INSTANCIAS[i]['tipo']}")
        for fase in "ABCD":
            a = argparse.Namespace(**{**vars(args), "fase": fase})
            print(f"Fase {fase}: {len(generar_tareas(a))} ejecuciones")
        return
    if not 1 <= args.semillas <= len(cfgx.SEMILLAS_AG):
        sys.exit(f"--semillas debe estar entre 1 y {len(cfgx.SEMILLAS_AG)}")

    tareas = generar_tareas(args)
    i, n = map(int, args.shard.split("/"))
    tareas = tareas[i::n]
    pendientes = [t for t in tareas if not os.path.exists(t["ruta"])]
    print(f"Fase {args.fase}: {len(tareas)} ejecuciones en este shard, {len(pendientes)} pendientes, "
          f"{args.workers} workers", flush=True)
    if not pendientes:
        return

    print("Construyendo instancias ...", flush=True)
    fallidas = precontruir_instancias(pendientes)
    pendientes = [t for t in pendientes if t["instancia"] not in fallidas]
    if not pendientes:
        sys.exit("Ninguna instancia se pudo construir; nada que ejecutar.")
    if fallidas:
        print(f"Se omiten las ejecuciones de: {sorted(fallidas)}", flush=True)

    # Las más caras primero (instancias grandes / poblaciones grandes) para equilibrar la carga.
    pendientes.sort(key=lambda t: (-ORDEN.index(t["instancia"]), -t["params"]["tam_poblacion"]))
    t0 = time.time()
    if args.workers > 1:
        # fork: los workers heredan las instancias ya construidas (sin volver a descargar).
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(args.workers) as pool:
            for k, msg in enumerate(pool.imap_unordered(ejecutar_tarea, pendientes), 1):
                print(f"({k}/{len(pendientes)}) {msg}", flush=True)
    else:
        for k, t in enumerate(pendientes, 1):
            print(f"({k}/{len(pendientes)}) {ejecutar_tarea(t)}", flush=True)
    print(f"Terminado en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
