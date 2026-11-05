"""
algoritmo_es.py


Enfriamiento Simulado  para el problema de
localización de estaciones de carga. Reutiliza tal cual las piezas del AG:

  - Codificación A (vector de genes en {0,1,2}) y REPARACIÓN (codificacion.py).
  - Fitness = flujo máximo del subproblema de y (fitness.py)
  - Soluciones iniciales: aleatoria / voraz (codificacion.py) o la mejor
    solución de una ejecución del AG (se pasa como array).
  - Tablas vectorizadas de presupuesto/subestaciones (ContextoMutacion de
    algoritmo_genetico.py) para comprobar factibilidad de forma barata.

Esquema (maximización, ver Tema 3):

    S_actual = S_inicial; S_mejor = S_actual; T = T0
    mientras no se cumpla el criterio de parada:
        S_nuevo <- vecino aleatorio de S_actual (UN movimiento; reparado si
                   resulta infactible)
        Δf = (f(S_nuevo) - f(S_actual)) / f_ref          <- Δf NORMALIZADO
        si Δf >= 0: aceptar; si no: aceptar con probabilidad exp(Δf / T)
        cada L vecinos evaluados: T <- esquema_enfriamiento(T)

Convenciones:
  - Una ITERACIÓN = un vecino evaluado. Es el mismo "contador de soluciones generadas" del AG, por lo que
    las curvas fitness-vs-evaluaciones de ES y AG son directamente
    comparables. La solución inicial cuenta como 1 evaluación.
  - Δf se normaliza dividiendo por f_ref (por defecto la demanda total de la
    ciudad).

Vecindad (operadores de UN movimiento, todos reversibles):
  - "estado":        abrir una estación en un candidato cerrado (lenta/rápida
                     al azar) o cerrar una abierta.
  - "swap":          cerrar una estación abierta y abrir otra en un candidato
                     cerrado (tipo al azar).
  - "tipo":          lenta <-> rápida en una estación abierta.
  - "desplazamiento": mover una estación abierta (conservando su tipo) a uno
                     de los k candidatos MÁS PRÓXIMOS (por d_G) que esté
                     cerrado. La relación de proximidad se simetriza (b es
                     vecino de a si b está entre los k más próximos de a O
                     viceversa) para que el movimiento sea reversible.
Si el vecino resulta infactible (presupuesto o subestación) se REPARA con
`reparar_individuo`; si la reparación deshace el movimiento (devuelve la
solución actual), se reintenta con otro movimiento.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from problema import ProblemaCarga
from distancias import TablaDistancias
from codificacion import (
    Individuo,
    generar_individuo_aleatorio,
    generar_individuo_greedy,
    reparar_individuo,
)
from fitness import evaluar_individuo
from algoritmo_genetico import construir_contexto_mutacion, _es_factible_ctx

ESQUEMAS = ("geometrico", "lineal", "inverso", "escalonado")
CRITERIOS = ("evaluaciones", "tiempo", "estancamiento", "temperatura")
INICIALES = ("aleatoria", "voraz", "ag")
NOMBRES_OPERADORES = ("estado", "swap", "tipo", "desplazamiento")


# ======================================================================
# Configuración y resultado
# ======================================================================
@dataclass
class ConfiguracionES:
    """Hiperparámetros del enfriamiento simulado.

    Enfriamiento (se aplica cada `L` evaluaciones, es decir, en cada "paso de
    temperatura"; T nunca baja de `T_min`):
      "geometrico"  -> T <- alpha * T
      "lineal"      -> T <- T - delta_lineal
      "inverso"     -> T <- T / (1 + beta * T)
      "escalonado"  -> T <- alpha * T solo cada `pasos_por_escalon` pasos
    Si `alpha`, `delta_lineal` o `beta` son None y el criterio de parada es
    "evaluaciones" (presupuesto conocido), se calculan automáticamente para
    que T pase exactamente de T0 a T_min a lo largo del presupuesto (así los
    esquemas se comparan con el mismo recorrido de temperatura). Con otros
    criterios, alpha=None usa 0.99 y los otros dos exigen un valor explícito.

    Criterios de parada:
      "evaluaciones"  -> `max_evaluaciones` vecinos evaluados (comparable al AG).
      "tiempo"        -> `tiempo_max_segundos`. La temperatura se calcula en función de la
                         fracción de tiempo consumida (T0 -> T_min exactamente al acabar), por lo que
                         alpha/delta/beta no se usan. No admite 'escalonado' ni recalentado.
      "estancamiento" -> `paciencia_estancamiento` evaluaciones sin que el mejor
                         fitness mejore más de `tol_estancamiento`.
      "temperatura"   -> T <= T_min.

    solucion_inicial: "aleatoria" | "voraz" | "ag" (en este último caso la
    solución se pasa a ejecutar_enfriamiento_simulado(solucion_inicial=...)).
    """

    # --- Temperatura y enfriamiento ---
    T0: float = 0.01
    T_min: float = 1e-5
    esquema_enfriamiento: str = "geometrico"
    alpha: Optional[float] = None
    delta_lineal: Optional[float] = None
    beta: Optional[float] = None
    pasos_por_escalon: int = 10
    L: int = 1                                   # vecinos evaluados por temperatura
    normalizacion: str = "demanda_total"         # "demanda_total" | "fitness_inicial" | "ninguna"
                                                 # "ninguna": Δf en unidades absolutas de fitness (T0 ~ 10-200)

    # --- Solución inicial ---
    solucion_inicial: str = "aleatoria"

    # --- Vecindad ---
    # Probabilidades de elegir cada operador: (estado, swap, tipo, desplazamiento)
    probs_operadores: tuple = (0.25, 0.35, 0.15, 0.25)
    max_movimientos: int = 1                     # movimientos por vecino: uniforme en 1..max
    k_vecinos_cercanos: int = 5                  # k del operador de desplazamiento
    max_reintentos: int = 10                     # reintentos si el vecino es inválido/igual al actual

    # --- Parada ---
    criterio_parada: str = "evaluaciones"
    max_evaluaciones: Optional[int] = 20000
    tiempo_max_segundos: Optional[float] = None
    paciencia_estancamiento: int = 3000
    tol_estancamiento: float = 1e-3

    # --- Recalentado (enfriamiento no monótono, opcional) ---
    recalentado: bool = False
    paciencia_recalentado: int = 2000            # evaluaciones sin mejorar el mejor
    factor_recalentado: float = 0.5              # T <- max(T, factor * T0) y se vuelve al mejor

    # --- Varios ---
    usar_cache_fitness: bool = True
    intervalo_historial: int = 50                # cada cuántas evaluaciones se guarda una muestra
    intervalo_seguimiento: int = 2000            # cada cuántas evaluaciones se imprime por consola
    verbose: bool = True
    semilla: Optional[int] = None

    def __post_init__(self):
        if self.esquema_enfriamiento not in ESQUEMAS:
            raise ValueError(f"esquema_enfriamiento debe ser uno de {ESQUEMAS}")
        if self.criterio_parada not in CRITERIOS:
            raise ValueError(f"criterio_parada debe ser uno de {CRITERIOS}")
        if self.solucion_inicial not in INICIALES:
            raise ValueError(f"solucion_inicial debe ser una de {INICIALES}")
        if self.normalizacion not in ("demanda_total", "fitness_inicial", "ninguna"):
            raise ValueError("normalizacion debe ser 'demanda_total', 'fitness_inicial' o 'ninguna'")
        if self.criterio_parada == "evaluaciones" and not self.max_evaluaciones:
            raise ValueError("Con criterio_parada='evaluaciones' hay que indicar max_evaluaciones")
        if self.criterio_parada == "tiempo" and not self.tiempo_max_segundos:
            raise ValueError("Con criterio_parada='tiempo' hay que indicar tiempo_max_segundos")
        if self.criterio_parada == "tiempo":
            if self.esquema_enfriamiento == "escalonado":
                raise ValueError("esquema 'escalonado' no está soportado con parada por tiempo")
            if self.recalentado:
                raise ValueError("recalentado no está soportado con parada por tiempo")
        if not (self.T0 > self.T_min > 0):
            raise ValueError("Se exige T0 > T_min > 0")
        if self.L < 1 or self.max_movimientos < 1 or self.pasos_por_escalon < 1:
            raise ValueError("L, max_movimientos y pasos_por_escalon deben ser >= 1")
        p = np.asarray(self.probs_operadores, dtype=float)
        if len(p) != 4 or np.any(p < 0) or p.sum() <= 0:
            raise ValueError("probs_operadores: 4 valores >= 0 (estado, swap, tipo, desplazamiento)")


@dataclass
class ResultadoES:
    """Historial de una ejecución. Los `historial_*` son MUESTRAS tomadas cada
    `intervalo_historial` evaluaciones (más la inicial y la final); todas
    tienen la misma longitud. Los nombres `historial_individuos`,
    `historial_mejor`, `historial_evaluaciones`, `historial_tiempo` y
    `historial_evaluaciones_reales` coinciden con los de ResultadoAG, de modo
    que parte de visualizacion.py (p. ej. el explorador) también sirve."""

    mejor_individuo: Individuo
    mejor_fitness: float
    solucion_inicial: Individuo
    fitness_inicial: float

    historial_individuos: List[Individuo] = field(default_factory=list)   # mejor-hasta-ahora
    historial_mejor: List[float] = field(default_factory=list)
    historial_actual: List[float] = field(default_factory=list)           # fitness de S_actual
    historial_temperatura: List[float] = field(default_factory=list)
    historial_tasa_aceptacion: List[float] = field(default_factory=list)          # ventana
    historial_tasa_aceptacion_peores: List[float] = field(default_factory=list)   # ventana
    historial_evaluaciones: List[int] = field(default_factory=list)
    historial_evaluaciones_reales: List[int] = field(default_factory=list)  # fallos de caché
    historial_tiempo: List[float] = field(default_factory=list)

    evaluaciones_totales: int = 0
    evaluaciones_previas: int = 0      # coste de obtener la solución inicial (p. ej. del AG)
    pasos_temperatura: int = 0
    T_final: float = 0.0
    n_recalentados: int = 0
    motivo_parada: str = ""
    tiempo_total: float = 0.0



# Enfriamiento
@dataclass
class _ParamsEnfriamiento:
    esquema: str
    T_min: float
    alpha: float = 0.99
    delta: float = 0.0
    beta: float = 0.0
    pasos_por_escalon: int = 1


def _temperatura_por_tiempo(esquema: str, T0: float, T_min: float, p: float) -> float:
    """Temperatura en función de la FRACCIÓN de tiempo consumida p in [0,1] (parada por
    tiempo). Recorre T0 -> T_min exactamente en el tiempo máximo, sea cual sea la velocidad
    de la máquina:
      geométrico: T = T0 * (T_min/T0)^p        (equivale a alpha constante por unidad de tiempo)
      lineal:     T = T0 - (T0 - T_min) * p
      inverso:    1/T = 1/T0 + (1/T_min - 1/T0) * p
    """
    p = min(max(p, 0.0), 1.0)
    if esquema == "geometrico":
        return T0 * (T_min / T0) ** p
    if esquema == "lineal":
        return T0 - (T0 - T_min) * p
    return 1.0 / (1.0 / T0 + (1.0 / T_min - 1.0 / T0) * p)


def _construir_enfriamiento(cfg: ConfiguracionES) -> _ParamsEnfriamiento:
    n = None
    if cfg.criterio_parada == "evaluaciones":
        n = max(1, (cfg.max_evaluaciones - 1) // cfg.L)   # pasos de temperatura que caben
    p = _ParamsEnfriamiento(cfg.esquema_enfriamiento, cfg.T_min,
                            pasos_por_escalon=cfg.pasos_por_escalon)
    e = cfg.esquema_enfriamiento
    if cfg.criterio_parada == "tiempo":
        return p        # la temperatura se calcula con _temperatura_por_tiempo, sin parámetros
    if e in ("geometrico", "escalonado"):
        if cfg.alpha is not None:
            p.alpha = cfg.alpha
        elif n is not None:
            n_efectivo = n if e == "geometrico" else max(1, n // cfg.pasos_por_escalon)
            p.alpha = (cfg.T_min / cfg.T0) ** (1.0 / n_efectivo)
        else:
            p.alpha = 0.99
        if not (0 < p.alpha < 1):
            raise ValueError("alpha debe estar en (0,1)")
    elif e == "lineal":
        if cfg.delta_lineal is not None:
            p.delta = cfg.delta_lineal
        elif n is not None:
            p.delta = (cfg.T0 - cfg.T_min) / n
        else:
            raise ValueError("esquema 'lineal': indica delta_lineal (o usa criterio 'evaluaciones')")
    else:  # inverso: 1/T crece linealmente en beta por paso
        if cfg.beta is not None:
            p.beta = cfg.beta
        elif n is not None:
            p.beta = (1.0 / cfg.T_min - 1.0 / cfg.T0) / n
        else:
            raise ValueError("esquema 'inverso': indica beta (o usa criterio 'evaluaciones')")
    return p


def _siguiente_temperatura(T: float, paso: int, p: _ParamsEnfriamiento) -> float:
    """`paso` = nº de pasos de temperatura completados (1, 2, 3, ...)."""
    if p.esquema == "geometrico":
        T = T * p.alpha
    elif p.esquema == "lineal":
        T = T - p.delta
    elif p.esquema == "inverso":
        T = T / (1.0 + p.beta * T)
    elif p.esquema == "escalonado" and paso % p.pasos_por_escalon == 0:
        T = T * p.alpha
    return max(T, p.T_min)


# Vecindad
def construir_vecinos_cercanos(
    problema: ProblemaCarga, distancias: TablaDistancias, k: int
) -> List[np.ndarray]:
    """Para cada candidato (por índice de gen), array con los índices de sus
    candidatos vecinos para el operador de desplazamiento: los k más próximos
    por d_G (dentro del radio calculado en distancias.py), simetrizado."""
    cands = problema.candidatos
    n = len(cands)
    D = np.full((n, n), np.inf)
    for a, ja in enumerate(cands):
        dj = distancias.get(ja, {})
        for b, jb in enumerate(cands):
            if a != b and jb in dj:
                D[a, b] = dj[jb]
    conj = [set() for _ in range(n)]
    for a in range(n):
        orden = np.argsort(D[a], kind="stable")
        for b in orden[:k]:
            if np.isfinite(D[a, b]):
                conj[a].add(int(b))
                conj[int(b)].add(a)          # simetría -> movimiento reversible
    return [np.array(sorted(c), dtype=int) for c in conj]


def _op_estado(ind, vecinos, rng):
    cand = ind.copy()
    abiertos, cerrados = np.flatnonzero(ind > 0), np.flatnonzero(ind == 0)
    if len(cerrados) and (len(abiertos) == 0 or rng.random() < 0.5):
        cand[int(rng.choice(cerrados))] = int(rng.integers(1, 3))
    elif len(abiertos):
        cand[int(rng.choice(abiertos))] = 0
    return cand


def _op_swap(ind, vecinos, rng):
    cand = ind.copy()
    abiertos, cerrados = np.flatnonzero(ind > 0), np.flatnonzero(ind == 0)
    if len(abiertos) and len(cerrados):
        cand[int(rng.choice(abiertos))] = 0
        cand[int(rng.choice(cerrados))] = int(rng.integers(1, 3))
    return cand


def _op_tipo(ind, vecinos, rng):
    cand = ind.copy()
    abiertos = np.flatnonzero(ind > 0)
    if len(abiertos):
        a = int(rng.choice(abiertos))
        cand[a] = 3 - cand[a]                 # 1 <-> 2
    return cand


def _op_desplazamiento(ind, vecinos, rng):
    cand = ind.copy()
    abiertos = np.flatnonzero(ind > 0)
    if len(abiertos):
        a = int(rng.choice(abiertos))
        libres = vecinos[a][ind[vecinos[a]] == 0]
        if len(libres):
            b = int(rng.choice(libres))
            cand[b], cand[a] = cand[a], 0     # mismo tipo, nueva ubicación cercana
    return cand


_OPERADORES = (_op_estado, _op_swap, _op_tipo, _op_desplazamiento)


def generar_vecino(
    actual: Individuo,
    problema: ProblemaCarga,
    ctx,
    vecinos: List[np.ndarray],
    probs: np.ndarray,
    max_movimientos: int,
    max_reintentos: int,
    rng: np.random.Generator,
) -> Individuo:
    """Vecino aleatorio y FACTIBLE de `actual` (1..max_movimientos movimientos
    encadenados; si el resultado viola presupuesto/subestación se repara)."""
    cand = actual
    for _ in range(max_reintentos):
        cand = actual.copy()
        for _m in range(int(rng.integers(1, max_movimientos + 1))):
            op = _OPERADORES[int(rng.choice(4, p=probs))]
            cand = op(cand, vecinos, rng)
        if not _es_factible_ctx(cand, ctx):
            cand = reparar_individuo(cand, problema)
        if not np.array_equal(cand, actual):
            return cand
    return cand        # no se encontró vecino distinto: se devuelve el último (puede ser igual)


# Bucle principal

def ejecutar_enfriamiento_simulado(
    problema: ProblemaCarga,
    distancias: TablaDistancias,
    config: ConfiguracionES = None,
    solucion_inicial: Optional[Individuo] = None,
    evaluaciones_previas: int = 0,
    callback=None,
) -> ResultadoES:
    """Ejecuta el ES completo.

    solucion_inicial: array {0,1,2}. Obligatorio si config.solucion_inicial ==
        "ag"; si se pasa con otro valor de config, tiene prioridad.
    evaluaciones_previas: evaluaciones gastadas para obtener esa solución (p. ej.
        las del AG). Solo se guarda en el resultado para poder contabilizarlas
        al comparar; no consume el presupuesto del ES.
    callback: opcional. callback(evaluaciones, tiempo_s, mejor_fitness, mejor_individuo)
        se llama cada vez que se guarda una muestra del historial (cada
        `intervalo_historial` evaluaciones). Sirve para guardar checkpoints en
        ejecuciones largas (12 h); no altera el algoritmo.
    """
    cfg = config or ConfiguracionES()
    rng = np.random.default_rng(cfg.semilla)
    ctx = construir_contexto_mutacion(problema)
    vecinos = construir_vecinos_cercanos(problema, distancias, cfg.k_vecinos_cercanos)
    probs = np.asarray(cfg.probs_operadores, dtype=float)
    probs = probs / probs.sum()
    enfr = _construir_enfriamiento(cfg)
    cache = {} if cfg.usar_cache_fitness else None
    n_reales = [0]

    def evaluar(ind: Individuo) -> float:
        clave = ind.astype(np.int8).tobytes()
        if cache is not None and clave in cache:
            return cache[clave]
        fit, _ = evaluar_individuo(ind, problema, distancias, reparar=False)
        n_reales[0] += 1
        if cache is not None:
            cache[clave] = fit
        return fit

    t_ini = time.time()

    # --- Solución inicial ---
    if solucion_inicial is not None:
        actual = reparar_individuo(np.asarray(solucion_inicial, dtype=int), problema)
    elif cfg.solucion_inicial == "aleatoria":
        actual = generar_individuo_aleatorio(problema, rng=rng)
    elif cfg.solucion_inicial == "voraz":
        actual = generar_individuo_greedy(problema, rng)
    else:
        raise ValueError("solucion_inicial='ag' requiere pasar solucion_inicial=<individuo del AG>")
    f_actual = evaluar(actual)
    sol_ini, f_ini = actual.copy(), f_actual
    mejor, f_mejor = actual.copy(), f_actual

    if cfg.normalizacion == "demanda_total":
        f_ref = float(sum(problema.demanda.values()))
    elif cfg.normalizacion == "ninguna":
        f_ref = 1.0                   # Δf absoluto: T se mide en unidades de fitness (demanda)
    else:
        f_ref = f_actual
    f_ref = f_ref if f_ref > 0 else 1.0

    # --- Estado del bucle ---
    T, paso, ev_en_T = cfg.T0, 0, 0
    evals = 1
    eval_ultima_mejora = 1            # para estancamiento (mejora > tol)
    eval_ultima_mejora_rec = 1        # para recalentado (cualquier mejora del mejor)
    n_recal = 0
    ventana = dict(prop=0, acep=0, peores=0, acep_peores=0)
    res = ResultadoES(mejor, f_mejor, sol_ini, f_ini, evaluaciones_previas=evaluaciones_previas)

    def registrar():
        t_prop, t_peor = ventana["prop"], ventana["peores"]
        res.historial_individuos.append(mejor.copy())
        res.historial_mejor.append(f_mejor)
        res.historial_actual.append(f_actual)
        res.historial_temperatura.append(T)
        res.historial_tasa_aceptacion.append(ventana["acep"] / t_prop if t_prop else float("nan"))
        res.historial_tasa_aceptacion_peores.append(
            ventana["acep_peores"] / t_peor if t_peor else float("nan"))
        res.historial_evaluaciones.append(evals)
        res.historial_evaluaciones_reales.append(n_reales[0])
        res.historial_tiempo.append(time.time() - t_ini)
        for key in ventana:
            ventana[key] = 0
        if callback is not None:
            callback(evals, res.historial_tiempo[-1], f_mejor, mejor)

    registrar()
    if cfg.verbose:
        print(f"[Eval 1] inicial={f_ini:.2f}  T0={cfg.T0:g}  esquema={cfg.esquema_enfriamiento}")

    motivo = ""
    while True:
        # --- Criterios de parada ---
        if cfg.criterio_parada == "evaluaciones" and evals >= cfg.max_evaluaciones:
            motivo = "máximo de evaluaciones"; break
        if cfg.criterio_parada == "tiempo" and time.time() - t_ini >= cfg.tiempo_max_segundos:
            motivo = "tiempo agotado"; break
        if cfg.criterio_parada == "estancamiento" and evals - eval_ultima_mejora >= cfg.paciencia_estancamiento:
            motivo = "estancamiento"; break
        if cfg.criterio_parada == "temperatura" and T <= cfg.T_min:
            motivo = "temperatura mínima"; break

        # --- Vecino, evaluación y aceptación ---
        vecino = generar_vecino(actual, problema, ctx, vecinos, probs,
                                cfg.max_movimientos, cfg.max_reintentos, rng)
        f_vecino = evaluar(vecino)
        evals += 1
        delta = (f_vecino - f_actual) / f_ref

        ventana["prop"] += 1
        if delta >= 0:
            aceptar = True
        else:
            ventana["peores"] += 1
            aceptar = rng.random() < np.exp(delta / T)
            if aceptar:
                ventana["acep_peores"] += 1
        if aceptar:
            ventana["acep"] += 1
            actual, f_actual = vecino, f_vecino
            if f_actual > f_mejor + 1e-12:
                if f_actual - f_mejor > cfg.tol_estancamiento:
                    eval_ultima_mejora = evals
                mejor, f_mejor = actual.copy(), f_actual
                eval_ultima_mejora_rec = evals

        # --- Enfriamiento ---
        ev_en_T += 1
        if ev_en_T >= cfg.L:
            paso += 1
            ev_en_T = 0
            if cfg.criterio_parada == "tiempo":
                T = max(cfg.T_min, _temperatura_por_tiempo(
                    cfg.esquema_enfriamiento, cfg.T0, cfg.T_min,
                    (time.time() - t_ini) / cfg.tiempo_max_segundos))
            else:
                T = _siguiente_temperatura(T, paso, enfr)

        # --- Recalentado opcional ---
        if cfg.recalentado and evals - eval_ultima_mejora_rec >= cfg.paciencia_recalentado:
            T = max(T, cfg.factor_recalentado * cfg.T0)
            actual, f_actual = mejor.copy(), f_mejor
            eval_ultima_mejora_rec = evals
            n_recal += 1

        # --- Historial y seguimiento ---
        if evals % cfg.intervalo_historial == 0:
            registrar()
        if cfg.verbose and evals % cfg.intervalo_seguimiento == 0:
            print(f"[Eval {evals}] mejor={f_mejor:.2f}  actual={f_actual:.2f}  T={T:.2e}  "
                  f"aceptación(peores, últ. ventana)="
                  f"{res.historial_tasa_aceptacion_peores[-1]:.2f}  "
                  f"eval.reales={n_reales[0]}  t={time.time() - t_ini:.1f}s")

    if res.historial_evaluaciones[-1] != evals:
        registrar()
    res.mejor_individuo, res.mejor_fitness = mejor, f_mejor
    res.evaluaciones_totales = evals
    res.pasos_temperatura = paso
    res.T_final = T
    res.n_recalentados = n_recal
    res.motivo_parada = motivo
    res.tiempo_total = time.time() - t_ini
    if cfg.verbose:
        print(f"[Fin] {motivo}: {evals} evaluaciones ({n_reales[0]} reales), "
              f"mejor={f_mejor:.2f}, T_final={T:.2e}, t={res.tiempo_total:.1f}s")
    return res
