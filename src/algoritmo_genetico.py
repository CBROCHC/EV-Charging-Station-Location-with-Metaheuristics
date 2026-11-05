"""
algoritmo_genetico.py
======================

Bucle principal del algoritmo genético para el problema de localización de
estaciones de carga, usando:

  - Codificación A (vector de genes en {0,1,2}), ver codificacion.py.
  - Fitness = flujo máximo del subproblema de y, ver fitness.py.
  - Selección por torneo.
  - Cruce por subestación (o uniforme, configurable) + reparación.
  - Mutación por gen + mutación de INTERCAMBIO (mover / cambiar tipo / abrir,
    respetando presupuesto y subestaciones) + reparación.
  - Reemplazo generacional con elitismo.

Novedades:
  - Mutación de intercambio: explora vecinos que conservan (casi) el coste,
    en lugar de depender de cambios de un solo gen que la reparación deshace.
  - Caché de fitness por genotipo: los clones y la élite no se reevalúan.
  - Cruce por SUBESTACIÓN: respeta la estructura del problema (cada
    subestación = un barrio); el hijo hereda barrios completos de cada padre.
  - Diversidad GENOTÍPICA (distancia de Hamming media y nº de genotipos
    distintos) en lugar de la desviación típica del fitness.
  - Criterio de parada por "estancamiento" (cuando el fitness no evoluciona).
  - Parámetros paciencia_estancamiento y tol_estancamiento configurables.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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


@dataclass
class ConfiguracionAG:
    """Hiperparámetros del algoritmo genético. Se pasan explícitamente
    desde main.py para poder variarlos fácilmente entre ejecuciones.

    criterio_parada:
        "generaciones"  -> se ejecutan exactamente `num_generaciones`.
        "tiempo"        -> se ejecuta hasta agotar `tiempo_max_segundos`.
        "evaluaciones"  -> se detiene al agotar `max_evaluaciones` SOLUCIONES GENERADAS
                           (población inicial + hijos de cada generación; la élite
                           copiada no cuenta). Es el criterio justo para comparar
                           configuraciones con distinto tamaño de población.
        "estancamiento" -> se detiene si el fitness no mejora al menos
                           `tol_estancamiento` durante `paciencia_estancamiento`
                           generaciones consecutivas.
    """

    tam_poblacion: int = 60

    criterio_parada: str = "estancamiento"        # "generaciones" | "tiempo" | "estancamiento"
    num_generaciones: int = 150                  # usado si criterio_parada == "generaciones"
    max_evaluaciones: Optional[int] = None       # usado si criterio_parada == "evaluaciones"
    tiempo_max_segundos: Optional[float] = None    # usado si criterio_parada == "tiempo"
    
    # Parámetros para la condición de parada por estancamiento
    paciencia_estancamiento: int = 20            # nº de generaciones sin mejora significativa para detener
    tol_estancamiento: float = 1e-3              # diferencia mínima en fitness para considerar evolución

    num_semillas_greedy: int = 2
    tipo_cruce: str = "subestacion"              # "subestacion" | "uniforme"
    prob_cruce: float = 0.9
    prob_mutacion_gen: Optional[float] = None    # si None, se usa 1/|J|
    prob_mutacion_intercambio: float = 0.7       # prob. por hijo de aplicar la mutación de intercambio
    max_intercambios: int = 3                    # nº de movimientos por mutación: uniforme en 1..max
    usar_cache_fitness: bool = True              # cachear fitness por genotipo
    tam_torneo: int = 3
    num_elite: int = 2

    intervalo_seguimiento: int = 25              # cada cuántas generaciones se informa por consola
    verbose: bool = True

    semilla: Optional[int] = None

    def __post_init__(self):
        if self.tipo_cruce not in ("subestacion", "uniforme"):
            raise ValueError("tipo_cruce debe ser 'subestacion' o 'uniforme'")
        if self.criterio_parada not in ("generaciones", "tiempo", "estancamiento", "evaluaciones"):
            raise ValueError(
                "criterio_parada debe ser 'generaciones', 'tiempo', 'estancamiento' o 'evaluaciones'"
            )
        if self.criterio_parada == "evaluaciones" and not self.max_evaluaciones:
            raise ValueError("Con criterio_parada='evaluaciones' hay que indicar max_evaluaciones")
        if self.criterio_parada == "tiempo" and not self.tiempo_max_segundos:
            raise ValueError(
                "Con criterio_parada='tiempo' hay que indicar tiempo_max_segundos (en segundos)"
            )


@dataclass
class ResultadoAG:
    """Historial COMPLETO de una ejecución del algoritmo genético.

    Todas las listas `historial_*` tienen la misma longitud
    (generaciones_ejecutadas + 1): la posición 0 es la población inicial
    (generación 0) y la posición g es el estado tras la generación g.
    """

    mejor_individuo: Individuo
    mejor_fitness: float

    historial_individuos: List[Individuo] = field(default_factory=list)  # mejor-hasta-ahora, por generación
    historial_mejor: List[float] = field(default_factory=list)
    historial_medio: List[float] = field(default_factory=list)
    historial_min: List[float] = field(default_factory=list)
    historial_max: List[float] = field(default_factory=list)
    historial_std: List[float] = field(default_factory=list)
    historial_hamming: List[float] = field(default_factory=list)         # dist. Hamming media normalizada [0,1]
    historial_genotipos_distintos: List[int] = field(default_factory=list)  # nº de genotipos únicos
    historial_evaluaciones_reales: List[int] = field(default_factory=list)  # evaluaciones de flujo máximo (fallos de caché)
    historial_tiempo: List[float] = field(default_factory=list)          # segundos acumulados
    historial_evaluaciones: List[int] = field(default_factory=list)      # nº acumulado de evaluaciones

    fitness_poblacion_final: List[float] = field(default_factory=list)
    generaciones_ejecutadas: int = 0


# ----------------------------------------------------------------------
# Población inicial
# ----------------------------------------------------------------------
def generar_poblacion_inicial(
    problema: ProblemaCarga, config: ConfiguracionAG, rng: np.random.Generator
) -> List[Individuo]:
    poblacion = [
        generar_individuo_greedy(problema, rng) for _ in range(config.num_semillas_greedy)
    ]
    while len(poblacion) < config.tam_poblacion:
        poblacion.append(generar_individuo_aleatorio(problema, rng=rng))
    return poblacion


# ----------------------------------------------------------------------
# Selección, cruce y mutación
# ----------------------------------------------------------------------
def seleccion_torneo(
    poblacion: List[Individuo], fitnesses: List[float], k: int, rng: np.random.Generator
) -> Individuo:
    indices = rng.integers(0, len(poblacion), size=k)
    mejor_idx = max(indices, key=lambda idx: fitnesses[idx])
    return poblacion[mejor_idx].copy()


def cruce_uniforme(
    padre1: Individuo, padre2: Individuo, rng: np.random.Generator
) -> Tuple[Individuo, Individuo]:
    mascara = rng.integers(0, 2, size=len(padre1)).astype(bool)
    hijo1 = np.where(mascara, padre1, padre2)
    hijo2 = np.where(mascara, padre2, padre1)
    return hijo1, hijo2


def construir_bloques_subestacion(problema: ProblemaCarga) -> List[np.ndarray]:
    """Lista de arrays de índices de gen agrupados por subestación g(j)."""
    por_sub = {}
    for idx, j in enumerate(problema.candidatos):
        por_sub.setdefault(problema.subestacion[j], []).append(idx)
    return [np.array(v, dtype=int) for _, v in sorted(por_sub.items())]


def cruce_por_subestacion(
    padre1: Individuo,
    padre2: Individuo,
    bloques: List[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Individuo, Individuo]:
    """Cruce que respeta la estructura: para cada subestación (barrio) se
    decide, al azar, si el hijo1 hereda TODOS sus genes del padre1 o del
    padre2 (el hijo2 recibe el bloque contrario). Así se conservan
    configuraciones locales coherentes (qué tipo de estación en cada barrio)
    en lugar de mezclar genes sueltos de todo el mapa."""
    hijo1, hijo2 = padre1.copy(), padre2.copy()
    for idxs in bloques:
        if rng.random() < 0.5:
            hijo1[idxs], hijo2[idxs] = padre2[idxs], padre1[idxs]
    return hijo1, hijo2


def diversidad_genotipica(poblacion: List[Individuo]) -> Tuple[float, int]:
    """(distancia de Hamming media normalizada entre pares, nº de genotipos
    distintos). La distancia normalizada es la fracción media de genes en
    que difieren dos individuos cualesquiera (0 = clones, ~0.6 = aleatorio).
    Se calcula en O(N*L) contando valores por gen, sin comparar todos los pares."""
    M = np.asarray(poblacion)
    n, L = M.shape
    if n < 2:
        return 0.0, n
    pares_distintos = np.zeros(L)
    for v in (0, 1, 2):
        c = (M == v).sum(axis=0)
        pares_distintos += c * (n - c)       # pares (a,b) con a==v, b!=v
    # cada par distinto se cuenta una vez por cada uno de sus dos valores -> /2
    pares_distintos /= 2.0
    hamming = float(pares_distintos.sum() / (n * (n - 1) / 2.0) / L)
    distintos = len({fila.astype(np.int8).tobytes() for fila in M})
    return hamming, distintos


@dataclass
class ContextoMutacion:
    """Tablas en numpy para comprobar rápido presupuesto y subestaciones.
    Fila 0 = sin estación (coste/capacidad 0), fila 1 = lenta, fila 2 = rápida."""
    coste: np.ndarray        # (3, n)
    cap: np.ndarray          # (3, n)
    sub: np.ndarray          # (n,) índice de subestación de cada candidato
    cap_sub: np.ndarray      # (n_sub,)
    presupuesto: float


def construir_contexto_mutacion(problema: ProblemaCarga) -> ContextoMutacion:
    n = len(problema.candidatos)
    coste = np.zeros((3, n))
    cap = np.zeros((3, n))
    for idx, j in enumerate(problema.candidatos):
        coste[1, idx], coste[2, idx] = problema.coste[(j, "lenta")], problema.coste[(j, "rapida")]
        cap[1, idx], cap[2, idx] = problema.capacidad[(j, "lenta")], problema.capacidad[(j, "rapida")]
    ids = sorted(problema.capacidad_subestacion)
    pos = {k: p for p, k in enumerate(ids)}
    sub = np.array([pos[problema.subestacion[j]] for j in problema.candidatos], dtype=int)
    cap_sub = np.array([problema.capacidad_subestacion[k] for k in ids])
    return ContextoMutacion(coste, cap, sub, cap_sub, problema.presupuesto)


def _es_factible_ctx(ind: Individuo, ctx: ContextoMutacion) -> bool:
    cols = np.arange(len(ind))
    if ctx.coste[ind, cols].sum() > ctx.presupuesto + 1e-9:
        return False
    uso = np.bincount(ctx.sub, weights=ctx.cap[ind, cols], minlength=len(ctx.cap_sub))
    return bool(np.all(uso <= ctx.cap_sub + 1e-9))


def mutacion_intercambio(
    individuo: Individuo, ctx: ContextoMutacion, max_movimientos: int, rng: np.random.Generator
) -> Individuo:
    """Aplica 1..max_movimientos movimientos de vecindad, aceptando solo los
    que dejan la solución factible (así la reparación no los deshace):
      - 'mover' (60 %): cierra una estación abierta y abre otra en un
        candidato cerrado (tipo al azar);
      - 'tipo'  (25 %): cambia una estación abierta de lenta a rápida o viceversa;
      - 'abrir' (15 %): abre una estación nueva si aún cabe en presupuesto y subestación.
    Cada movimiento reintenta hasta 10 veces con elecciones distintas."""
    ind = individuo.copy()
    n_mov = int(rng.integers(1, max_movimientos + 1))
    for _ in range(n_mov):
        u = rng.random()
        for _intento in range(10):
            abiertos = np.flatnonzero(ind > 0)
            cerrados = np.flatnonzero(ind == 0)
            cand = ind.copy()
            if u < 0.60 and len(abiertos) and len(cerrados):
                cand[int(rng.choice(abiertos))] = 0
                cand[int(rng.choice(cerrados))] = int(rng.integers(1, 3))
            elif u < 0.85 and len(abiertos):
                a = int(rng.choice(abiertos))
                cand[a] = 3 - cand[a]
            elif len(cerrados):
                cand[int(rng.choice(cerrados))] = int(rng.integers(1, 3))
            else:
                break
            if _es_factible_ctx(cand, ctx):
                ind = cand
                break
    return ind


def mutacion(individuo: Individuo, prob_mutacion_gen: float, rng: np.random.Generator) -> Individuo:
    individuo = individuo.copy()
    for idx in range(len(individuo)):
        if rng.random() < prob_mutacion_gen:
            opciones = [g for g in (0, 1, 2) if g != individuo[idx]]
            individuo[idx] = rng.choice(opciones)
    return individuo


# ----------------------------------------------------------------------
# Bucle principal
# ----------------------------------------------------------------------
def _evaluar_poblacion(poblacion, problema, distancias, cache=None, contador=None):
    """Evalúa toda la población y devuelve la lista de fitnesses. Si se pasa
    `cache` (dict genotipo->fitness) los genotipos ya vistos no se
    reevalúan; `contador` (lista de 1 elemento) acumula las evaluaciones reales."""
    fitnesses = []
    for idx, ind in enumerate(poblacion):
        clave = ind.astype(np.int8).tobytes() if cache is not None else None
        if clave is not None and clave in cache:
            fitnesses.append(cache[clave])
            continue
        fit, ind_reparado = evaluar_individuo(ind, problema, distancias, reparar=False)
        poblacion[idx] = ind_reparado
        if contador is not None:
            contador[0] += 1
        if clave is not None:
            cache[clave] = fit
        fitnesses.append(fit)
    return fitnesses


def ejecutar_algoritmo_genetico(
    problema: ProblemaCarga,
    distancias: TablaDistancias,
    config: ConfiguracionAG = None,
    callback=None,
) -> ResultadoAG:
    """Ejecuta el algoritmo genético completo.

    callback: opcional. callback(evaluaciones, tiempo_s, mejor_fitness, mejor_individuo)
    se llama al terminar cada generación (y tras la población inicial). Sirve para
    guardar checkpoints en ejecuciones largas (12 h); no altera el algoritmo."""
    config = config or ConfiguracionAG()
    rng = np.random.default_rng(config.semilla)
    prob_mutacion_gen = config.prob_mutacion_gen or (1.0 / len(problema.candidatos))
    bloques_subestacion = construir_bloques_subestacion(problema)
    ctx_mut = construir_contexto_mutacion(problema)
    cache = {} if config.usar_cache_fitness else None
    n_eval_reales = [0]

    tiempo_inicio = time.time()

    # --- Población inicial + evaluación (generación 0) ---
    poblacion = generar_poblacion_inicial(problema, config, rng)
    fitnesses = _evaluar_poblacion(poblacion, problema, distancias, cache, n_eval_reales)

    idx_mejor = int(np.argmax(fitnesses))
    mejor_individuo_global = poblacion[idx_mejor].copy()
    mejor_fitness_global = fitnesses[idx_mejor]

    historial_individuos = [mejor_individuo_global.copy()]
    historial_mejor = [mejor_fitness_global]
    historial_medio = [float(np.mean(fitnesses))]
    historial_min = [float(np.min(fitnesses))]
    historial_max = [float(np.max(fitnesses))]
    historial_std = [float(np.std(fitnesses))]
    _h, _d = diversidad_genotipica(poblacion)
    historial_hamming = [_h]
    historial_genotipos_distintos = [_d]
    historial_tiempo = [time.time() - tiempo_inicio]
    historial_evaluaciones = [len(poblacion)]
    historial_evaluaciones_reales = [n_eval_reales[0]]

    ultimo_checkpoint = mejor_fitness_global
    generaciones_sin_mejora = 0
    if callback is not None:
        callback(historial_evaluaciones[-1], historial_tiempo[-1], mejor_fitness_global, mejor_individuo_global)

    if config.verbose:
        print(
            f"[Gen 0] mejor={mejor_fitness_global:.2f}  medio={historial_medio[0]:.2f}  "
            f"t={historial_tiempo[0]:.1f}s"
        )

    # --- Bucle de generaciones ---
    generacion = 0
    while True:
        generacion += 1

        # Control de criterios de parada
        if config.criterio_parada == "generaciones" and generacion > config.num_generaciones:
            generacion -= 1
            break
        if (
            config.criterio_parada == "evaluaciones"
            and historial_evaluaciones[-1] + (config.tam_poblacion - config.num_elite)
            > config.max_evaluaciones
        ):
            generacion -= 1
            break
        if (
            config.criterio_parada == "tiempo"
            and (time.time() - tiempo_inicio) >= config.tiempo_max_segundos
        ):
            generacion -= 1
            break
        if (
            config.criterio_parada == "estancamiento"
            and generaciones_sin_mejora >= config.paciencia_estancamiento
        ):
            if config.verbose:
                print(
                    f"\n[Parada por estancamiento] Se han alcanzado {generaciones_sin_mejora} generaciones consecutivas "
                    f"sin mejora significativa (umbral >= {config.tol_estancamiento}). Deteniendo ejecución."
                )
            generacion -= 1
            break

        # Elitismo
        orden_elite = np.argsort(fitnesses)[::-1][: config.num_elite]
        elite = [poblacion[i].copy() for i in orden_elite]

        nueva_poblacion: List[Individuo] = []
        while len(nueva_poblacion) < config.tam_poblacion - config.num_elite:
            padre1 = seleccion_torneo(poblacion, fitnesses, config.tam_torneo, rng)
            padre2 = seleccion_torneo(poblacion, fitnesses, config.tam_torneo, rng)

            if rng.random() < config.prob_cruce:
                if config.tipo_cruce == "subestacion":
                    hijo1, hijo2 = cruce_por_subestacion(padre1, padre2, bloques_subestacion, rng)
                else:
                    hijo1, hijo2 = cruce_uniforme(padre1, padre2, rng)
            else:
                hijo1, hijo2 = padre1.copy(), padre2.copy()

            hijo1 = mutacion(hijo1, prob_mutacion_gen, rng)
            hijo2 = mutacion(hijo2, prob_mutacion_gen, rng)

            if rng.random() < config.prob_mutacion_intercambio:
                hijo1 = mutacion_intercambio(hijo1, ctx_mut, config.max_intercambios, rng)
            if rng.random() < config.prob_mutacion_intercambio:
                hijo2 = mutacion_intercambio(hijo2, ctx_mut, config.max_intercambios, rng)

            hijo1 = reparar_individuo(hijo1, problema)
            hijo2 = reparar_individuo(hijo2, problema)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < config.tam_poblacion - config.num_elite:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion + elite
        fitnesses = _evaluar_poblacion(poblacion, problema, distancias, cache, n_eval_reales)

        idx_mejor_gen = int(np.argmax(fitnesses))
        mejora_actual = fitnesses[idx_mejor_gen] - mejor_fitness_global

        if mejora_actual > config.tol_estancamiento:
            mejor_fitness_global = fitnesses[idx_mejor_gen]
            mejor_individuo_global = poblacion[idx_mejor_gen].copy()
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1
            if fitnesses[idx_mejor_gen] > mejor_fitness_global:
                mejor_fitness_global = fitnesses[idx_mejor_gen]
                mejor_individuo_global = poblacion[idx_mejor_gen].copy()

        historial_individuos.append(mejor_individuo_global.copy())
        historial_mejor.append(mejor_fitness_global)
        historial_medio.append(float(np.mean(fitnesses)))
        historial_min.append(float(np.min(fitnesses)))
        historial_max.append(float(np.max(fitnesses)))
        historial_std.append(float(np.std(fitnesses)))
        _h, _d = diversidad_genotipica(poblacion)
        historial_hamming.append(_h)
        historial_genotipos_distintos.append(_d)
        historial_tiempo.append(time.time() - tiempo_inicio)
        # Soluciones GENERADAS: solo los hijos (la élite es una copia, no se cuenta).
        historial_evaluaciones.append(historial_evaluaciones[-1] + len(nueva_poblacion))
        historial_evaluaciones_reales.append(n_eval_reales[0])
        if callback is not None:
            callback(historial_evaluaciones[-1], historial_tiempo[-1], mejor_fitness_global, mejor_individuo_global)

        if config.verbose and (generacion % config.intervalo_seguimiento == 0):
            mejora = mejor_fitness_global - ultimo_checkpoint
            print(
                f"[Gen {generacion}] mejor={mejor_fitness_global:.2f}  "
                f"medio={historial_medio[-1]:.2f}  "
                f"hamming={historial_hamming[-1]:.3f}  distintos={historial_genotipos_distintos[-1]}  "
                f"eval.reales={n_eval_reales[0]}/{historial_evaluaciones[-1]}  "
                f"mejora últimas {config.intervalo_seguimiento} gen: +{mejora:.2f}  "
                f"t={historial_tiempo[-1]:.1f}s"
            )
            ultimo_checkpoint = mejor_fitness_global

    if config.verbose:
        print(
            f"[Fin] {generacion} generaciones ejecutadas, mejor fitness={mejor_fitness_global:.2f}, "
            f"tiempo total={historial_tiempo[-1]:.1f}s"
        )

    return ResultadoAG(
        mejor_individuo=mejor_individuo_global,
        mejor_fitness=mejor_fitness_global,
        historial_individuos=historial_individuos,
        historial_mejor=historial_mejor,
        historial_medio=historial_medio,
        historial_min=historial_min,
        historial_max=historial_max,
        historial_std=historial_std,
        historial_hamming=historial_hamming,
        historial_genotipos_distintos=historial_genotipos_distintos,
        historial_tiempo=historial_tiempo,
        historial_evaluaciones=historial_evaluaciones,
        historial_evaluaciones_reales=historial_evaluaciones_reales,
        fitness_poblacion_final=fitnesses,
        generaciones_ejecutadas=generacion,
    )