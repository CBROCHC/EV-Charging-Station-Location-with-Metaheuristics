"""
codificacion.py
================

Implementa la CODIFICACIÓN A elegida para el individuo del algoritmo
genético:

    individuo = vector de longitud |J|, con genes en {0, 1, 2}
        0 -> no se instala ninguna estación en ese candidato
        1 -> se instala una estación LENTA
        2 -> se instala una estación RÁPIDA

Esta codificación garantiza AUTOMÁTICAMENTE la restricción 1 del enunciado
(como mucho un tipo de estación por candidato), porque cada gen solo puede
tomar uno de los tres valores. Las restricciones 2 (presupuesto) y 6
(capacidad de cada subestación eléctrica) NO están garantizadas por la
codificación y se imponen con el operador de reparación `reparar_individuo`.

La demanda atendida y_ij^t NO forma parte del individuo: se calcula de
forma óptima y exacta a partir de x en fitness.py (subproblema de flujo
máximo), así que aquí solo nos ocupamos de x.
"""

from typing import Dict, List, Tuple

import numpy as np

from problema import ProblemaCarga, Nodo

# Un individuo es, literalmente, un array de numpy de enteros en {0,1,2}.
Individuo = np.ndarray

SIN_ESTACION = 0
LENTA = 1
RAPIDA = 2
_GEN_A_TIPO = {LENTA: "lenta", RAPIDA: "rapida"}


# ----------------------------------------------------------------------
# Codificación / decodificación
# ----------------------------------------------------------------------
def decodificar_individuo(individuo: Individuo, problema: ProblemaCarga) -> Dict[Tuple[Nodo, str], int]:
    """Traduce el vector de genes {0,1,2} a las variables binarias x_j^t
    del modelo matemático. Devuelve un diccionario con entrada 0/1 para
    TODOS los pares (j, t) de problema.candidatos x problema.tipos (no solo
    los que están a 1), para poder usarlo directamente en las
    comprobaciones de restricciones y en la construcción de la red de
    flujo de fitness.py.
    """
    x: Dict[Tuple[Nodo, str], int] = {}
    for gen, j in zip(individuo, problema.candidatos):
        for t in problema.tipos:
            x[(j, t)] = 0
        if gen != SIN_ESTACION:
            x[(j, _GEN_A_TIPO[gen])] = 1
    return x


# ----------------------------------------------------------------------
# Comprobación de restricciones 2 y 6 (las que la codificación no garantiza)
# ----------------------------------------------------------------------
def coste_total(x: Dict[Tuple[Nodo, str], int], problema: ProblemaCarga) -> float:
    """Coste total de instalación de la solución x (para comprobar la
    restricción 2, sum_j sum_t c_j^t * x_j^t <= B)."""
    return sum(
        problema.coste[(j, t)] * x[(j, t)]
        for j in problema.candidatos
        for t in problema.tipos
    )


def uso_por_subestacion(x: Dict[Tuple[Nodo, str], int], problema: ProblemaCarga) -> Dict[int, float]:
    """Capacidad total INSTALADA (no la demanda realmente servida) que
    carga cada subestación k, para comprobar la restricción 6:
    sum_{j: g(j)=k} sum_t cap_j^t * x_j^t <= Cap_k.

    Nótese que esta restricción depende solo de x (qué se instala), no de
    y (cuánta demanda se sirve realmente) — por eso se puede comprobar sin
    necesidad de resolver el subproblema de flujo.
    """
    uso = {k: 0.0 for k in problema.capacidad_subestacion}
    for j in problema.candidatos:
        k = problema.subestacion[j]
        for t in problema.tipos:
            if x[(j, t)]:
                uso[k] += problema.capacidad[(j, t)]
    return uso


def es_factible(x: Dict[Tuple[Nodo, str], int], problema: ProblemaCarga) -> bool:
    """True si x respeta el presupuesto (restr. 2) y todas las subestaciones
    (restr. 6). La restricción 1 se asume garantizada por la codificación."""
    if coste_total(x, problema) > problema.presupuesto + 1e-9:
        return False
    uso = uso_por_subestacion(x, problema)
    return all(uso[k] <= problema.capacidad_subestacion[k] + 1e-9 for k in uso)


# ----------------------------------------------------------------------
# Operador de reparación (lamarckiano: modifica el propio genotipo)
# ----------------------------------------------------------------------
def _ratio_eficiencia(j: Nodo, t: str, problema: ProblemaCarga) -> float:
    """cap_j^t / c_j^t: cuánta capacidad de atención de demanda se obtiene
    por unidad de coste. Se usa como criterio para decidir qué estación
    "sobra" primero cuando hay que apagar alguna para recuperar
    factibilidad: apagamos siempre la de peor ratio."""
    return problema.capacidad[(j, t)] / problema.coste[(j, t)]


def reparar_individuo(individuo: Individuo, problema: ProblemaCarga) -> Individuo:
    """Operador de reparación LAMARCKIANO para la codificación A.

    Mientras la solución viole el presupuesto (restricción 2) o la
    capacidad de alguna subestación (restricción 6), apaga (pone a 0) la
    estación abierta con PEOR ratio capacidad/coste entre las que
    contribuyen a la restricción violada. Es exactamente la misma lógica
    que vaciar una mochila que pesa de más quitando primero el objeto
    menos "rentable".

    Se modifica y se devuelve una COPIA del individuo (no se toca el
    original in-place, para que el llamante decida qué hacer con él).

    Este mismo operador se usa en tres sitios del algoritmo genético:
    generación de la población inicial, después del cruce y después de
    la mutación — siempre que un genotipo pueda haberse vuelto
    infactible en presupuesto o capacidad de subestación.
    """
    individuo = individuo.copy()

    while True:
        x = decodificar_individuo(individuo, problema)
        coste = coste_total(x, problema)
        uso = uso_por_subestacion(x, problema)

        # Candidatas a "apagar": todas las estaciones abiertas que están
        # contribuyendo a ALGUNA restricción violada en este momento.
        subestaciones_saturadas = {
            k for k, usado in uso.items() if usado > problema.capacidad_subestacion[k] + 1e-9
        }
        presupuesto_excedido = coste > problema.presupuesto + 1e-9

        if not presupuesto_excedido and not subestaciones_saturadas:
            break  # ya es factible: restricciones 2 y 6 cumplidas

        candidatas = []
        for idx, j in enumerate(problema.candidatos):
            gen = individuo[idx]
            if gen == SIN_ESTACION:
                continue
            t = _GEN_A_TIPO[gen]
            contribuye = presupuesto_excedido or (problema.subestacion[j] in subestaciones_saturadas)
            if contribuye:
                candidatas.append((idx, j, t))

        if not candidatas:
            # No debería ocurrir si el problema está bien formado (Cap_k y
            # B siempre permiten la solución vacía), pero por seguridad
            # evitamos un bucle infinito.
            break

        # Apagamos la de peor ratio capacidad/coste.
        idx_peor, j_peor, t_peor = min(
            candidatas, key=lambda c: _ratio_eficiencia(c[1], c[2], problema)
        )
        individuo[idx_peor] = SIN_ESTACION

    return individuo


# ----------------------------------------------------------------------
# Generación de individuos
# ----------------------------------------------------------------------
def generar_individuo_aleatorio(
    problema: ProblemaCarga,
    probabilidades: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    rng: np.random.Generator = None,
) -> Individuo:
    """Genera un individuo al azar con sesgo hacia "no instalar" (ver
    justificación de diseño: reduce el trabajo del operador de reparación
    y da una población inicial más cercana a la región factible), y lo
    repara antes de devolverlo."""
    rng = rng or np.random.default_rng()
    n = len(problema.candidatos)
    individuo = rng.choice([SIN_ESTACION, LENTA, RAPIDA], size=n, p=probabilidades)
    return reparar_individuo(individuo, problema)


def generar_individuo_greedy(problema: ProblemaCarga, rng: np.random.Generator = None) -> Individuo:
    """Construye un individuo con una heurística voraz (greedy): ordena los
    candidatos por su mejor ratio capacidad/coste (probando primero el tipo
    rápida y si no cabe, la lenta) y los va abriendo mientras quepan en
    presupuesto y en la subestación correspondiente.

    Se usa como "semilla" en la población inicial (mezclada con individuos
    aleatorios) para dar al algoritmo genético un buen punto de partida sin
    sesgar toda la población hacia una única heurística.
    """
    rng = rng or np.random.default_rng()
    n = len(problema.candidatos)
    individuo = np.zeros(n, dtype=int)

    # Orden de prioridad: mejor ratio (capacidad/coste) usando el tipo
    # rápida como referencia (suele ser el tipo con mejor capacidad
    # marginal); un pequeño ruido evita que el greedy sea 100% determinista
    # y ayuda a que distintas semillas den distintas soluciones iniciales.
    orden = sorted(
        range(n),
        key=lambda idx: -_ratio_eficiencia(problema.candidatos[idx], "rapida", problema)
        * rng.uniform(0.9, 1.1),
    )

    presupuesto_restante = problema.presupuesto
    capacidad_restante = dict(problema.capacidad_subestacion)

    for idx in orden:
        j = problema.candidatos[idx]
        k = problema.subestacion[j]
        # Probamos primero rápida (más capacidad, más aporte potencial) y
        # si no cabe, probamos lenta.
        for gen, t in ((RAPIDA, "rapida"), (LENTA, "lenta")):
            c, cap = problema.coste[(j, t)], problema.capacidad[(j, t)]
            if c <= presupuesto_restante and cap <= capacidad_restante[k]:
                individuo[idx] = gen
                presupuesto_restante -= c
                capacidad_restante[k] -= cap
                break

    return individuo  # ya es factible por construcción, no necesita reparación
