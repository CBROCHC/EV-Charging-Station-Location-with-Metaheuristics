"""
fitness.py
==========

Evaluación del individuo (función objetivo F(x,y) del enunciado).

Dado un x YA FACTIBLE (después de reparar_individuo), las variables y_ij^t
óptimas no hacen falta buscarlas con el algoritmo genético: se obtienen
resolviendo un problema de FLUJO MÁXIMO en una red bipartita, que es
exactamente equivalente al subproblema:

    max  sum_i sum_j sum_t y_ij^t
    s.a. sum_i y_ij^t <= cap_j^t                 (restr. 3, con x_j^t=1 fijo)
         sum_j sum_t y_ij^t <= w_i               (restr. 4)
         y_ij^t = 0 si d_G(i,j) > R^t            (restr. 5)
         y_ij^t >= 0                             (restr. 7)

Construcción de la red de flujo:

    S -> i         capacidad w_i           (oferta de demanda del nodo i)
    i -> (j,t)     capacidad infinita      (solo si d_G(i,j) <= R^t y x_j^t=1)
    (j,t) -> T     capacidad cap_j^t       (capacidad de la estación abierta)

El valor del flujo máximo de S a T es exactamente sum y_ij^t óptimo, y con
networkx se resuelve de forma exacta y muy rápida (Dinic/preflow-push),
mucho más barato que llamar a un solver LP genérico en cada evaluación
--y el algoritmo genético evalúa miles de individuos, así que esto importa.
"""

from typing import Dict, Tuple

import networkx as nx

from problema import ProblemaCarga, Nodo
from distancias import TablaDistancias, nodos_cubiertos
from codificacion import Individuo, decodificar_individuo, reparar_individuo

FUENTE = "__FUENTE__"
SUMIDERO = "__SUMIDERO__"


def _nodo_estacion(j: Nodo, t: str):
    """Identificador único del nodo (j,t) dentro de la red de flujo (para
    no confundirlo con un nodo de demanda i si J y V comparten ids)."""
    return ("estacion", j, t)


def construir_red_flujo(
    x: Dict[Tuple[Nodo, str], int],
    problema: ProblemaCarga,
    distancias: TablaDistancias,
) -> nx.DiGraph:
    """Construye la red de flujo descrita arriba para un x dado (ya se
    asume factible en presupuesto/capacidad de subestación; esta red solo
    modela las restricciones 3, 4 y 5, que son las que definen a y)."""
    G = nx.DiGraph()
    G.add_node(FUENTE)
    G.add_node(SUMIDERO)

    estaciones_abiertas = [
        (j, t) for j in problema.candidatos for t in problema.tipos if x[(j, t)]
    ]

    # (j,t) -> T con capacidad cap_j^t (restricción 3)
    for j, t in estaciones_abiertas:
        nodo_est = _nodo_estacion(j, t)
        G.add_edge(nodo_est, SUMIDERO, capacity=problema.capacidad[(j, t)])

    # S -> i con capacidad w_i (restricción 4), solo para los nodos i que
    # tengan AL MENOS una estación abierta dentro de su radio (para no
    # meter nodos irrelevantes en la red).
    nodos_relevantes = set()
    for j, t in estaciones_abiertas:
        for i in nodos_cubiertos(distancias, j, t, problema):
            nodos_relevantes.add(i)
            # i -> (j,t) con capacidad "infinita" (restricción 5: la arista
            # solo existe si d_G(i,j) <= R^t; el límite real lo ponen las
            # capacidades de S->i y (j,t)->T).
            G.add_edge(i, _nodo_estacion(j, t), capacity=float("inf"))

    for i in nodos_relevantes:
        G.add_edge(FUENTE, i, capacity=problema.demanda[i])

    return G


def evaluar_x(
    x: Dict[Tuple[Nodo, str], int],
    problema: ProblemaCarga,
    distancias: TablaDistancias,
    devolver_flujo: bool = False,
):
    """Resuelve el subproblema de flujo máximo para un x dado y devuelve la
    demanda total atendida (el fitness). Si devolver_flujo=True, además
    devuelve el diccionario de flujo y_ij^t (útil solo para inspección o
    visualización final, no durante la evolución del AG por eficiencia).
    """
    red = construir_red_flujo(x, problema, distancias)
    if not red.has_node(FUENTE) or FUENTE not in red or red.out_degree(FUENTE) == 0:
        # No hay ninguna estación abierta / ningún nodo cubierto: fitness 0.
        return (0.0, {}) if devolver_flujo else 0.0

    if devolver_flujo:
        valor, flujo = nx.maximum_flow(red, FUENTE, SUMIDERO, capacity="capacity")
        # Extraemos solo las aristas i -> (j,t) con flujo > 0, que son las
        # y_ij^t > 0 del modelo original.
        y = {}
        for i, destinos in flujo.items():
            if i == FUENTE or not isinstance(i, (int, str)) or i == SUMIDERO:
                continue
            for destino, cantidad in destinos.items():
                if isinstance(destino, tuple) and destino[0] == "estacion" and cantidad > 1e-9:
                    _, j, t = destino
                    y[(i, j, t)] = cantidad
        return valor, y
    else:
        valor = nx.maximum_flow_value(red, FUENTE, SUMIDERO, capacity="capacity")
        return valor


def evaluar_individuo(
    individuo: Individuo,
    problema: ProblemaCarga,
    distancias: TablaDistancias,
    reparar: bool = True,
):
    """Función de evaluación completa usada por el algoritmo genético:

        1. (opcional) repara el individuo -> garantiza restricciones 2 y 6.
        2. decodifica a x -> garantiza restricción 1 (por construcción).
        3. resuelve el flujo máximo -> y óptimo, restricciones 3, 4, 5, 7.
        4. fitness = valor del flujo máximo = sum y_ij^t.

    Devuelve (fitness, individuo_reparado) para que el llamante pueda
    sustituir el genotipo original por el reparado (la mejora se hereda en la siguiente generación).
    """
    if reparar:
        individuo = reparar_individuo(individuo, problema)
    x = decodificar_individuo(individuo, problema)
    fitness = evaluar_x(x, problema, distancias)
    return fitness, individuo
