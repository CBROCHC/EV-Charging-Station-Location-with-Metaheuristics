"""
distancias.py
=============

Cálculo de d_G(i,j): la distancia de camino más corto entre cada nodo
i in V y cada candidato j in J, sobre el grafo G=(V,E) usando Dijkstra
(networkx ya trae una implementación eficiente).

Esto se calcula UNA sola vez por problema (no depende de qué estaciones
se instalen finalmente), por lo que se hace fuera del bucle del algoritmo
genético y se pasa como tabla precalculada a la función de fitness.
"""

from typing import Dict, Hashable

import networkx as nx

from problema import ProblemaCarga, Nodo

# distancias[j][i] = d_G(i, j), solo para los nodos i que estén a distancia
# <= radio_maximo del candidato j (más allá de eso, ninguna estación -ni
# lenta ni rápida- podría cubrir a i, ver restricción 5 del enunciado).
TablaDistancias = Dict[Hashable, Dict[Hashable, float]]


def calcular_distancias(problema: ProblemaCarga) -> TablaDistancias:
    """Calcula d_G(i,j) para todo candidato j in J y todo nodo i in V,
    limitando la búsqueda al radio de cobertura más grande entre los tipos
    de estación (normalmente el de la estación rápida), ya que más allá de
    ese radio ninguna y_ij^t podrá ser distinta de cero.

    Se ejecuta un Dijkstra (con corte/cutoff) por cada candidato j, lo cual
    es barato porque |J| es mucho menor que |V| en un caso realista.

    Devuelve
    --------
    Un diccionario anidado: distancias[j][i] = d_G(i,j) (float), presente
    solo si d_G(i,j) <= max(R^t para t en T).
    """
    radio_maximo = max(problema.radio.values())
    distancias: TablaDistancias = {}
    for j in problema.candidatos:
        # single_source_dijkstra_path_length con cutoff: mucho más barato
        # que calcular TODAS las distancias y filtrar después, sobre todo
        # cuando el grafo real de una ciudad completa sea grande.
        distancias[j] = nx.single_source_dijkstra_path_length(
            problema.grafo, source=j, cutoff=radio_maximo, weight="length"
        )
    return distancias


def nodos_cubiertos(distancias: TablaDistancias, j: Nodo, tipo: str, problema: ProblemaCarga):
    """Devuelve la lista de nodos i in V tales que d_G(i,j) <= R^tipo, es
    decir, los nodos que SÍ podrían recibir demanda atendida por una
    estación de tipo `tipo` instalada en el candidato `j` (restricción 5).
    """
    radio_tipo = problema.radio[tipo]
    return [i for i, d in distancias.get(j, {}).items() if d <= radio_tipo]
