"""
grafo_real.py
=============

Carga el grafo de calles de una ciudad o barrio REAL desde OpenStreetMap
usando la librería `osmnx`, y lo deja en el mismo formato que
generar_grafo_sintetico() de problema.py (nx.Graph con atributo de arista
'length' en metros y atributo de nodo 'pos'), de forma que
generar_parametros_problema() y todo el resto del pipeline (distancias,
codificación, fitness, algoritmo genético, visualización) funcionan EXACTAMENTE
igual sobre un grafo real que sobre uno sintético.

Diferencia importante con el grafo sintético: aquí 'pos' = (lon, lat) en
coordenadas geográficas reales (igual que las usa osmnx/folium), no (x, y)
en [0,1]^2. Por eso dibujar_mapa_real() de visualizacion.py solo tiene
sentido geográfico con grafos que vengan de este módulo.

AVISO IMPORTANTE (entorno de este chat)
----------------------------------------
Descargar un grafo real requiere consultar los servidores de OpenStreetMap
(Overpass API / Nominatim) por Internet. El entorno de ejecución de código
de este chat tiene el acceso a red restringido a una lista blanca de
dominios (PyPI, GitHub, etc.) que NO incluye los servidores de OpenStreetMap,
así que las funciones de este archivo NO se pueden ejecutar aquí dentro
-fallan por timeout de red-. Están escritas y listas para que las ejecutes
tú en tu propio ordenador, en Google Colab, o en cualquier entorno con
salida a Internet normal; el resto del proyecto (grafos sintéticos,
algoritmo genético, visualización matplotlib) sí se ha probado
íntegramente en este chat.
"""

import networkx as nx

from problema import ProblemaCarga, generar_parametros_problema


def cargar_grafo_real(
    lugar: str = None,
    punto: tuple = None,
    distancia_metros: float = 1000.0,
    tipo_red: str = "drive",
) -> nx.Graph:
    """Descarga y prepara el grafo de calles de una zona real.

    Parámetros (usar UNO de los dos primeros)
    ------------------------------------------
    lugar: nombre de lugar reconocible por Nominatim, p.ej.
        "Chamberí, Madrid, España" o "Gràcia, Barcelona, España".
        Útil para un barrio o distrito completo.
    punto: tupla (lat, lon) — se descarga la red de calles en un radio de
        `distancia_metros` alrededor de ese punto. Es la opción recomendada
        para las pruebas "reducidas" que comentas en el enunciado: basta
        con bajar `distancia_metros` (p.ej. 400-600 m) para tener un grafo
        pequeño de un barrio real con el que probar el algoritmo antes de
        escalar a la ciudad completa (subiendo `distancia_metros` o usando
        `lugar` con el nombre de la ciudad entera).
    distancia_metros: radio (solo si se usa `punto`).
    tipo_red: 'drive' (calles para coche, recomendado aquí), 'walk', 'bike'.

    Devuelve
    --------
    Un nx.Graph (no dirigido, sin aristas paralelas) con:
      - atributo de arista 'length' en metros (ya lo trae OSM)
      - atributo de nodo 'pos' = (lon, lat)
    listo para pasar a generar_parametros_problema().
    """
    import osmnx as ox

    
    if lugar is not None:
        G_dirigido = ox.graph_from_place(lugar, network_type=tipo_red)
    elif punto is not None:
        G_dirigido = ox.graph_from_point(punto, dist=distancia_metros, network_type=tipo_red)
    else:
        raise ValueError("Hay que indicar `lugar` o `punto`")

    # osmnx devuelve un MultiDiGraph; lo pasamos a Graph simple no dirigido
    # (nos basta con la topología y la longitud para el modelo del enunciado,
    # no necesitamos sentidos de circulación ni aristas múltiples).
    G = nx.Graph(ox.convert.to_undirected(G_dirigido))

    # Marcamos el grafo como "real" (coordenadas geográficas reales). Esto
    # se propaga a ProblemaCarga.es_real en generar_parametros_problema, y
    # main.py lo usa para generar el mapa interactivo automáticamente sin
    # que haya que acordarse de llamarlo a mano.
    G.graph["es_real"] = True

    # 'pos' = (lon, lat), igual que usan internamente osmnx/folium.
    for nodo, datos in G.nodes(data=True):
        G.nodes[nodo]["pos"] = (datos["x"], datos["y"])

    # Marca a nivel de grafo para que generar_parametros_problema() propague
    # problema.es_real=True automáticamente, y main.py sepa sin más
    # comprobación que debe generar también el mapa real interactivo.
    G.graph["es_real"] = True

    # Nos aseguramos de que todas las aristas tengan 'length' (osmnx ya lo
    # rellena al descargar, pero por si acaso tras el convert.to_undirected
    # alguna arista quedara sin el atributo, lo recalculamos como último
    # recurso a partir de las coordenadas).
    for u, v, datos in G.edges(data=True):
        if "length" not in datos:
            import numpy as np

            lon_u, lat_u = G.nodes[u]["pos"]
            lon_v, lat_v = G.nodes[v]["pos"]
            datos["length"] = float(np.hypot(lon_u - lon_v, lat_u - lat_v) * 111_000)

    # Nos quedamos con la componente conexa más grande (una red vial real
    # puede tener pequeños fragmentos inconexos, p.ej. una calle privada).
    if not nx.is_connected(G):
        componente = max(nx.connected_components(G), key=len)
        G = G.subgraph(componente).copy()

    return G


def generar_problema_real(
    n_candidatos: int,
    n_subestaciones: int,
    lugar: str = None,
    punto: tuple = None,
    distancia_metros: float = 1000.0,
    tipo_red: str = "drive",
    semilla: int = None,
    **kwargs,
) -> ProblemaCarga:
    """Atajo equivalente a problema.generar_problema_sintetico() pero para
    una zona real: descarga el grafo con cargar_grafo_real() y genera sobre
    él los parámetros del problema (demanda, costes, capacidades,
    subestaciones, presupuesto) con la MISMA función que se usa para los
    grafos sintéticos, para que el algoritmo genético no tenga que cambiar
    ni una línea al pasar de pruebas reducidas a una ciudad real.
    """
    grafo = cargar_grafo_real(
        lugar=lugar, punto=punto, distancia_metros=distancia_metros, tipo_red=tipo_red
    )
    return generar_parametros_problema(
        grafo,
        n_candidatos=n_candidatos,
        n_subestaciones=n_subestaciones,
        semilla=semilla,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Guardado / carga de grafos reales en disco (GraphML limpio)
# ----------------------------------------------------------------------
# Motivo: descargar de OpenStreetMap necesita Internet y osmnx, y el
# resultado puede variar entre descargas. Se descarga UNA vez en el PC
# (preparar_grafos.py), se guarda como GraphML y el servidor solo necesita
# networkx para cargarlo: instancias idénticas y reproducibles.
def guardar_grafo_real(G: nx.Graph, ruta: str) -> None:
    """Guarda un grafo producido por cargar_grafo_real() como GraphML limpio
    (solo x, y en nodos y length en aristas; GraphML no admite tuplas ni
    listas, que son los atributos que osmnx añade)."""
    import os

    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    L = nx.Graph()
    for n, d in G.nodes(data=True):
        x, y = d["pos"]
        L.add_node(int(n), x=float(x), y=float(y))
    for u, v, d in G.edges(data=True):
        L.add_edge(int(u), int(v), length=float(d["length"]))
    nx.write_graphml(L, ruta)


def cargar_grafo_guardado(ruta: str) -> nx.Graph:
    """Lee un GraphML creado con guardar_grafo_real() y lo deja en el mismo
    formato que cargar_grafo_real(): 'pos'=(lon,lat), 'length' y es_real=True.
    No necesita osmnx."""
    G = nx.read_graphml(ruta, node_type=int)
    for n, d in G.nodes(data=True):
        d["pos"] = (float(d["x"]), float(d["y"]))
    for _, _, d in G.edges(data=True):
        d["length"] = float(d["length"])
    G.graph["es_real"] = True
    return G
