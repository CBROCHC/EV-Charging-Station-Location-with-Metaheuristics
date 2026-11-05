"""
problema.py
===========

Define la estructura de datos del problema de localización de estaciones
de carga para vehículos eléctricos (tal y como aparece en el enunciado del
proyecto) y proporciona un generador de grafos y problemas SINTÉTICOS de
tamaño reducido, pensados para depurar y validar el algoritmo genético
antes de pasar a grafos de ciudades reales (ver grafo_real.py).

Correspondencia con la formulación matemática del enunciado:

    V                -> problema.grafo.nodes
    E                -> problema.grafo.edges (con atributo 'length' = l_e)
    J                -> problema.candidatos
    T                -> problema.tipos  ('lenta', 'rapida')
    K, g(j)          -> problema.subestacion[j] = k
    w_i              -> problema.demanda[i]
    c_j^t            -> problema.coste[(j, t)]
    cap_j^t          -> problema.capacidad[(j, t)]
    R^t              -> problema.radio[t]
    B                -> problema.presupuesto
    Cap_k            -> problema.capacidad_subestacion[k]

Las distancias d_G(i,j) NO se guardan aquí: se calculan aparte con Dijkstra
en distancias.py, porque dependen únicamente del grafo y se reutilizan en
todas las evaluaciones de fitness (no tiene sentido recalcularlas por cada
individuo del algoritmo genético).
"""

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np

# Tipo genérico para identificar nodos del grafo (puede ser un entero, un
# id de OSM, una tupla de coordenadas, etc.)
Nodo = Hashable


@dataclass
class ProblemaCarga:
    """Contiene TODOS los conjuntos y parámetros del modelo matemático.

    Esta clase es agnóstica de si el grafo es sintético o proviene de un
    mapa real (OpenStreetMap): lo único que necesita es un nx.Graph con
    un atributo de arista 'length' (l_e, en metros) y, opcionalmente, un
    atributo de nodo 'pos' con coordenadas (x, y) o (lon, lat) para poder
    dibujarlo después.
    """

    grafo: nx.Graph
    candidatos: List[Nodo]                       # J
    tipos: Tuple[str, str] = ("lenta", "rapida")  # T
    demanda: Dict[Nodo, float] = field(default_factory=dict)          # w_i
    coste: Dict[Tuple[Nodo, str], float] = field(default_factory=dict)     # c_j^t
    capacidad: Dict[Tuple[Nodo, str], float] = field(default_factory=dict)  # cap_j^t
    radio: Dict[str, float] = field(default_factory=dict)             # R^t
    presupuesto: float = 0.0                                          # B
    subestacion: Dict[Nodo, int] = field(default_factory=dict)        # g(j)
    capacidad_subestacion: Dict[int, float] = field(default_factory=dict)  # Cap_k
    es_real: bool = False  # True si el grafo viene de grafo_real.cargar_grafo_real (coordenadas
                            # 'pos' = (lon, lat) reales); se usa para decidir automáticamente si
                            # generar también el mapa interactivo (ver main.py y visualizacion.py)

    # ------------------------------------------------------------------
    def validar(self) -> None:
        """Comprueba que los datos son internamente consistentes.

        No valida factibilidad de ninguna solución, solo que el problema
        está bien formado (todas las tablas tienen entradas para todos los
        candidatos y tipos, etc.). Lanza AssertionError con un mensaje
        claro si algo falta, para detectar errores de generación pronto.
        """
        nodos = set(self.grafo.nodes)
        assert set(self.candidatos).issubset(nodos), (
            "Hay candidatos que no son nodos del grafo"
        )
        for j in self.candidatos:
            assert j in self.subestacion, f"Candidato {j} sin subestación asignada (g(j))"
            for t in self.tipos:
                assert (j, t) in self.coste, f"Falta coste c_{j}^{t}"
                assert (j, t) in self.capacidad, f"Falta capacidad cap_{j}^{t}"
        for t in self.tipos:
            assert t in self.radio, f"Falta radio de cobertura R^{t}"
        for i in nodos:
            assert i in self.demanda, f"Falta demanda w_{i}"
        for k in set(self.subestacion.values()):
            assert k in self.capacidad_subestacion, f"Falta Cap_{k}"

    # ------------------------------------------------------------------
    def resumen(self) -> str:
        """Devuelve una cadena legible con el tamaño del problema."""
        return (
            f"Nodos |V|={self.grafo.number_of_nodes()}, "
            f"Aristas |E|={self.grafo.number_of_edges()}, "
            f"Candidatos |J|={len(self.candidatos)}, "
            f"Subestaciones |K|={len(self.capacidad_subestacion)}, "
            f"Presupuesto B={self.presupuesto:.1f},"
            f"Demanda total={sum(self.demanda.values()):.1f}"
        )


# ----------------------------------------------------------------------
# Generación de un grafo SINTÉTICO reducido (sin depender de Internet)
# ----------------------------------------------------------------------
def generar_grafo_sintetico(
    n_nodos: int = 25,
    radio_conexion: float = 0.1,
    escala_metros: float = 1200.0,
    semilla: int = None,
) -> nx.Graph:
    """Genera un grafo geométrico aleatorio pequeño que hace de "ciudad de
    prueba": nodos con coordenadas 2D aleatorias en un cuadrado, conectados
    si están a menos de `radio_conexion` (en coordenadas normalizadas
    [0,1]x[0,1]). Se usa como sustituto barato y reproducible de un barrio
    real mientras se depura el algoritmo genético.

    Parámetros
    ----------
    n_nodos: número de intersecciones (|V|).
    radio_conexion: distancia máxima (en [0,1]) para que dos nodos estén
        conectados por una calle. Cuanto mayor, más denso el grafo.
    escala_metros: factor para convertir las coordenadas normalizadas en
        una longitud de arista realista (metros), guardada en 'length'.
    semilla: semilla aleatoria para reproducibilidad.

    Devuelve
    --------
    Un nx.Graph conexo (si el grafo generado queda desconectado, se
    reintenta con más radio de conexión) con:
      - atributo de nodo 'pos' = (x, y) en [0,1]^2
      - atributo de arista 'length' = l_e en metros (>0)
    """
    rng_seed = semilla
    radio = radio_conexion
    intentos = 0
    while True:
        G = nx.random_geometric_graph(n_nodos, radio, seed=rng_seed)
        if nx.is_connected(G):
            break
        # Si no queda conexo, aumentamos un poco el radio y reintentamos.
        radio *= 1.15
        intentos += 1
        if intentos > 30:
            # Como último recurso, nos quedamos con la componente conexa
            # más grande (no debería hacer falta llegar aquí en la práctica).
            componente = max(nx.connected_components(G), key=len)
            G = G.subgraph(componente).copy()
            break

    # 'pos' ya lo pone random_geometric_graph; añadimos 'length' a las aristas
    # como distancia euclídea escalada a metros (l_e del enunciado).
    for u, v in G.edges():
        xu, yu = G.nodes[u]["pos"]
        xv, yv = G.nodes[v]["pos"]
        dist = np.hypot(xu - xv, yu - yv) * escala_metros
        G.edges[u, v]["length"] = float(dist)

    # nx.random_geometric_graph ya numera los nodos como enteros 0..n-1 y
    # deja el atributo 'pos' en cada uno, así que no hace falta relabeling.
    return G


# ----------------------------------------------------------------------
# Generadores auxiliares de parámetros (demanda, subestaciones, calibrado)
# ----------------------------------------------------------------------
def _generar_demanda_con_hotspots(
    grafo: nx.Graph,
    nodos: List[Nodo],
    rng: np.random.Generator,
    fraccion_hotspots: float,
    sigma_rango: Tuple[float, float] = (120.0, 300.0),
) -> Dict[Nodo, float]:
    """Demanda ESPACIALMENTE agrupada: una demanda base pequeña en todos los
    nodos más varios focos ("hotspots": plazas, centros comerciales...) cuya
    demanda decae de forma gaussiana con la distancia por calle al centro del
    foco. Así la posición de las estaciones importa de verdad (no todas
    quedan saturadas), y los solapes de cobertura también.

    Usa distancias de camino más corto (atributo 'length'), por lo que
    funciona igual con grafos sintéticos y reales.
    """
    demanda = {i: float(rng.uniform(1.0, 5.0)) for i in nodos}
    n_hot = max(3, int(round(fraccion_hotspots * len(nodos))))
    idx_centros = rng.choice(len(nodos), size=min(n_hot, len(nodos)), replace=False)
    for idx in idx_centros:
        centro = nodos[int(idx)]
        sigma = float(rng.uniform(*sigma_rango))
        amplitud = float(rng.uniform(40.0, 120.0))
        dists = nx.single_source_dijkstra_path_length(
            grafo, centro, cutoff=3.0 * sigma, weight="length"
        )
        for i, d in dists.items():
            demanda[i] += amplitud * float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
    return demanda


def _asignar_subestaciones_por_cercania(
    grafo: nx.Graph,
    candidatos: List[Nodo],
    n_subestaciones: int,
    rng: np.random.Generator,
) -> Dict[Nodo, int]:
    """g(j): agrupa los candidatos en `n_subestaciones` barrios con k-means
    (inicialización k-means++) sobre sus coordenadas 'pos', de modo que cada
    subestación alimenta a candidatos CERCANOS entre sí (un barrio), en lugar
    de candidatos repartidos al azar por toda la ciudad."""
    pos = np.array([grafo.nodes[j]["pos"] for j in candidatos], dtype=float)
    if grafo.graph.get("es_real", False):
        # (lon, lat) -> aproximadamente metros relativos: corrige el
        # encogimiento de la longitud con la latitud.
        pos[:, 0] *= np.cos(np.radians(pos[:, 1].mean()))
    k = max(1, min(n_subestaciones, len(candidatos)))

    centros = [pos[int(rng.integers(len(pos)))]]
    for _ in range(1, k):
        d2 = np.min([((pos - c) ** 2).sum(axis=1) for c in centros], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centros.append(pos[int(rng.choice(len(pos), p=p))])
    centros = np.array(centros)

    for _ in range(30):
        etiquetas = ((pos[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        nuevos = np.array([
            pos[etiquetas == c].mean(axis=0) if (etiquetas == c).any() else centros[c]
            for c in range(k)
        ])
        if np.allclose(nuevos, centros):
            break
        centros = nuevos
    etiquetas = ((pos[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return {int(j): int(e) for j, e in zip(candidatos, etiquetas)}


def _capacidad_comprable_con_presupuesto(
    candidatos: List[Nodo],
    coste: Dict[Tuple[Nodo, str], float],
    capacidad: Dict[Tuple[Nodo, str], float],
    tipos: Tuple[str, ...],
    presupuesto: float,
) -> float:
    """Cota superior (mochila fraccionaria) de la capacidad total que el
    presupuesto B permitiría instalar, ignorando subestaciones y cobertura.
    Sirve para CALIBRAR demanda y Cap_k respecto a lo que realmente se puede
    comprar."""
    items = []
    for j in candidatos:
        # Por candidato, el tipo con mejor ratio capacidad/coste.
        t = max(tipos, key=lambda tt: capacidad[(j, tt)] / coste[(j, tt)])
        items.append((capacidad[(j, t)] / coste[(j, t)], coste[(j, t)], capacidad[(j, t)]))
    items.sort(reverse=True)
    restante, total = presupuesto, 0.0
    for _, c, cap in items:
        if c <= restante:
            restante -= c
            total += cap
        else:
            total += cap * (restante / c)
            break
    return total


# ----------------------------------------------------------------------
# Generación de los PARÁMETROS del problema sobre un grafo ya construido
# ----------------------------------------------------------------------
def generar_parametros_problema(
    grafo: nx.Graph,
    n_candidatos: int,
    n_subestaciones: int,
    fraccion_presupuesto: float = 0.2,
    fraccion_hotspots: float = 0.03,
    radio_lenta: float = 120.0,
    radio_rapida: float = 220.0,
    ratio_demanda_capacidad: float = 0.8,
    rango_cap_subestaciones: Tuple[float, float] = (0.9, 1.1),
    semilla: int = None,
) -> ProblemaCarga:
    """Genera de forma aleatoria (pero CALIBRADA) todos los parámetros del
    modelo sobre un grafo ya existente (sintético o real).

    Parámetros
    ----------
    grafo: nx.Graph con atributo de arista 'length' (metros) y de nodo 'pos'.
    n_candidatos: tamaño de J (se eligen al azar entre los nodos de V).
    n_subestaciones: tamaño de K. Los candidatos se asignan a la subestación
        de su barrio (k-means geográfico), no al azar.
    fraccion_presupuesto: B = esta fracción del coste de abrir TODOS los
        candidatos en versión rápida.
    fraccion_hotspots: fracción de nodos que son centros de un foco de
        demanda (la demanda decae gaussianamente a su alrededor).
    radio_lenta, radio_rapida: R^t en metros. Deben ser pequeños frente al
        tamaño del grafo para que ~40-60 estaciones NO lo cubran entero.
    ratio_demanda_capacidad: demanda total = ratio x capacidad que el
        presupuesto puede comprar (cota de mochila fraccionaria, que es
        OPTIMISTA porque ignora subestaciones y usa siempre el mejor tipo).
        Con ~0.8 la demanda queda algo por debajo de la capacidad instalable
        y las estaciones no se saturan todas: importa DÓNDE se ponen.
    rango_cap_subestaciones: sum_k Cap_k = U(rango) x esa misma cota,
        repartida de forma heterogénea entre subestaciones. Con ~1.0 el
        presupuesto y las subestaciones compiten como límites.
    semilla: semilla aleatoria.
    """
    rng = np.random.default_rng(semilla)
    nodos = list(grafo.nodes())
    if n_candidatos > len(nodos):
        n_candidatos = len(nodos)
        print(f"Advertencia: n_candidatos > |V|, se ajusta a {n_candidatos}")

    # int nativo de Python (rng.choice devuelve numpy.int64, que confunde a
    # networkx al mezclarse con los ids enteros del grafo).
    candidatos = [int(c) for c in rng.choice(nodos, size=n_candidatos, replace=False)]

    radio = {"lenta": float(radio_lenta), "rapida": float(radio_rapida)}

    # --- Coste y capacidad con FACTOR DE TERRENO compartido por ambos tipos
    # (un sitio caro lo es para lenta y rápida). La rápida cuesta ~2.5x y da
    # ~2x de capacidad (peor ratio cap/coste que la lenta) pero cubre un
    # radio mayor: lenta y rápida compiten de verdad (capacidad por euro vs.
    # cobertura).
    coste: Dict[Tuple[Nodo, str], float] = {}
    capacidad: Dict[Tuple[Nodo, str], float] = {}
    for j in candidatos:
        terreno = float(rng.uniform(0.8, 1.3))
        coste[(j, "lenta")] = 16_000.0 * terreno * float(rng.uniform(0.95, 1.05))
        coste[(j, "rapida")] = 40_000.0 * terreno * float(rng.uniform(0.95, 1.05))
        capacidad[(j, "lenta")] = 20.0 * float(rng.uniform(0.85, 1.15))
        capacidad[(j, "rapida")] = 40.0 * float(rng.uniform(0.85, 1.15))

    # --- Presupuesto B.
    coste_maximo_total = sum(coste[(j, "rapida")] for j in candidatos)
    presupuesto = float(coste_maximo_total * fraccion_presupuesto)

    # --- Calibrado respecto a lo que el presupuesto puede comprar.
    tipos = ("lenta", "rapida")
    cap_comprable = _capacidad_comprable_con_presupuesto(
        candidatos, coste, capacidad, tipos, presupuesto
    )

    # --- Demanda agrupada en focos, reescalada a ratio x cap_comprable.
    demanda = _generar_demanda_con_hotspots(grafo, nodos, rng, fraccion_hotspots)
    escala = ratio_demanda_capacidad * cap_comprable / sum(demanda.values())
    demanda = {i: float(w * escala) for i, w in demanda.items()}

    # --- Subestaciones por cercanía geográfica.
    subestacion = _asignar_subestaciones_por_cercania(grafo, candidatos, n_subestaciones, rng)
    ids_sub = list(range(n_subestaciones))

    # --- Cap_k heterogénea: sum_k Cap_k = U(rango) x cap_comprable, repartido
    # según (nº de candidatos de la subestación) x (peso aleatorio 0.5-1.5).
    pesos = {
        k: sum(1 for j in candidatos if subestacion[j] == k) * float(rng.uniform(0.5, 1.5))
        for k in ids_sub
    }
    total_pesos = sum(pesos.values())
    objetivo_total = float(rng.uniform(*rango_cap_subestaciones)) * cap_comprable
    capacidad_subestacion: Dict[int, float] = {}
    for k in ids_sub:
        miembros = [j for j in candidatos if subestacion[j] == k]
        if not miembros:
            capacidad_subestacion[k] = 0.0
            continue
        cap_k = objetivo_total * pesos[k] / total_pesos
        # Suelo: al menos cabe una estación lenta en la subestación.
        cap_k = max(cap_k, min(capacidad[(j, "lenta")] for j in miembros))
        capacidad_subestacion[k] = float(cap_k)

    problema = ProblemaCarga(
        grafo=grafo,
        candidatos=candidatos,
        tipos=tipos,
        demanda=demanda,
        coste=coste,
        capacidad=capacidad,
        radio=radio,
        presupuesto=presupuesto,
        subestacion=subestacion,
        capacidad_subestacion=capacidad_subestacion,
        es_real=bool(grafo.graph.get("es_real", False)),
    )
    problema.validar()
    return problema


def generar_problema_sintetico(
    n_nodos: int = 25,
    n_candidatos: int = 10,
    n_subestaciones: int = 3,
    semilla: int = None,
    **kwargs,
) -> ProblemaCarga:
    """Atajo: genera un grafo sintético reducido y, sobre él, un problema
    completo con generar_parametros_problema. Es el punto de entrada que
    se usa para las pruebas rápidas del algoritmo genético (main.py).
    """
    grafo = generar_grafo_sintetico(n_nodos=n_nodos, semilla=semilla)
    return generar_parametros_problema(
        grafo,
        n_candidatos=n_candidatos,
        n_subestaciones=n_subestaciones,
        semilla=semilla,
        **kwargs,
    )