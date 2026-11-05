"""
visualizacion.py
=================

Cuatro bloques:

1. dibujar_grafo_solucion: grafo (sintético o real, en su plano de
   coordenadas propio) con la solución del AG, vía matplotlib.

2. Gráficas de ANÁLISIS del algoritmo genético (todas reciben un
   `ResultadoAG` de algoritmo_genetico.py): convergencia (con banda
   min-max), diversidad de la población, evolución de la composición de
   la solución, uso de presupuesto/capacidad de subestación, fitness vs
   nº de evaluaciones (soluciones generadas) y tiempo de cómputo, más un
   histograma de la población final. `generar_todas_las_graficas` las
   genera todas de una vez.

3. generar_explorador_generaciones: HTML autocontenido con un slider para
   recorrer las generaciones y ver, para cada una, la solución (mejor
   individuo encontrado hasta esa generación).

4. dibujar_mapa_real: mapa interactivo (folium) sobre un mapa real, con
   TODOS los nodos del grafo marcados, aristas bien visibles, y capas
   conmutables (calles / nodos / estaciones / cobertura / asignaciones)
   para poder simplificar la vista según convenga.
"""

import base64
import io
import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla, para poder generar PNG en servidor
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from problema import ProblemaCarga, Nodo
from distancias import TablaDistancias
from codificacion import decodificar_individuo, coste_total, uso_por_subestacion, Individuo
from fitness import evaluar_x
from algoritmo_genetico import ResultadoAG

COLOR_SIN_ESTACION = "#B0B0B0"      # candidato no elegido
COLOR_LENTA = "#F2A93B"            # estación lenta
COLOR_RAPIDA = "#D6432D"           # estación rápida
COLOR_NODO_NORMAL = "#8FA8C4"      # nodo normal (no candidato)


def _color_por_gen(gen: int) -> str:
    return {0: COLOR_SIN_ESTACION, 1: COLOR_LENTA, 2: COLOR_RAPIDA}[int(gen)]


# ----------------------------------------------------------------------
# 1. Grafo con la solución (matplotlib) - válido para sintético o real
# ----------------------------------------------------------------------
def _dibujar_solucion_en_ejes(ax, problema: ProblemaCarga, individuo: Individuo, distancias: TablaDistancias):
    """Lógica de dibujo compartida entre dibujar_grafo_solucion (a fichero)
    y el explorador de generaciones (a memoria, para el slider)."""
    x = decodificar_individuo(individuo, problema)
    fitness, y = evaluar_x(x, problema, distancias, devolver_flujo=True)

    pos = nx.get_node_attributes(problema.grafo, "pos")
    nx.draw_networkx_edges(problema.grafo, pos, ax=ax, edge_color="#DDDDDD", width=1.0)

    nodos_normales = [n for n in problema.grafo.nodes() if n not in problema.candidatos]
    nx.draw_networkx_nodes(
        problema.grafo, pos, nodelist=nodos_normales, node_size=40,
        node_color=COLOR_NODO_NORMAL, ax=ax, linewidths=0,
    )

    for gen_valor in (0, 1, 2):
        nodos_gen = [j for j, g in zip(problema.candidatos, individuo) if int(g) == gen_valor]
        if nodos_gen:
            nx.draw_networkx_nodes(
                problema.grafo, pos, nodelist=nodos_gen,
                node_size=220 if gen_valor else 120,
                node_color=_color_por_gen(gen_valor),
                edgecolors="black", linewidths=1.0, ax=ax,
            )

    max_flujo = max(y.values()) if y else 1.0
    for (i, j, _t), cantidad in y.items():
        if i == j:
            continue
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot(
            [xi, xj], [yi, yj], linestyle="--", color="#555555", alpha=0.5,
            linewidth=0.5 + 3.0 * (cantidad / max_flujo), zorder=1,
        )
    ax.set_axis_off()
    return fitness


def dibujar_grafo_solucion(
    problema: ProblemaCarga,
    individuo: Individuo,
    distancias: TablaDistancias,
    ruta_salida: str = "solucion_grafo.png",
) -> str:
    """Dibuja el grafo completo con la solución del algoritmo genético y lo
    guarda en `ruta_salida` (PNG)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    fitness = _dibujar_solucion_en_ejes(ax, problema, individuo, distancias)

    leyenda = [
        plt.Line2D([0], [0], marker="o", color="w", label="Nodo normal",
                    markerfacecolor=COLOR_NODO_NORMAL, markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Candidato sin estación",
                    markerfacecolor=COLOR_SIN_ESTACION, markersize=10, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", label="Estación lenta",
                    markerfacecolor=COLOR_LENTA, markersize=12, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", label="Estación rápida",
                    markerfacecolor=COLOR_RAPIDA, markersize=12, markeredgecolor="black"),
    ]
    ax.legend(handles=leyenda, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"Solución del AG — demanda atendida: {fitness:.1f}")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


# ----------------------------------------------------------------------
# 2. Gráficas de análisis del algoritmo genético
# ----------------------------------------------------------------------
def dibujar_convergencia(resultado: ResultadoAG, ruta_salida: str = "convergencia.png") -> str:
    """Mejor fitness histórico + fitness medio, con una banda sombreada
    entre el mínimo y el máximo de la población en cada generación (para
    ver de un vistazo tanto la tendencia como la dispersión)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    gens = range(len(resultado.historial_mejor))
    ax.fill_between(gens, resultado.historial_min, resultado.historial_max,
                     color="#3B6FF2", alpha=0.15, label="Rango [mín, máx] de la población")
    ax.plot(gens, resultado.historial_medio, label="Fitness medio", color="#3B6FF2", linewidth=1.5)
    ax.plot(gens, resultado.historial_mejor, label="Mejor histórico", color="#D6432D", linewidth=2)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Demanda total atendida (fitness)")
    ax.set_title("Convergencia del algoritmo genético")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def dibujar_diversidad(resultado: ResultadoAG, ruta_salida: str = "diversidad.png") -> str:
    """Diversidad GENOTÍPICA de la población por generación:
      - eje izquierdo: distancia de Hamming media normalizada entre pares de
        individuos (% de genes en que difieren dos individuos cualesquiera);
      - eje derecho: nº de genotipos distintos en la población.
    A diferencia de la desviación del fitness, detecta población colapsada
    en clones aunque sus fitness sean distintos o casi iguales."""
    fig, ax = plt.subplots(figsize=(8, 4))
    gens = range(len(resultado.historial_hamming))
    ax.plot(gens, [100 * h for h in resultado.historial_hamming],
            color="#7A3BF2", label="Distancia de Hamming media (%)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Distancia de Hamming media (% de genes)", color="#7A3BF2")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(gens, resultado.historial_genotipos_distintos,
             color="#2A9D8F", linestyle="--", label="Genotipos distintos")
    ax2.set_ylabel("Genotipos distintos", color="#2A9D8F")
    ax2.set_ylim(bottom=0)

    ax.set_title("Diversidad genotípica de la población")
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def dibujar_evolucion_estaciones(
    problema: ProblemaCarga, resultado: ResultadoAG, ruta_salida: str = "evolucion_estaciones.png"
) -> str:
    """Nº de estaciones lentas y rápidas del mejor individuo (histórico),
    por generación: muestra cómo cambia la COMPOSICIÓN de la solución
    mientras el AG evoluciona, no solo su fitness."""
    n_lenta = [int(np.sum(ind == 1)) for ind in resultado.historial_individuos]
    n_rapida = [int(np.sum(ind == 2)) for ind in resultado.historial_individuos]
    gens = range(len(resultado.historial_individuos))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, n_lenta, label="Estaciones lentas", color=COLOR_LENTA)
    ax.plot(gens, n_rapida, label="Estaciones rápidas", color=COLOR_RAPIDA)
    ax.plot(gens, np.array(n_lenta) + np.array(n_rapida), label="Total instaladas",
            color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Nº de estaciones (mejor individuo)")
    ax.set_title("Evolución de la composición de la solución")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def dibujar_uso_recursos(
    problema: ProblemaCarga, resultado: ResultadoAG, ruta_salida: str = "uso_recursos.png"
) -> str:
    """% de presupuesto usado y % medio de capacidad de subestación usada
    por el mejor individuo, por generación: muestra si el AG realmente
    está apurando los recursos disponibles o dejando margen sin usar."""
    pct_presupuesto, pct_subestaciones = [], []
    for ind in resultado.historial_individuos:
        x = decodificar_individuo(ind, problema)
        pct_presupuesto.append(100 * coste_total(x, problema) / problema.presupuesto)
        uso = uso_por_subestacion(x, problema)
        pcts_k = [
            100 * uso[k] / problema.capacidad_subestacion[k]
            for k in uso if problema.capacidad_subestacion[k] > 0
        ]
        pct_subestaciones.append(float(np.mean(pcts_k)) if pcts_k else 0.0)

    gens = range(len(resultado.historial_individuos))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, pct_presupuesto, label="% presupuesto usado", color="#2E8B57")
    ax.plot(gens, pct_subestaciones, label="% capacidad media de subestación usada", color="#B8860B")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Generación")
    ax.set_ylabel("% usado")
    ax.set_ylim(0, 110)
    ax.set_title("Uso de recursos del mejor individuo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def dibujar_fitness_vs_evaluaciones(resultado: ResultadoAG, ruta_salida: str = "fitness_vs_evaluaciones.png") -> str:
    """Fitness vs Nº ACUMULADO de evaluaciones (soluciones generadas), en
    vez de vs generación: permite comparar configuraciones con distinto
    tamaño de población en igualdad de condiciones (el guion del curso lo
    pide explícitamente: "Fitness versus Soluciones generadas")."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(resultado.historial_evaluaciones, resultado.historial_mejor, color="#D6432D")
    ax.set_xlabel("Nº acumulado de evaluaciones de fitness (soluciones generadas)")
    ax.set_ylabel("Mejor fitness")
    ax.set_title("Fitness vs soluciones generadas")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def dibujar_tiempo_computo(resultado: ResultadoAG, ruta_salida: str = "tiempo_computo.png") -> str:
    """Tiempo acumulado (segundos) por generación: para el análisis de
    "tiempo de cómputo" del guion del curso, y para comparar el coste real
    de distintos tamaños de problema/población."""
    fig, ax = plt.subplots(figsize=(8, 4))
    gens = range(len(resultado.historial_tiempo))
    ax.plot(gens, resultado.historial_tiempo, color="#555555")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Tiempo acumulado (s)")
    ax.set_title(
        f"Tiempo de cómputo (total: {resultado.historial_tiempo[-1]:.1f} s, "
        f"{resultado.generaciones_ejecutadas} generaciones)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def dibujar_histograma_poblacion_final(resultado: ResultadoAG, ruta_salida: str = "histograma_final.png") -> str:
    """Distribución del fitness en la ÚLTIMA población: da una idea de
    cuánta diversidad de soluciones (buenas y menos buenas) queda al
    terminar la ejecución.

    Caso especial: si toda la población ha convergido al mismo óptimo
    (típico cuando sobra presupuesto/capacidad y se cubre el 100% de la
    demanda), los distintos individuos pueden dar un fitness que
    matemáticamente es idéntico pero que el cálculo de flujo máximo
    arrastra como números de coma flotante casi-pero-no-exactamente
    iguales (diferencias del orden de 1e-10 o menores). Repartir ese rango
    minúsculo en 15 bins hace que numpy no tenga precisión suficiente y
    falle con "Too many bins for data range" -> se detecta este caso y se
    dibuja una única barra en el valor común en lugar de un histograma.
    """
    datos = np.asarray(resultado.fitness_poblacion_final, dtype=float)
    rango = float(datos.max() - datos.min())
    escala = max(1.0, float(abs(datos.mean())))

    fig, ax = plt.subplots(figsize=(7, 4))
    if rango < 1e-6 * escala:
        # Toda la población converge (a efectos prácticos) al mismo fitness.
        ax.bar([0], [len(datos)], color="#3B6FF2", edgecolor="black", width=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([f"{datos.mean():.2f}"])
        ax.set_xlabel("Fitness (toda la población converge a este valor)")
    else:
        ax.hist(datos, bins=15, color="#3B6FF2", edgecolor="black", alpha=0.85)
        ax.axvline(resultado.mejor_fitness, color="#D6432D", linewidth=2, label="Mejor individuo")
        ax.legend()
        ax.set_xlabel("Fitness")
    ax.set_ylabel("Nº de individuos")
    ax.set_title("Distribución de fitness — población final")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return ruta_salida


def generar_todas_las_graficas(
    problema: ProblemaCarga, resultado: ResultadoAG, carpeta_salida: str = "."
) -> dict:
    """Genera de una vez toda la batería de gráficas de análisis y
    devuelve un diccionario {nombre: ruta_del_fichero}."""
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta = lambda nombre: os.path.join(carpeta_salida, nombre)
    return {
        "convergencia": dibujar_convergencia(resultado, ruta("convergencia.png")),
        "diversidad": dibujar_diversidad(resultado, ruta("diversidad.png")),
        "evolucion_estaciones": dibujar_evolucion_estaciones(problema, resultado, ruta("evolucion_estaciones.png")),
        "uso_recursos": dibujar_uso_recursos(problema, resultado, ruta("uso_recursos.png")),
        "fitness_vs_evaluaciones": dibujar_fitness_vs_evaluaciones(resultado, ruta("fitness_vs_evaluaciones.png")),
        "tiempo_computo": dibujar_tiempo_computo(resultado, ruta("tiempo_computo.png")),
        "histograma_final": dibujar_histograma_poblacion_final(resultado, ruta("histograma_final.png")),
    }


# ----------------------------------------------------------------------
# 3. Explorador interactivo de generaciones (HTML con slider)
# ----------------------------------------------------------------------
def _renderizar_png_base64(problema: ProblemaCarga, individuo: Individuo, distancias: TablaDistancias) -> str:
    """Dibuja la solución en memoria (sin pasar por disco) y la devuelve
    como data-URI base64, para incrustarla directamente en un HTML."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _dibujar_solucion_en_ejes(ax, problema, individuo, distancias)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def generar_explorador_generaciones(
    problema: ProblemaCarga,
    resultado: ResultadoAG,
    distancias: TablaDistancias,
    ruta_salida: str = "explorador_generaciones.html",
    paso: int = 5,
) -> str:
    """Genera un HTML autocontenido (sin dependencias externas) con un
    slider para recorrer las generaciones y ver, en cada una, la solución
    (mejor individuo encontrado hasta esa generación).

    Por eficiencia NO se renderiza una imagen por cada generación (podrían
    ser cientos): se toma una de cada `paso` generaciones, incluyendo
    siempre la primera y la última. Con `paso=1` se incluyen todas.
    """
    n = len(resultado.historial_individuos)
    indices = sorted(set(list(range(0, n, paso)) + [n - 1]))

    imagenes: List[str] = []
    info: List[dict] = []
    for g in indices:
        individuo = resultado.historial_individuos[g]
        imagenes.append(_renderizar_png_base64(problema, individuo, distancias))
        info.append({
            "gen": g,
            "fitness": round(float(resultado.historial_mejor[g]), 2),
            "lenta": int(np.sum(individuo == 1)),
            "rapida": int(np.sum(individuo == 2)),
        })

    html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Explorador de generaciones</title>
<style>
  body {{ font-family: system-ui, sans-serif; text-align: center; background: #fafafa; padding: 24px; }}
  img {{ max-width: 640px; width: 100%; border: 1px solid #ccc; border-radius: 8px; background: white; }}
  #info {{ margin: 14px 0; font-size: 17px; }}
  input[type=range] {{ width: 60%; max-width: 500px; }}
  .valor {{ font-weight: bold; color: #D6432D; }}
  h2 {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<h2>Explorador de generaciones del algoritmo genético</h2>
<p>Arrastra el control para ver la mejor solución encontrada hasta cada generación.</p>
<div id="info">
  Generación <span id="gen" class="valor">-</span> &nbsp;|&nbsp;
  Fitness: <span id="fitness" class="valor">-</span> &nbsp;|&nbsp;
  Lentas: <span id="lenta" class="valor">-</span> &nbsp;|&nbsp;
  Rápidas: <span id="rapida" class="valor">-</span>
</div>
<input type="range" id="slider" min="0" max="{n_frames}" value="0" step="1">
<br><br>
<img id="imagen" alt="Solución de la generación seleccionada">
<script>
  const imagenes = {imagenes_json};
  const info = {info_json};
  const slider = document.getElementById('slider');
  const imagen = document.getElementById('imagen');
  function actualizar(i) {{
    imagen.src = imagenes[i];
    document.getElementById('gen').textContent = info[i].gen;
    document.getElementById('fitness').textContent = info[i].fitness;
    document.getElementById('lenta').textContent = info[i].lenta;
    document.getElementById('rapida').textContent = info[i].rapida;
  }}
  slider.addEventListener('input', () => actualizar(parseInt(slider.value, 10)));
  actualizar(0);
</script>
</body>
</html>""".format(
        n_frames=len(indices) - 1,
        imagenes_json=json.dumps(imagenes),
        info_json=json.dumps(info),
    )

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html)
    return ruta_salida


# ----------------------------------------------------------------------
# 4. Mapa real interactivo (folium) con capas conmutables
# ----------------------------------------------------------------------
def _resolver_tiles(estilo_mapa: str, clave_carto: str = None):
    """Traduce el nombre corto de estilo a lo que espera folium.

    IMPORTANTE (descubierto probando en este mismo proyecto): desde hace
    poco, CARTO exige una API key gratuita para sus basemaps
    ("CartoDB Positron"/"dark_matter") — xyzservices avisa de ello con un
    UserWarning. Con clave_carto=None las teselas oscuras de CARTO siguen
    cargando hoy (el aviso no es un error, solo una advertencia de que
    podría dejar de ser gratis sin registrar una clave en
    https://carto.com/basemaps/apikey/), pero para no depender de eso el
    estilo "claro" por defecto usa Esri World Gray Canvas: un basemap
    gris neutro pensado precisamente para resaltar overlays -sin
    necesitar ninguna clave y sin avisos-, que da un aspecto limpio muy
    parecido a Google Maps en modo claro.
    """
    if estilo_mapa == "claro":
        return "Esri WorldGrayCanvas"
    if estilo_mapa == "osm":
        return "OpenStreetMap"
    if estilo_mapa == "oscuro":
        if clave_carto:
            import xyzservices.providers as xyz
            return xyz.providers.CartoDB.DarkMatter(key=clave_carto)
        return "CartoDB dark_matter"
    raise ValueError(f"estilo_mapa debe ser uno de {list(_ESTILOS_MAPA)}")


_ESTILOS_MAPA = ("claro", "oscuro", "osm")


def dibujar_mapa_real(
    problema: ProblemaCarga,
    individuo: Individuo,
    distancias: TablaDistancias,
    ruta_salida: str = "solucion_mapa.html",
    estilo_mapa: str = "claro",
    clave_carto: str = None,
    mostrar_cobertura_por_defecto: bool = False,
) -> str:
    """Mapa interactivo (HTML, folium) con la solución superpuesta sobre un
    mapa real, organizado en CAPAS conmutables desde el propio mapa
    (control arriba a la derecha):

      - "Calles"                 -> aristas del grafo, en oscuro y grosor
                                     visible (antes casi se perdían contra
                                     las teselas del mapa base).
      - "Nodos"                  -> TODOS los nodos del grafo que no son
                                     candidatos, como puntos oscuros
                                     pequeños (antes solo se veían los
                                     candidatos).
      - "Candidatos sin estación", "Estaciones lentas", "Estaciones
        rápidas" -> una capa por estado, coloreadas igual que en
        dibujar_grafo_solucion.
      - "Cobertura"               -> círculos de radio R^t (oculta por
                                      defecto: en el mapa de una ciudad
                                      real se solapan mucho y ensucian la
                                      vista; actívala desde el control de
                                      capas si la necesitas).
      - "Asignaciones de demanda" -> líneas i -> estación con grosor ~ y_ij^t.

    `estilo_mapa`: "claro" (por defecto: Esri World Gray Canvas, un
    basemap gris neutro sin coste ni clave, pensado para que resalten los
    overlays -es la opción más parecida en limpieza visual a Google Maps),
    "osm" (teselas clásicas de OpenStreetMap, más recargadas visualmente),
    u "oscuro" (CartoDB Dark Matter; sigue funcionando sin clave hoy, pero
    CARTO ya avisa de que en el futuro puede exigir una API key gratuita —
    pásala en `clave_carto` si la tienes). No se pueden incrustar
    literalmente las teselas de Google Maps: desde 2018 requieren una
    clave de la Google Maps Platform (API de pago) y no se pueden usar
    libremente como capa de un mapa propio.
    """
    import folium

    if estilo_mapa not in _ESTILOS_MAPA:
        raise ValueError(f"estilo_mapa debe ser uno de {list(_ESTILOS_MAPA)}")
    tiles = _resolver_tiles(estilo_mapa, clave_carto)

    x = decodificar_individuo(individuo, problema)
    fitness, y = evaluar_x(x, problema, distancias, devolver_flujo=True)

    pos = nx.get_node_attributes(problema.grafo, "pos")  # (lon, lat)
    lats = [p[1] for p in pos.values()]
    lons = [p[0] for p in pos.values()]
    centro = (sum(lats) / len(lats), sum(lons) / len(lons))

    mapa = folium.Map(location=centro, zoom_start=15, tiles=tiles)

    capa_calles = folium.FeatureGroup(name="Calles", show=True)
    capa_nodos = folium.FeatureGroup(name="Nodos", show=True)
    capa_sin_estacion = folium.FeatureGroup(name="Candidatos sin estación", show=True)
    capa_lenta = folium.FeatureGroup(name="Estaciones lentas", show=True)
    capa_rapida = folium.FeatureGroup(name="Estaciones rápidas", show=True)
    capa_cobertura = folium.FeatureGroup(name="Cobertura (R^t)", show=mostrar_cobertura_por_defecto)
    capa_asignaciones = folium.FeatureGroup(name="Asignaciones de demanda", show=True)

    # --- Calles: aristas del grafo, en oscuro y bien visibles ---
    for u, v in problema.grafo.edges():
        lon_u, lat_u = pos[u]
        lon_v, lat_v = pos[v]
        folium.PolyLine(
            [(lat_u, lon_u), (lat_v, lon_v)], color="#2B2B2B", weight=2.2, opacity=0.75
        ).add_to(capa_calles)

    # --- Nodos: TODOS los nodos que no son candidatos, como puntos oscuros ---
    for n in problema.grafo.nodes():
        if n in problema.candidatos:
            continue
        lon_n, lat_n = pos[n]
        folium.CircleMarker(
            (lat_n, lon_n), radius=2.2, color="#333333", weight=1,
            fill=True, fill_color="#333333", fill_opacity=0.8,
        ).add_to(capa_nodos)

    # --- Candidatos: color y radio de cobertura según el gen ---
    capas_por_gen = {0: capa_sin_estacion, 1: capa_lenta, 2: capa_rapida}
    for gen, j in zip(individuo, problema.candidatos):
        lon_j, lat_j = pos[j]
        gen = int(gen)
        color = _color_por_gen(gen)
        capa = capas_por_gen[gen]
        if gen == 0:
            folium.CircleMarker(
                (lat_j, lon_j), radius=5, color="black", weight=1, fill=True,
                fill_color=color, fill_opacity=0.9,
                popup=f"Candidato {j}: sin estación",
            ).add_to(capa)
        else:
            tipo = "lenta" if gen == 1 else "rapida"
            folium.CircleMarker(
                (lat_j, lon_j), radius=8, color="black", weight=1, fill=True,
                fill_color=color, fill_opacity=0.95,
                popup=(
                    f"Estación {tipo} en {j}<br>"
                    f"Capacidad: {problema.capacidad[(j, tipo)]:.1f}<br>"
                    f"Coste: {problema.coste[(j, tipo)]:.0f}"
                ),
            ).add_to(capa)
            folium.Circle(
                (lat_j, lon_j), radius=problema.radio[tipo],
                color=color, weight=1, fill=True, fill_opacity=0.04,
            ).add_to(capa_cobertura)

    # --- Asignaciones de demanda -> estación (y_ij^t) ---
    max_flujo = max(y.values()) if y else 1.0
    for (i, j, _t), cantidad in y.items():
        if i == j:
            continue
        lon_i, lat_i = pos[i]
        lon_j, lat_j = pos[j]
        folium.PolyLine(
            [(lat_i, lon_i), (lat_j, lon_j)],
            color="#111111", weight=1 + 4 * (cantidad / max_flujo),
            opacity=0.7, dash_array="4,6",
        ).add_to(capa_asignaciones)

    for capa in (
        capa_calles, capa_nodos, capa_sin_estacion, capa_lenta,
        capa_rapida, capa_cobertura, capa_asignaciones,
    ):
        capa.add_to(mapa)
    folium.LayerControl(collapsed=False).add_to(mapa)

    titulo_html = f"<h4 style='margin:8px'>Demanda atendida: {fitness:.1f}</h4>"
    mapa.get_root().html.add_child(folium.Element(titulo_html))

    mapa.save(ruta_salida)
    return ruta_salida