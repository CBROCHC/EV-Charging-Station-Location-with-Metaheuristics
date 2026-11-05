"""
main.py

Script plantilla de ejecución. TODO lo que normalmente se
quiere cambiar entre ejecuciones está en la sección "PARÁMETROS" de más
abajo: no hace falta tocar el resto del fichero para experimentar.

Ejecutar con:  python3 main.py
"""

import os
from datetime import datetime

from problema import generar_grafo_sintetico, generar_parametros_problema
from distancias import calcular_distancias
from algoritmo_genetico import ejecutar_algoritmo_genetico, ConfiguracionAG
from codificacion import decodificar_individuo, coste_total, uso_por_subestacion, es_factible
from visualizacion import (
    dibujar_grafo_solucion,
    dibujar_mapa_real,
    generar_todas_las_graficas,
    generar_explorador_generaciones,
)

# PARÁMETROS 

# --- 1) Origen del grafo: sintético reducido (para pruebas) o real (OSM) ---
USAR_GRAFO_REAL = True  # True = grafo real (Madrid, Sol) | False = grafo sintético  

GRAFO_REAL_PUNTO = (41.38896, 2.16179)   # (lat, lon) — aquí: Madrid, Sol
GRAFO_REAL_DISTANCIA_M = 2000            # radio en metros alrededor del punto
GRAFO_REAL_N_CANDIDATOS = 3000            # antes 500: con ~700 nodos, 250 candidatos da un espacio de búsqueda interesante
GRAFO_REAL_N_SUBESTACIONES = 10

# --- Parámetros de calibrado del problema (se pasan a generar_parametros_problema) ---
# Objetivo: que NO sea un simple problema de mochila (todas las estaciones saturadas),
# sino que cobertura, solapes, tipo de estación y subestaciones influyan de verdad.
FRACCION_PRESUPUESTO = 0.30               # B = 35 % del coste de abrir TODOS los candidatos en rápida
RADIO_LENTA_M = 120.0                    # antes 200
RADIO_RAPIDA_M = 220.0                   # antes 350
RATIO_DEMANDA_CAPACIDAD = 0.8            # demanda total = 0.8 x cota de capacidad que B puede comprar (calibrado empíricamente)
RANGO_CAP_SUBESTACIONES = (0.9, 1.1)     # sum Cap_k ≈ 90-110 % de esa cota (presupuesto y subestaciones compiten)
FRACCION_HOTSPOTS = 0.1                 # % de nodos que son centro de un foco de demanda


SINTETICO_N_NODOS = 500
SINTETICO_ESCALA_METROS = 2600.0
SINTETICO_N_CANDIDATOS = 175
SINTETICO_N_SUBESTACIONES = 8
#SINTETICO_FRACCION_PRESUPUESTO = 0.25

SEMILLA = 7  # semilla común para la generación del problema

# --- 2) Configuración del algoritmo genético ---
CONFIG_AG = ConfiguracionAG(
    tam_poblacion=50,   

    # Criterio de parada: "generaciones" (nº fijo), "tiempo" (segundos) o "estancamiento".
    criterio_parada="generaciones",
    num_generaciones=150,          # usado si criterio_parada == "generaciones"
    tiempo_max_segundos=28800.0,      # usado si criterio_parada == "tiempo" (evitar con PCs que se suspenden)
    
    # Parámetros para la condición de parada por estancamiento:
    paciencia_estancamiento=25,    # número de generaciones sin mejora para cortar
    tol_estancamiento=1e-3,        # umbral de mejora en el fitness

    num_semillas_greedy=2,         # antes 10: el greedy ya daba ~99 % del óptimo y eclipsaba al AG, aporta mucho que en pocas generaciones lleguemos al tope
    tipo_cruce="subestacion",      # "subestacion" (cruce por barrios) | "uniforme"
    prob_cruce=0.9,
    prob_mutacion_gen=None,
    prob_mutacion_intercambio=0.7, # prob. por hijo de aplicar mutación de intercambio (mover/cambiar tipo/abrir)
    max_intercambios=3,            # movimientos por mutación: uniforme en 1..max
    usar_cache_fitness=True,       # no reevaluar genotipos ya vistos (clones, élite)
    tam_torneo=3,                 # Antes 3
    num_elite=4,

    intervalo_seguimiento=25,      # cada cuántas generaciones se informa por consola
    verbose=True,

    semilla=3,
)

# --- 3) Explorador de generaciones (HTML con slider) ---
GENERAR_EXPLORADOR = True
EXPLORADOR_PASO = 5  # 1 de cada `paso` generaciones se renderiza en el slider

# --- 4) Estilo del mapa real (si USAR_GRAFO_REAL = True) ---
MAPA_ESTILO = "osm"                    # "claro" | "oscuro" | "osm"
MAPA_MOSTRAR_COBERTURA_POR_DEFECTO = False


def main():
    # Garantizar existencia de la carpeta de salidas y sello temporal único
    os.makedirs("salidas", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # 1) Construcción del problema.
    if USAR_GRAFO_REAL:
        from grafo_real import generar_problema_real
        problema = generar_problema_real(
            n_candidatos=GRAFO_REAL_N_CANDIDATOS,
            n_subestaciones=GRAFO_REAL_N_SUBESTACIONES,
            punto=GRAFO_REAL_PUNTO,
            distancia_metros=GRAFO_REAL_DISTANCIA_M,
            semilla=SEMILLA,
            fraccion_presupuesto=FRACCION_PRESUPUESTO,
            fraccion_hotspots=FRACCION_HOTSPOTS,
            radio_lenta=RADIO_LENTA_M,
            radio_rapida=RADIO_RAPIDA_M,
            ratio_demanda_capacidad=RATIO_DEMANDA_CAPACIDAD,
            rango_cap_subestaciones=RANGO_CAP_SUBESTACIONES,
        )
    else:
        grafo = generar_grafo_sintetico(
            n_nodos=SINTETICO_N_NODOS, escala_metros=SINTETICO_ESCALA_METROS, semilla=SEMILLA
        )
        problema = generar_parametros_problema(
            grafo,
            n_candidatos=SINTETICO_N_CANDIDATOS,
            n_subestaciones=SINTETICO_N_SUBESTACIONES,
            fraccion_presupuesto=FRACCION_PRESUPUESTO,
            fraccion_hotspots=FRACCION_HOTSPOTS,
            radio_lenta=RADIO_LENTA_M,
            radio_rapida=RADIO_RAPIDA_M,
            ratio_demanda_capacidad=RATIO_DEMANDA_CAPACIDAD,
            rango_cap_subestaciones=RANGO_CAP_SUBESTACIONES,
            semilla=SEMILLA,
        )
    print("Problema generado:", problema.resumen())

    # 2) Distancias de camino más corto d_G(i,j) (Dijkstra), una sola vez.
    distancias = calcular_distancias(problema)

    # 3) Algoritmo genético
    resultado = ejecutar_algoritmo_genetico(problema, distancias, CONFIG_AG)

    # 4) Resultados por consola.
    demanda_total = sum(problema.demanda.values())
    x = decodificar_individuo(resultado.mejor_individuo, problema)

    print("\n--- Resultado ---")
    print(f"Demanda total de la ciudad:      {demanda_total:.2f}")
    print(f"Demanda atendida (fitness):      {resultado.mejor_fitness:.2f} "
          f"({100 * resultado.mejor_fitness / demanda_total:.1f} %)")
    print(f"Coste usado / Presupuesto:       {coste_total(x, problema):.1f} / {problema.presupuesto:.1f}")
    print(f"¿Solución factible?              {es_factible(x, problema)}")
    print(f"Uso por subestación:              {uso_por_subestacion(x, problema)}")
    print(f"Capacidad por subestación:        {problema.capacidad_subestacion}")

    n_lenta = sum(1 for g in resultado.mejor_individuo if g == 1)
    n_rapida = sum(1 for g in resultado.mejor_individuo if g == 2)
    print(f"Estaciones instaladas: {n_lenta} lentas + {n_rapida} rápidas "
          f"(de {len(problema.candidatos)} candidatos)")

    # 5) Visualización de la solución final con nombres timestamped
    ruta_grafo = dibujar_grafo_solucion(
        problema,
        resultado.mejor_individuo,
        distancias,
        ruta_salida=f"salidas/solucion_grafo_{timestamp}.png"
    )
    print(f"\nFigura del grafo con la solución guardada en: {ruta_grafo}")

    if problema.es_real:
        ruta_mapa = dibujar_mapa_real(
            problema,
            resultado.mejor_individuo,
            distancias,
            ruta_salida=f"salidas/solucion_mapa_{timestamp}.html",
            estilo_mapa=MAPA_ESTILO,
            mostrar_cobertura_por_defecto=MAPA_MOSTRAR_COBERTURA_POR_DEFECTO,
        )
        print(f"Mapa real con la solución guardado en:         {ruta_mapa}")

    # 6) Batería de gráficas de análisis del algoritmo genético
    carpeta_graficas = os.path.join("salidas", f"graficas_{timestamp}")
    rutas_graficas = generar_todas_las_graficas(problema, resultado, carpeta_salida=carpeta_graficas)
    print("\nGráficas de análisis generadas:")
    for nombre, ruta in rutas_graficas.items():
        print(f"  - {nombre}: {ruta}")

    # 7) Explorador interactivo de generaciones (slider)
    if GENERAR_EXPLORADOR:
        ruta_explorador = generar_explorador_generaciones(
            problema,
            resultado,
            distancias,
            ruta_salida=f"salidas/explorador_generaciones_{timestamp}.html",
            paso=EXPLORADOR_PASO,
        )
        print(f"\nExplorador de generaciones (abrir en el navegador): {ruta_explorador}")


if __name__ == "__main__":
    main()