"""
main_es.py

El PROBLEMA (grafo, candidatos, presupuesto, subestaciones...) se construye
con los mismos parámetros de main.py: este fichero los importa de allí, así
que ES y AG resuelven exactamente la misma instancia y solo hay una fuente de
verdad. Aquí solo están los parámetros propios del ES.

Ejecutar con:  python3 main_es.py
"""

import dataclasses
import os
from datetime import datetime

import main as cfg_ag   # parámetros del problema y CONFIG_AG (no ejecuta nada: main() está protegido)
from problema import generar_grafo_sintetico, generar_parametros_problema
from distancias import calcular_distancias
from algoritmo_genetico import ejecutar_algoritmo_genetico
from algoritmo_es import ConfiguracionES, ejecutar_enfriamiento_simulado
from codificacion import decodificar_individuo, coste_total, uso_por_subestacion, es_factible
from visualizacion import dibujar_grafo_solucion, dibujar_mapa_real, generar_explorador_generaciones
from visualizacion_es import generar_todas_las_graficas_es

# PARÁMETROS DEL ES 
CONFIG_ES = ConfiguracionES(
    # --- Temperatura (Δf está normalizado: fracción de la demanda total) ---
    T0=200,                          # temperatura inicial: se ajusta experimentalmente (ver experimentos_es.py)
    T_min=1e-5,                       # temperatura mínima (suelo del enfriamiento)
    esquema_enfriamiento="geometrico",  # "geometrico" | "lineal" | "inverso" | "escalonado"
    alpha=0.99,                       # factor de enfriamiento para "geometrico"
    delta_lineal=None,                # idem para "lineal"
    beta=None,                        # idem para "inverso"
    pasos_por_escalon=10,             # solo "escalonado"
    L=1,                              # vecinos evaluados por temperatura
    normalizacion="ninguna",          # "ninguna": delta-f en unidades absolutas (T0=200, como en la memoria) | "demanda_total" | "fitness_inicial"

    # --- Solución inicial: "aleatoria" | "voraz" | "ag" ---
    solucion_inicial="ag",

    # --- Vecindad: probabilidades de (estado, swap, tipo, desplazamiento) ---
    probs_operadores=(0.25, 0.35, 0.15, 0.25),
    max_movimientos=1,                # 1 = un único movimiento por vecino
    k_vecinos_cercanos=5,             # k del operador de desplazamiento

    # --- Parada: "evaluaciones" | "tiempo" | "estancamiento" | "temperatura" ---
    criterio_parada="evaluaciones",
    max_evaluaciones=20000,
    tiempo_max_segundos=3600,
    paciencia_estancamiento=3000,
    tol_estancamiento=1e-3,

    # --- Recalentado---
    recalentado=False,
    paciencia_recalentado=2000,
    factor_recalentado=0.5,

    usar_cache_fitness=True,
    intervalo_historial=50,           # cada cuántas evaluaciones se guarda una muestra
    intervalo_seguimiento=2000,       # cada cuántas evaluaciones se informa por consola
    verbose=True,
    semilla=3,
)

# Si solucion_inicial == "ag": se ejecuta antes el AG de main.py durante este tiempo (s)
AG_TIEMPO_PARA_INICIO_S = 1200.0

GENERAR_EXPLORADOR = True
EXPLORADOR_PASO = 2      # 1 de cada `paso` muestras del historial entra en el slider


def construir_problema():
    """Construye la MISMA instancia que main.py (sintética o real)."""
    c = cfg_ag
    comunes = dict(
        fraccion_presupuesto=c.FRACCION_PRESUPUESTO,
        fraccion_hotspots=c.FRACCION_HOTSPOTS,
        radio_lenta=c.RADIO_LENTA_M,
        radio_rapida=c.RADIO_RAPIDA_M,
        ratio_demanda_capacidad=c.RATIO_DEMANDA_CAPACIDAD,
        rango_cap_subestaciones=c.RANGO_CAP_SUBESTACIONES,
    )
    if c.USAR_GRAFO_REAL:
        from grafo_real import generar_problema_real
        return generar_problema_real(
            n_candidatos=c.GRAFO_REAL_N_CANDIDATOS,
            n_subestaciones=c.GRAFO_REAL_N_SUBESTACIONES,
            punto=c.GRAFO_REAL_PUNTO,
            distancia_metros=c.GRAFO_REAL_DISTANCIA_M,
            semilla=c.SEMILLA,
            **comunes,
        )
    grafo = generar_grafo_sintetico(
        n_nodos=c.SINTETICO_N_NODOS, escala_metros=c.SINTETICO_ESCALA_METROS, semilla=c.SEMILLA
    )
    return generar_parametros_problema(
        grafo,
        n_candidatos=c.SINTETICO_N_CANDIDATOS,
        n_subestaciones=c.SINTETICO_N_SUBESTACIONES,
        semilla=c.SEMILLA,
        **comunes,
    )


def main():
    os.makedirs("salidas", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    problema = construir_problema()
    print("Problema generado:", problema.resumen())
    distancias = calcular_distancias(problema)

    # Solución inicial 
    inicial, previas = None, 0
    if CONFIG_ES.solucion_inicial == "ag":
        cfg_ini = dataclasses.replace(
            cfg_ag.CONFIG_AG,
            criterio_parada="tiempo",
            tiempo_max_segundos=AG_TIEMPO_PARA_INICIO_S,
            verbose=False,
        )
        res_ag = ejecutar_algoritmo_genetico(problema, distancias, cfg_ini)
        inicial, previas = res_ag.mejor_individuo, res_ag.historial_evaluaciones[-1]
        print(f"Solución inicial del AG: fitness={res_ag.mejor_fitness:.2f} "
              f"({previas} evaluaciones, {res_ag.historial_tiempo[-1]:.1f} s)")

    
    # Enfriamiento simulado
    resultado = ejecutar_enfriamiento_simulado(
        problema, distancias, CONFIG_ES, solucion_inicial=inicial, evaluaciones_previas=previas
    )

    # Resultados por consola
    demanda_total = sum(problema.demanda.values())
    x = decodificar_individuo(resultado.mejor_individuo, problema)
    print("\n--- Resultado (ES) ---")
    print(f"Demanda total de la ciudad:      {demanda_total:.2f}")
    print(f"Demanda atendida (fitness):      {resultado.mejor_fitness:.2f} "
          f"({100 * resultado.mejor_fitness / demanda_total:.1f} %)")
    print(f"Fitness de la solución inicial:  {resultado.fitness_inicial:.2f}")
    print(f"Coste usado / Presupuesto:       {coste_total(x, problema):.1f} / {problema.presupuesto:.1f}")
    print(f"¿Solución factible?              {es_factible(x, problema)}")
    print(f"Uso por subestación:              {uso_por_subestacion(x, problema)}")
    print(f"Capacidad por subestación:        {problema.capacidad_subestacion}")
    n_len = int((resultado.mejor_individuo == 1).sum())
    n_rap = int((resultado.mejor_individuo == 2).sum())
    print(f"Estaciones instaladas: {n_len} lentas + {n_rap} rápidas (de {len(problema.candidatos)} candidatos)")
    print(f"Evaluaciones: {resultado.evaluaciones_totales} (+{resultado.evaluaciones_previas} previas), "
          f"T final = {resultado.T_final:.2e}, parada: {resultado.motivo_parada}")

    # Visualización
    ruta_grafo = dibujar_grafo_solucion(
        problema, resultado.mejor_individuo, distancias,
        ruta_salida=f"salidas/es_solucion_grafo_{timestamp}.png",
    )
    print(f"\nFigura del grafo con la solución guardada en: {ruta_grafo}")

    if problema.es_real:
        ruta_mapa = dibujar_mapa_real(
            problema, resultado.mejor_individuo, distancias,
            ruta_salida=f"salidas/es_solucion_mapa_{timestamp}.html",
            estilo_mapa=cfg_ag.MAPA_ESTILO,
            mostrar_cobertura_por_defecto=cfg_ag.MAPA_MOSTRAR_COBERTURA_POR_DEFECTO,
        )
        print(f"Mapa real con la solución guardado en: {ruta_mapa}")

    carpeta = os.path.join("salidas", f"graficas_es_{timestamp}")
    print("\nGráficas de análisis generadas:")
    for nombre, ruta in generar_todas_las_graficas_es(problema, resultado, carpeta).items():
        print(f"  - {nombre}: {ruta}")

    if GENERAR_EXPLORADOR:
        ruta_exp = generar_explorador_generaciones(
            problema, resultado, distancias,
            ruta_salida=f"salidas/es_explorador_{timestamp}.html",
            paso=EXPLORADOR_PASO,
        )
        print(f"\nExplorador (abrir en el navegador): {ruta_exp}")


if __name__ == "__main__":
    main()
