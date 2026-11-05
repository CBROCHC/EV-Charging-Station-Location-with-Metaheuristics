# Code guide

Identifiers, docstrings and comments in `src/` are in **Spanish** (course language). This guide maps them to English and documents every tunable parameter.

All scripts are meant to be run **from inside `src/`** (flat module layout, `main_es.py` does `import main`).

## Module map

| Module | Role |
|---|---|
| `problema.py` | `ProblemaCarga` dataclass (sets and parameters of the model) + synthetic graph / hotspot-demand / substation generators |
| `grafo_real.py` | Builds a problem from a real OpenStreetMap graph (OSMnx) with the same format as synthetic ones |
| `distancias.py` | Dijkstra distances between every node and every candidate (limited to the largest radius) |
| `codificacion.py` | Encoding `x ∈ {0,1,2}^|J|`, decoding to `x_j^t`, cost / substation usage, feasibility check, **repair**, random and greedy individuals |
| `fitness.py` | Builds the bipartite flow network and computes the optimal `y` by **max-flow**; returns total demand served |
| `algoritmo_genetico.py` | GA: population, tournament, substation/uniform crossover, two mutation families, elitism, stop criteria, history, optional checkpoint callback |
| `algoritmo_es.py` | SA: `ConfiguracionES`, 4 neighbourhood operators, 4 cooling schemes, stop criteria, optional reheating |
| `visualizacion.py` | Solution graph, analysis plots, generation-slider explorer, interactive Folium map with layers |
| `visualizacion_es.py` | SA plots: convergence, temperature, acceptance rate, fitness vs T, composition, resource use |
| `instancias.py` | The 8 benchmark instances (`S-100 … R-ciudad`) and the greedy reference |
| `config_experimentos.py` | Experimental design: seeds, per-phase budgets, GA default config and factor levels, problem-parameter levels |
| `experimentos.py` | Parallel runner of GA phases **A–D** (resumable: reruns only missing runs) |
| `analisis.py` | Aggregates GA experiment results into plots / report folder |
| `experimentos_es.py` | SA experiments: T₀×α sweep, cooling schemes, initial solutions, operators, GA vs SA |
| `experimentos_instancias_es.py` | SA experiments across instances |
| `run_ag_2h.py`, `run_es_2h.py`, `comun_2h.py` | 2-hour time-budgeted runs with periodic checkpoints (`.pkl`) |
| `comparar_2h.py` | Reads the two `.pkl` files and draws the GA-vs-SA comparison + summary CSV |
| `main.py`, `main_es.py` | Single-run entry points; parameters at the top of each file |

## Spanish → English glossary

| Spanish | English |
|---|---|
| problema, instancia | problem, instance |
| nodos / aristas / candidatos | nodes / edges / candidates |
| subestación(es) | substation(s) |
| lenta / rápida | slow / fast |
| demanda (atendida) | demand (served) |
| presupuesto | budget |
| individuo / población | individual / population |
| torneo / élite | tournament / elite |
| cruce (uniforme, por subestación) | crossover (uniform, per-substation) |
| mutación (por gen, de intercambio) | mutation (per-gene, swap-style) |
| reparación / reparar | repair |
| voraz / greedy | greedy |
| enfriamiento simulado (ES) | simulated annealing (SA) |
| algoritmo genético (AG) | genetic algorithm (GA) |
| vecindad / vecino | neighbourhood / neighbour |
| estado / swap / tipo / desplazamiento | state change / swap / type change / displacement |
| esquema geométrico / lineal / inverso / escalonado | geometric / linear / inverse / stepped scheme |
| estancamiento | stagnation |
| recalentado | reheating |
| semilla | seed |
| barrido | sweep |
| salidas / resultados / informe | outputs / results / report |

## `ConfiguracionAG` (GA)

| Parameter | Meaning | Default |
|---|---|---|
| `tam_poblacion` | Population size | 60 |
| `criterio_parada` | `"generaciones"`, `"tiempo"`, `"evaluaciones"` or `"estancamiento"` | `"estancamiento"` |
| `num_generaciones` / `tiempo_max_segundos` / `max_evaluaciones` | Budget for the matching stop criterion (`max_evaluaciones` counts *generated solutions*: initial population + children; copied elite not counted) | 150 / `None` / `None` |
| `paciencia_estancamiento`, `tol_estancamiento` | Stagnation stop: generations without ≥ tol improvement | 20, 1e-3 |
| `num_semillas_greedy` | Greedy individuals in the initial population | 2 |
| `tipo_cruce` | `"subestacion"` (substation blocks) or `"uniforme"` | `"subestacion"` |
| `prob_cruce` | Crossover probability | 0.9 |
| `prob_mutacion_gen` | Per-gene mutation probability (`None` → 1/\|J\|) | `None` |
| `prob_mutacion_intercambio` | Per-child probability of swap-style mutation | 0.7 |
| `max_intercambios` | Structured moves per mutation (uniform in 1..max) | 3 |
| `tam_torneo` | Tournament size | 3 |
| `num_elite` | Elite individuals kept each generation | 2 |
| `usar_cache_fitness` | Cache fitness per genotype | `True` |
| `intervalo_seguimiento`, `verbose` | Console progress | 25, `True` |
| `semilla` | Random seed | `None` |

The **recommended** values found in the study (population 100, tournament 5, elite 4, …) are listed in the main README. They are *not* all the code defaults: `config_experimentos.DEFECTO` is the study's reference level (tournament 3), the final configuration is applied on top of it via an optional `config_final.json`, and `main.py` carries its own example values (population 50, tournament 3, elite 4) — set `CONFIG_AG` there to the recommended values to reproduce the report.

## `ConfiguracionES` (SA)

| Parameter | Meaning | Default |
|---|---|---|
| `T0`, `T_min` | Initial / minimum temperature | 0.01, 1e-5 |
| `normalizacion` | `"ninguna"` (Δf in absolute demand units — used in the report, e.g. `T0=200`), `"demanda_total"` (Δf divided by total demand — `T0≈0.01`), `"fitness_inicial"` | `"demanda_total"` |
| `esquema_enfriamiento` | `geometrico` / `lineal` / `inverso` / `escalonado` | `geometrico` |
| `alpha`, `delta_lineal`, `beta` | Cooling rate per scheme (`None` → derived so T goes T0→T_min over the whole budget) | `None` |
| `pasos_por_escalon` | Only for `escalonado` | 10 |
| `L` | Neighbours evaluated per temperature | 1 |
| `solucion_inicial` | `aleatoria` / `voraz` / `ag` | `aleatoria` |
| `probs_operadores` | Probabilities of (state, swap, type, displacement) | (.25, .35, .15, .25) |
| `max_movimientos` | Moves per neighbour (uniform in 1..max) | 1 |
| `k_vecinos_cercanos` | k nearest candidates for displacement | 5 |
| `criterio_parada` | `evaluaciones` / `tiempo` / `estancamiento` / `temperatura` | `evaluaciones` |
| `max_evaluaciones`, `tiempo_max_segundos` | Budgets | 20000, `None` |
| `paciencia_estancamiento`, `tol_estancamiento` | Stagnation stop | 3000, 1e-3 |
| `recalentado`, `paciencia_recalentado`, `factor_recalentado` | Optional non-monotonic cooling | `False`, 2000, 0.5 |
| `usar_cache_fitness` | Do not re-evaluate seen genotypes | `True` |
| `intervalo_historial`, `intervalo_seguimiento` | History sampling / console progress | 50, 2000 |
| `semilla` | Random seed | `None` |

**Time-based stop.** With `criterio_parada="tiempo"`, temperature follows the fraction of time consumed (`T0 → T_min` exactly when time ends), so `alpha` is not used.

## Experiment phases (`config_experimentos.py`)

| Phase | What | Budget per run | Runs |
|---|---|---|---|
| **A** | One GA factor at a time on the reference instance (`R-centro`) | 10,000 solutions | 345 |
| **B** | Final GA config on all 8 instances | 20,000 | 120 |
| **C** | Key factors repeated on `R-pueblo`, `R-grid`, `S-250` | 10,000 | 450 |
| **D** | Sensitivity to *problem* parameters: budget fraction, demand/capacity ratio, radii scale, hotspot fraction, substation count | 10,000 | 165 |

Seeds: 15 GA seeds (101–115) per configuration; the instance itself always uses a fixed instance seed, so only the algorithm varies between repetitions.

