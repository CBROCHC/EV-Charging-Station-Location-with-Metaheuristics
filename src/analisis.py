"""
analisis.py
===========

Lee los JSON de `resultados/` y genera en `informe/` las tablas (CSV + LaTeX)
y figuras (PNG) del apartado de Evaluación:

  T1  tabla_instancias            instancias (|V|, |E|, |J|, K, B, demanda total)
  F2  convergencia_*              fitness vs soluciones generadas / vs tiempo (fase B)
  F3  sensibilidad_*              cajas por factor (fases A y C) + tabla_parametros_*
  F4  escala                      tiempo, soluciones hasta el 99 % y calidad vs |V| (fase B)
  T5  tabla_resumen               AG vs greedy, tiempos, composición de la solución (fase B)
  F6  exploracion                 diversidad y fitness de dos configuraciones contrastadas
  F7  mapas (opcional, --mapas)   solución sobre el grafo/mapa (mejor ejecución de la fase B)
  F8  impacto_problema_*          efecto de los parámetros del PROBLEMA en la solución (fase D)
                                  + tabla_impacto_problema_*

Uso:
  python analisis.py                       # todo lo que haya
  python analisis.py --explora tam_torneo=2 tam_torneo=10
  python analisis.py --mapas R-centro      # (en el PC: descarga el grafo con osmnx; necesita folium)

Nota estadística: con 15 semillas por configuración se usa Mann-Whitney U
(muestras independientes: aunque la semilla sea la misma, las trayectorias divergen).
Con muchos factores y niveles hay muchas comparaciones: los p-valores son orientativos
y conviene fijarse sobre todo en el tamaño del efecto y en la dispersión (cajas).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_experimentos as cfgx

try:
    from scipy.stats import mannwhitneyu
except ImportError:          # sin scipy: se omiten los p-valores
    mannwhitneyu = None


# ----------------------------------------------------------------------
def cargar(dir_res):
    runs = []
    for f in glob.glob(os.path.join(dir_res, "**", "*.json"), recursive=True):
        with open(f, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def tabla(runs):
    filas = []
    for r in runs:
        filas.append({"fase": r["fase"], "instancia": r["instancia"], "factor": r["factor"],
                      "nivel": r["nivel"], "semilla": r["semilla"],
                      "greedy_pct": 100.0 * r["info"]["greedy_medio"] / r["info"]["demanda_total"],
                      **r["metricas"]})
    return pd.DataFrame(filas)


def _esc(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("±", r"$\pm$").replace("|", r"$|$"))


def guardar_tabla(df, ruta_sin_ext, caption, label):
    """CSV + LaTeX (tabular escrita a mano para no depender de jinja2)."""
    df.to_csv(ruta_sin_ext + ".csv", index=False, encoding="utf-8")
    cols = "l" + "r" * (len(df.columns) - 1)
    lineas = [r"\begin{table}[ht]\centering\small", rf"\begin{{tabular}}{{{cols}}}", r"\hline",
              " & ".join(_esc(c) for c in df.columns) + r" \\ \hline"]
    for _, fila in df.iterrows():
        lineas.append(" & ".join(_esc(v) for v in fila.values) + r" \\")
    lineas += [r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    with open(ruta_sin_ext + ".tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))


def _curva(runs, xk, yfun, n=200):
    """Media y desviación de una curva escalonada sobre una rejilla común."""
    xmax = min(r["hist"][xk][-1] for r in runs)
    grid = np.linspace(0, xmax, n)
    Y = []
    for r in runs:
        x = np.asarray(r["hist"][xk], float)
        idx = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(x) - 1)
        Y.append(np.asarray(yfun(r), float)[idx])
    Y = np.array(Y)
    return grid, Y.mean(0), Y.std(0)


def _pct(r):
    return np.asarray(r["hist"]["mejor"]) * 100.0 / r["info"]["demanda_total"]


# ----------------------------------------------------------------------
# T1, T5 y F4 (fase B)
# ----------------------------------------------------------------------
def tablas_instancias(runs, df, out):
    infos = {r["instancia"]: r["info"] for r in runs if not r["problema"]}
    orden = sorted(infos.values(), key=lambda i: i["n_nodos"])
    t1 = pd.DataFrame([{
        "Instancia": i["id"], "Tipo": i["tipo"], "|V|": i["n_nodos"], "|E|": i["n_aristas"],
        "|J|": i["n_candidatos"], "K": i["n_subestaciones"],
        "B": f"{i['presupuesto']:.0f}", "Demanda total": f"{i['demanda_total']:.0f}"} for i in orden])
    guardar_tabla(t1, os.path.join(out, "tabla_instancias"), "Instancias de prueba.", "tab:instancias")

    b = df[df.fase == "B"]
    if b.empty:
        return
    filas = []
    for i in orden:
        g = b[b.instancia == i["id"]]
        if g.empty:
            continue
        m = g.pct_demanda
        greedy = 100.0 * i["greedy_medio"] / i["demanda_total"]
        filas.append({
            "Instancia": i["id"], "|V|": i["n_nodos"],
            "Greedy (%)": f"{greedy:.1f}",
            "AG (%)": f"{m.mean():.1f} ± {m.std(ddof=0):.1f}",
            "Mejora vs greedy (pp)": f"{m.mean() - greedy:.1f}",
            "Sol. hasta 99 %": f"{g.evals_99.mean():.0f}",
            "Tiempo (s)": f"{g.tiempo.mean():.0f}",
            "Lentas/Rápidas": f"{g.n_lentas.mean():.0f}/{g.n_rapidas.mean():.0f}",
            "Presup. (%)": f"{g.pct_presupuesto.mean():.0f}",
            "Subest. (%)": f"{g.pct_subestaciones.mean():.0f}",
        })
    guardar_tabla(pd.DataFrame(filas), os.path.join(out, "tabla_resumen"),
                  "Resultados del AG con la configuración final y comparación con el greedy "
                  "(% de la demanda total atendida).", "tab:resumen")

    # F4: escala
    g = b.merge(pd.DataFrame([{"instancia": i["id"], "n_nodos": i["n_nodos"], "tipo": i["tipo"]}
                              for i in orden]), on="instancia")
    agg = g.groupby(["instancia", "n_nodos", "tipo"]).agg(
        tiempo=("tiempo", "mean"), evals_99=("evals_99", "mean"),
        pct=("pct_demanda", "mean"), pct_sd=("pct_demanda", "std")).reset_index().sort_values("n_nodos")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col, et in zip(ax, ["tiempo", "evals_99", "pct"],
                          ["Tiempo de cómputo (s)", "Soluciones generadas hasta el 99 %",
                           "Demanda atendida (%)"]):
        for tipo, c in (("sintetico", "#2A9D8F"), ("real", "#D6432D")):
            s = agg[agg.tipo == tipo]
            yerr = s.pct_sd if col == "pct" else None
            a.errorbar(s.n_nodos, s[col], yerr=yerr, marker="o", color=c, label=tipo, capsize=3)
        a.set_xscale("log")
        if col != "pct":
            a.set_yscale("log")
        a.set_xlabel("|V| (nodos)")
        a.set_ylabel(et)
        a.grid(alpha=0.3, which="both")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "escala.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# F2: convergencia (fase B)
# ----------------------------------------------------------------------
def figuras_convergencia(runs, out):
    rb = [r for r in runs if r["fase"] == "B"]
    if not rb:
        return
    por_inst = {}
    for r in rb:
        por_inst.setdefault(r["instancia"], []).append(r)
    orden = sorted(por_inst, key=lambda i: por_inst[i][0]["info"]["n_nodos"])
    for xk, nombre, xlab in (("evaluaciones", "convergencia_evaluaciones", "Soluciones generadas"),
                             ("tiempo", "convergencia_tiempo", "Tiempo (s)")):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, tipo in zip(axs, ("sintetico", "real")):
            ids = [i for i in orden if por_inst[i][0]["info"]["tipo"] == tipo]
            cmap = plt.get_cmap("viridis")
            for k, i in enumerate(ids):
                x, m, s = _curva(por_inst[i], xk, _pct)
                c = cmap(k / max(1, len(ids) - 1))
                ax.plot(x, m, color=c, label=f"{i} (|V|={por_inst[i][0]['info']['n_nodos']})")
                ax.fill_between(x, m - s, m + s, color=c, alpha=0.2)
            ax.set_title("Instancias " + ("sintéticas" if tipo == "sintetico" else "reales"))
            ax.set_xlabel(xlab)
            ax.grid(alpha=0.3)
            if ids:
                ax.legend(fontsize=8)
        axs[0].set_ylabel("Mejor demanda atendida (% del total)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, nombre + ".png"), dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------
# F3: sensibilidad de parámetros (fases A y C)
# ----------------------------------------------------------------------
def _datos_nivel(d, factor, nivel):
    """Filas de un nivel; el nivel por defecto viene de la configuración 'defecto'."""
    if str(nivel) == str(cfgx.DEFECTO[factor]):
        return d[d.factor == "defecto"]
    return d[(d.factor == factor) & (d.nivel == str(nivel))]


def figuras_sensibilidad(df, out):
    for (fase, inst), d in df[df.fase.isin(["A", "C"])].groupby(["fase", "instancia"]):
        factores = [f for f in cfgx.NIVELES if (d.factor == f).any()]
        if not factores:
            continue
        nc = 3
        nf = int(np.ceil(len(factores) / nc))
        fig, axs = plt.subplots(nf, nc, figsize=(4.2 * nc, 3.3 * nf), squeeze=False)
        filas = []
        for ax, f in zip(axs.ravel(), factores):
            niveles = cfgx.NIVELES[f]
            datos = [_datos_nivel(d, f, n).pct_demanda.values for n in niveles]
            bp = ax.boxplot(datos, patch_artist=True, widths=0.55,
                            medianprops=dict(color="k"))
            for k, (n, box) in enumerate(zip(niveles, bp["boxes"])):
                es_def = str(n) == str(cfgx.DEFECTO[f])
                box.set_facecolor("#F4A261" if es_def else "#8ecae6")
                ax.scatter(np.full(len(datos[k]), k + 1) + np.linspace(-.12, .12, max(1, len(datos[k]))),
                           datos[k], s=10, color="k", zorder=3)
            ax.set_xticklabels([str(n) for n in niveles])
            ax.set_title(f, fontsize=10)
            ax.set_ylabel("Demanda atendida (%)")
            ax.grid(alpha=0.3)
            base = _datos_nivel(d, f, cfgx.DEFECTO[f])
            for n in niveles:
                g = _datos_nivel(d, f, n)
                if g.empty:
                    continue
                p = np.nan
                if mannwhitneyu is not None and str(n) != str(cfgx.DEFECTO[f]) \
                        and len(g) > 1 and len(base) > 1:
                    p = mannwhitneyu(g.pct_demanda, base.pct_demanda, alternative="two-sided").pvalue
                filas.append({
                    "Factor": f, "Nivel": ("%s (defecto)" % n) if str(n) == str(cfgx.DEFECTO[f]) else n,
                    "n": len(g), "Demanda (%)": f"{g.pct_demanda.mean():.2f} ± {g.pct_demanda.std(ddof=0):.2f}",
                    "Sol. hasta 99 %": f"{g.evals_99.mean():.0f}", "Tiempo (s)": f"{g.tiempo.mean():.0f}",
                    "Hamming final (%)": f"{100 * g.hamming_final.mean():.1f}",
                    "p vs defecto": "-" if np.isnan(p) else f"{p:.3f}"})
        for ax in axs.ravel()[len(factores):]:
            ax.axis("off")
        fig.suptitle(f"Fase {fase} — {inst} (naranja = valor por defecto)", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"sensibilidad_{fase}_{inst}.png"), dpi=150)
        plt.close(fig)
        guardar_tabla(pd.DataFrame(filas), os.path.join(out, f"tabla_parametros_{fase}_{inst}"),
                      f"Efecto de cada parámetro del AG en {inst} (fase {fase}).", f"tab:param_{fase}_{inst}")


# ----------------------------------------------------------------------
# F8: sensibilidad a los parámetros del PROBLEMA (fase D)
# ----------------------------------------------------------------------
def _nivel_problema(d, p, n):
    if n == cfgx.PROBLEMA_DEFECTO[p]:
        return d[d.factor == "defecto"]
    return d[(d.factor == p) & (d.nivel == str(n))]


def figuras_impacto_problema(df, out):
    for inst, d in df[df.fase == "D"].groupby("instancia"):
        params = [p for p in cfgx.NIVELES_PROBLEMA if (d.factor == p).any()]
        if not params:
            continue
        fig, axs = plt.subplots(3, len(params), figsize=(3.4 * len(params), 8.6), squeeze=False)
        filas = []
        for c, p in enumerate(params):
            niveles = cfgx.NIVELES_PROBLEMA[p]
            sub = [_nivel_problema(d, p, n) for n in niveles]
            x = np.arange(1, len(niveles) + 1)
            etiquetas = [str(n) for n in niveles]

            ax = axs[0][c]
            bp = ax.boxplot([g.pct_demanda.values for g in sub], patch_artist=True, widths=0.55,
                            medianprops=dict(color="k"))
            for n, box in zip(niveles, bp["boxes"]):
                box.set_facecolor("#F4A261" if n == cfgx.PROBLEMA_DEFECTO[p] else "#8ecae6")
            ax.scatter(x, [g.greedy_pct.mean() for g in sub], marker="_", s=400, color="#D6432D",
                       zorder=4, label="greedy (media)")
            ax.set_xticklabels(etiquetas)
            ax.set_title(p, fontsize=10)
            ax.set_ylabel("Demanda atendida (%)")
            ax.grid(alpha=0.3)
            if c == 0:
                ax.legend(fontsize=8)

            ax = axs[1][c]
            lentas = [g.n_lentas.mean() for g in sub]
            rapidas = [g.n_rapidas.mean() for g in sub]
            ax.bar(x, lentas, color="#F4A261", label="lentas")
            ax.bar(x, rapidas, bottom=lentas, color="#D6432D", label="rápidas")
            ax.set_xticks(x)
            ax.set_xticklabels(etiquetas)
            ax.set_ylabel("Estaciones instaladas")
            ax.grid(alpha=0.3, axis="y")
            if c == 0:
                ax.legend(fontsize=8)

            ax = axs[2][c]
            ax.bar(x - 0.2, [g.pct_presupuesto.mean() for g in sub], 0.4, color="#2A9D8F", label="presupuesto")
            ax.bar(x + 0.2, [g.pct_subestaciones.mean() for g in sub], 0.4, color="#7A3BF2", label="subestaciones (media)")
            ax.set_xticks(x)
            ax.set_xticklabels(etiquetas)
            ax.set_ylim(0, 110)
            ax.set_ylabel("Uso (%)")
            ax.set_xlabel("nivel")
            ax.grid(alpha=0.3, axis="y")
            if c == 0:
                ax.legend(fontsize=8)

            for n, g in zip(niveles, sub):
                if g.empty:
                    continue
                filas.append({
                    "Parámetro": p, "Nivel": f"{n} (defecto)" if n == cfgx.PROBLEMA_DEFECTO[p] else n,
                    "Greedy (%)": f"{g.greedy_pct.mean():.1f}",
                    "AG (%)": f"{g.pct_demanda.mean():.1f} ± {g.pct_demanda.std(ddof=0):.1f}",
                    "Lentas/Rápidas": f"{g.n_lentas.mean():.0f}/{g.n_rapidas.mean():.0f}",
                    "Presup. (%)": f"{g.pct_presupuesto.mean():.0f}",
                    "Subest. (%)": f"{g.pct_subestaciones.mean():.0f}"})
        fig.suptitle(f"Impacto de los parámetros del problema — {inst} (naranja = por defecto)", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"impacto_problema_{inst}.png"), dpi=150)
        plt.close(fig)
        guardar_tabla(pd.DataFrame(filas), os.path.join(out, f"tabla_impacto_problema_{inst}"),
                      f"Efecto de los parámetros del problema en {inst}.", f"tab:impacto_{inst}")


# ----------------------------------------------------------------------
# F6: exploración y explotación
# ----------------------------------------------------------------------
def figura_exploracion(runs, specs, out):
    ra = [r for r in runs if r["fase"] == "A"]
    if not ra or len(specs) < 2:
        return
    inst = ra[0]["instancia"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for spec in specs:
        f, n = spec.split("=")
        if str(n) == str(cfgx.DEFECTO.get(f)):
            sel = [r for r in ra if r["factor"] == "defecto"]
        else:
            sel = [r for r in ra if r["factor"] == f and r["nivel"] == n]
        if not sel:
            print(f"(sin datos para {spec})")
            continue
        x, m, s = _curva(sel, "evaluaciones", _pct)
        axs[1].plot(x, m, label=spec)
        axs[1].fill_between(x, m - s, m + s, alpha=0.2)
        x, m, s = _curva(sel, "evaluaciones", lambda r: 100 * np.asarray(r["hist"]["hamming"]))
        axs[0].plot(x, m, label=spec)
        axs[0].fill_between(x, m - s, m + s, alpha=0.2)
    axs[0].set_ylabel("Distancia de Hamming media (%)")
    axs[1].set_ylabel("Mejor demanda atendida (%)")
    for a in axs:
        a.set_xlabel("Soluciones generadas")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"Exploración y explotación — {inst}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exploracion.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# F7: mapas (opcional)
# ----------------------------------------------------------------------
def figuras_mapas(runs, ids, out):
    import numpy as np
    from instancias import construir_instancia
    from visualizacion import dibujar_grafo_solucion
    for inst in ids:
        rb = [r for r in runs if r["fase"] == "B" and r["instancia"] == inst]
        if not rb:
            print(f"(sin ejecuciones de fase B para {inst})")
            continue
        mejor = max(rb, key=lambda r: r["metricas"]["fitness"])
        P, D, _ = construir_instancia(inst)
        ind = np.array(mejor["mejor_individuo"])
        dibujar_grafo_solucion(P, ind, D, ruta_salida=os.path.join(out, f"solucion_{inst}.png"))
        if P.es_real:
            try:
                from visualizacion import dibujar_mapa_real
                dibujar_mapa_real(P, ind, D, ruta_salida=os.path.join(out, f"solucion_mapa_{inst}.html"),
                                  estilo_mapa="osm", mostrar_cobertura_por_defecto=False)
            except Exception as e:
                print(f"(mapa HTML de {inst} no generado: {e})")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default="resultados")
    ap.add_argument("--out", default="informe")
    ap.add_argument("--explora", nargs="+", default=["tam_torneo=2", "tam_torneo=10"],
                    help="configuraciones a contrastar en la figura de exploración (factor=nivel)")
    ap.add_argument("--mapas", nargs="*", default=[], help="instancias cuya solución se dibuja")
    a = ap.parse_args()

    runs = cargar(a.res)
    if not runs:
        raise SystemExit(f"No hay JSON en {a.res}/")
    os.makedirs(a.out, exist_ok=True)
    df = tabla(runs)
    print(f"{len(runs)} ejecuciones cargadas: " + ", ".join(f"fase {k}: {v}" for k, v in df.fase.value_counts().items()))
    if not df.factible.all():
        print("AVISO: hay soluciones NO factibles:", df[~df.factible].shape[0])

    # Consistencia: una instancia (sin overrides) debe tener siempre el mismo grafo. Con grafos
    # descargados de OSM puede variar si cambia la caché de osmnx entre trabajos.
    base = {}
    for r in runs:
        if not r["problema"]:
            base.setdefault(r["instancia"], set()).add((r["info"]["n_nodos"], r["info"]["n_aristas"]))
    for inst, tallas in base.items():
        if len(tallas) > 1:
            print(f"AVISO: {inst} tiene grafos distintos entre ejecuciones (|V|,|E|): {sorted(tallas)}. "
                  f"Los resultados no son comparables; repite con la misma caché de osmnx.")

    tablas_instancias(runs, df, a.out)
    figuras_convergencia(runs, a.out)
    figuras_sensibilidad(df, a.out)
    figuras_impacto_problema(df, a.out)
    figura_exploracion(runs, a.explora, a.out)
    if a.mapas:
        figuras_mapas(runs, a.mapas, a.out)
    print("Generado en", a.out, ":", ", ".join(sorted(os.listdir(a.out))))


if __name__ == "__main__":
    main()
