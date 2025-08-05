# ---------------- Imports -----------------
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as hm
from pm4py.algo.discovery.inductive import algorithm as im_alg
from pm4py.algo.discovery.ilp       import algorithm as ilp_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision      import algorithm as precision
from pm4py.algo.evaluation.generalization import algorithm as generalization
from pm4py.algo.evaluation.simplicity     import algorithm as simplicity
import pm4py, time, pathlib, pandas as pd, glob
import random
from pm4py.objects.log.obj import EventLog

# -----------------------------------------------------------
# Soundness
# -----------------------------------------------------------
def simple_wfnet_soundness(net, imark, fmark):
    start = [p for p in net.places if not p.in_arcs]
    end   = [p for p in net.places if not p.out_arcs]
    if len(start) != 1 or len(end) != 1:
        return False
    s, t = start[0], end[0]
    visited, stack = set(), [s]
    while stack:
        n = stack.pop()
        if n in visited: continue
        visited.add(n)
        for a in n.out_arcs:               # gilt für Place *und* Transition
            stack.append(a.target)
    all_nodes = set(net.places) | set(net.transitions)
    return visited == all_nodes and t in visited

# -----------------------------------------------------------
# Laufzeit-Wrapper
# -----------------------------------------------------------
def timed_discovery(miner_func, *args, **kwargs):
    t0 = time.perf_counter()
    net, imark, fmark = miner_func(*args, **kwargs)
    return net, imark, fmark, time.perf_counter() - t0

# -----------------------------------------------------------
# Kennzahlen
# -----------------------------------------------------------
def _to_float(x):
    """
    Holt den numerischen Wert aus Float oder Dict
    (unterstützt alle PM4Py-Varianten).
    """
    if isinstance(x, (float, int)):
        return float(x)

    if isinstance(x, dict):
        key_order = [
            "averageFitness",         
            "fitness",
            "average_trace_fitness",   
            "log_fitness",
            "precision",              
            "et_precision",
        ]
        for k in key_order:
            if k in x:
                return float(x[k])
    raise TypeError(f"Unbekanntes Format: {type(x)} – {x}")


def calc_metrics(log, net, imark, fmark, elapsed, label):
    # ---------- Fitness -------------------------------------
    fit_raw = replay_fitness.apply(
            log, net, imark, fmark,
            variant=replay_fitness.Variants.TOKEN_BASED)
    fit = _to_float(fit_raw)

# ---------- Precision -----------------------------------
    prec_raw = precision.apply(
            log, net, imark, fmark,
            variant=precision.Variants.ETCONFORMANCE_TOKEN)

    prec = _to_float(prec_raw)
    gen  = generalization.apply(log, net, imark, fmark)
    simp = simplicity.apply(net)
    fscore = 2*fit*prec/(fit+prec) if fit + prec else 0
    sound  = simple_wfnet_soundness(net, imark, fmark)



    return {
        "Model":  label,
        "Fitness": round(fit,3),
        "Precision": round(prec,3),
        "FScore": round(fscore,3),
        "Generalization": round(gen,3),
        "Simplicity": round(simp,3),
        "Time_s": round(elapsed,2),
        "Sound": sound,
        "Places": len(net.places),
        "Transitions": len(net.transitions)
    }

# -----------------------------------------------------------
# Parameter
# -----------------------------------------------------------
hm_params  = {"dependency_thresh":0.30,
              "and_measure_thresh":0.65,
              "min_act_count":3,
              "return_petri_net":True}

ilp_params = {"alpha":0.05,
              "variant":"ilp2",
              "apply_noise_filter":True}

# -----------------------------------------------------------
# Log-Ordner anpassen
# -----------------------------------------------------------
LOG_DIR = pathlib.Path(r"C:\Users\robin\OneDrive - informatik.hs-fulda.de\Master\Semester3\Wissenschaftliches Arbeiten")   
log_files = sorted(LOG_DIR.glob("*.xes"))

rows = []

for file in log_files:
    log_name = file.stem
    print(f" Bearbeite {log_name} …")
    log = xes_importer.apply(str(file))

    # ---------- HM classic ----------
    net_hm, im_hm, fm_hm, t_hm = timed_discovery(
        hm.apply, log, variant=hm.Variants.CLASSIC, parameters=hm_params)
    rows.append(calc_metrics(log, net_hm, im_hm, fm_hm, t_hm,
                             f"HM-Classic | {log_name}"))

    # ---------- Inductive Miner f ----------
    net_im, im_im, fm_im, t_im = timed_discovery(
        pm4py.discover_petri_net_inductive,
        log, noise_threshold=0.30)
    rows.append(calc_metrics(log, net_im, im_im, fm_im, t_im,
                             f"IMf | {log_name}"))

    # ---------- ILP Miner 2 ----------
    net_ilp, im_ilp, fm_ilp, t_ilp = timed_discovery(
        ilp_miner.apply, log, parameters=ilp_params)
    rows.append(calc_metrics(log, net_ilp, im_ilp, fm_ilp, t_ilp,
                             f"ILP2 | {log_name}"))

# -----------------------------------------------------------
# Tabellen ausgeben
# -----------------------------------------------------------
df_all = pd.DataFrame(rows)
print("\n### Einzel-Ergebnisse")
print(df_all.to_markdown(index=False))

# Miner-Namen herausziehen 
df_all["Miner"] = df_all["Model"].str.split(r"\s*\|\s*", n=1).str[0]

# Spalten, die numerisch gemittelt werden sollen
num_cols = ["Fitness", "Precision", "FScore",
            "Generalization", "Simplicity",
            "Time_s", "Places", "Transitions"]

# Mittelwert 
summary = (df_all.groupby("Miner")[num_cols]
           .mean()
           .round(3)
           .reset_index())

# Ausgabe 
print("\n### Durchschnitt über alle Logs (je Algorithmus)")
print(summary.to_markdown(index=False))