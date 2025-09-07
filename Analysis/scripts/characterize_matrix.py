import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from util.load_experiment import load_all_matrices

# ---------- utilities ----------
def _offdiag_vals(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    x = M[mask]
    return x[np.isfinite(x)]

def _row_offdiag(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # returns (row_min, row_sum) excluding diag and infs
    n = M.shape[0]
    row_min = np.full(n, np.nan)
    row_sum = np.zeros(n)
    for i in range(n):
        row = np.array([v for j, v in enumerate(M[i]) if j != i and np.isfinite(v)], dtype=float)
        if row.size:
            row_min[i] = row.min()
            row_sum[i] = row.sum()
    return row_min, row_sum

def _entropy(x: np.ndarray, bins: int = 30) -> float:
    if x.size == 0: return np.nan
    cnts, _ = np.histogram(x, bins=bins, density=True)
    p = cnts / cnts.sum() if cnts.sum() > 0 else cnts
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def _gini(x: np.ndarray) -> float:
    if x.size == 0: return np.nan
    x = np.sort(x[x >= 0])
    if x.size == 0: return np.nan
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx.sum() / cumx[-1])) / n)

def _asymmetry(M: np.ndarray) -> float:
    # mean |d_ij - d_ji| normalized by median off-diagonal distance
    D = np.abs(M - M.T)
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    diffs = D[mask]
    diffs = diffs[np.isfinite(diffs)]
    base = np.median(_offdiag_vals(M))
    if base == 0 or not np.isfinite(base): return np.nan
    return float(np.mean(diffs) / base) if diffs.size else 0.0

def _triangle_violations(M: np.ndarray, max_samples: int = 200_000, rng=np.random.default_rng(0)) -> Dict[str, float]:
    n = M.shape[0]
    # compute exact if small; otherwise sample triples (i,j,k), all distinct
    def severity(i, j, k):
        dij = M[i, j]; dik = M[i, k]; dkj = M[k, j]
        if not (np.isfinite(dij) and np.isfinite(dik) and np.isfinite(dkj)): return 0.0
        return max(0.0, dij - (dik + dkj))
    if n <= 60:
        total = 0; vcount = 0; vsum = 0.0
        for i in range(n):
            for j in range(n):
                if j == i: continue
                for k in range(n):
                    if k == i or k == j: continue
                    s = severity(i, j, k)
                    if s > 0: vcount += 1; vsum += s
                    total += 1
    else:
        total = max_samples
        vcount = 0; vsum = 0.0
        for _ in range(max_samples):
            i, j, k = rng.choice(n, size=3, replace=False)
            s = severity(i, j, k)
            if s > 0: vcount += 1; vsum += s
    rate = vcount / total if total else np.nan
    sev_mean = vsum / vcount if vcount else 0.0
    return {"tri_violation_rate": float(rate), "tri_violation_severity": float(sev_mean)}

def _mst_weight_sym_min(M: np.ndarray) -> Tuple[float, float]:
    # undirected proxy: w_ij = min(d_ij, d_ji). Prim's algorithm (O(n^2))
    n = M.shape[0]
    W = np.minimum(M, M.T)
    # replace inf with large number
    X = W.copy()
    X[~np.isfinite(X)] = np.inf
    in_mst = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    key[0] = 0.0
    total = 0.0
    edges = 0
    for _ in range(n):
        u = np.argmin(key)
        if not np.isfinite(key[u]): break
        in_mst[u] = True
        total += key[u]
        key[u] = np.inf
        edges += 1
        # relax
        for v in range(n):
            if not in_mst[v] and np.isfinite(X[u, v]) and X[u, v] < key[v]:
                key[v] = X[u, v]
    # subtract the first added 0 and compute avg edge
    total = float(total)
    avg = total / max(1, edges - 1)
    return total, avg

def _one_shot_reduction_lb(M: np.ndarray) -> float:
    # Row reduction then column reduction LB (B&B-style rough proxy)
    A = M.copy()
    A[~np.isfinite(A)] = np.inf
    # row mins
    rmins = np.min(A, axis=1)
    rmins[~np.isfinite(rmins)] = 0.0
    LB = np.sum(rmins[np.isfinite(rmins)])
    A = A - rmins[:, None]
    # col mins
    cmins = np.min(A, axis=0)
    cmins[~np.isfinite(cmins)] = 0.0
    LB += np.sum(cmins[np.isfinite(cmins)])
    return float(LB)

def compute_features(M: np.ndarray, knn_k: int = 3) -> Dict[str, Any]:
    n = M.shape[0]
    assert M.shape[0] == M.shape[1]
    # collect off-diagonal finite distances
    x = _offdiag_vals(M)
    if x.size == 0:
        return {"n": n}
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    cv = float(np.std(x) / p50) if p50 > 0 else np.nan
    gini = _gini(x)
    ent = _entropy(x, bins=30)
    # neighborhood stats
    row_min, row_sum = _row_offdiag(M)
    nn_mean = np.nanmean(row_min); nn_std = np.nanstd(row_min); row_sum_cv = float(np.std(row_sum)/np.mean(row_sum)) if np.mean(row_sum) > 0 else np.nan
    # kNN mean
    knn_vals = []
    for i in range(n):
        row = np.array([v for j, v in enumerate(M[i]) if j != i and np.isfinite(v)], dtype=float)
        if row.size:
            knn_vals.append(np.sort(row)[:min(knn_k, row.size)].mean())
    knn_mean = float(np.mean(knn_vals)) if knn_vals else np.nan
    # epsilon density (epsilon = 25th percentile)
    eps = np.percentile(x, 25)
    eps_density = float((x <= eps).mean()) if np.isfinite(eps) else np.nan
    # asymmetry
    asym = _asymmetry(M)
    # triangle metrics
    tri = _triangle_violations(M)
    # MST proxy
    mst_total, mst_avg = _mst_weight_sym_min(M)
    # reduction LB
    red_lb = _one_shot_reduction_lb(M)
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "median": float(p50),
        "std": float(np.std(x)),
        "cv": cv,
        "iqr": float(iqr),
        "p10": float(p10), "p90": float(p90),
        "entropy": float(ent),
        "gini": float(gini),
        "nn_mean": float(nn_mean), "nn_std": float(nn_std),
        "row_sum_cv": float(row_sum_cv),
        "knn3_mean": float(knn_mean),
        "eps25_density": float(eps_density),
        "asymmetry": float(asym),
        **tri,
        "mst_total": float(mst_total), "mst_avg": float(mst_avg),
        "reduction_lb": float(red_lb),
        "unique_ratio": float(len(np.unique(x)) / x.size)
    }

def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    # Cohen's d
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2: return np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    n1, n2 = a.size, b.size
    sp = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    return (m2 - m1) / sp if sp > 0 else np.nan

def characterize_easy_hard(by: Tuple[str, ...] = None, hardest_q=0.95, easiest_q=0.05) -> Dict[str, pd.DataFrame]:
    df = load_all_matrices()
    # keep rows with a real iteration and a square numeric matrix with inf diagonal
    df = df[df['iteration'].notna()].copy()
    def diag_inf(M):
        try:
            d = np.diag(M)
            return np.all(np.isinf(d))
        except Exception:
            return False
    df = df[df['matrix'].apply(lambda M: isinstance(M, np.ndarray) and M.ndim == 2 and M.shape[0]==M.shape[1] and diag_inf(M))]
    # optionally stratify by config to reduce confounding
    if by:
        parts = []
        for keys, sub in df.groupby(list(by)):
            thr_hi = sub['iteration'].quantile(hardest_q)
            thr_lo = sub['iteration'].quantile(easiest_q)
            parts.append(sub.assign(
                bucket=np.where(sub['iteration']>=thr_hi, 'hard', np.where(sub['iteration']<=thr_lo, 'easy', 'mid'))
            ))
        df = pd.concat(parts, ignore_index=True)
    else:
        thr_hi = df['iteration'].quantile(hardest_q)
        thr_lo = df['iteration'].quantile(easiest_q)
        df['bucket'] = np.where(df['iteration']>=thr_hi, 'hard', np.where(df['iteration']<=thr_lo, 'easy', 'mid'))
    # compute features
    feats = []
    for i, row in df.iterrows():
        if row['bucket'] in ('easy','hard'):
            f = compute_features(row['matrix'])
            f.update({'bucket': row['bucket'], 'iteration': row['iteration']})
            # carry a few config knobs for later slicing
            for k in ('generation_type','distribution','city_size','range','mutation_type'):
                if k in row: f[k] = row[k]
            feats.append(f)
    F = pd.DataFrame(feats)
    # summarize by bucket
    num_cols = [c for c in F.columns if c not in ('bucket','generation_type','distribution','city_size','range','mutation_type','iteration')]
    summary = F.groupby('bucket')[num_cols].mean().T.sort_index()
    # effect sizes (easy -> hard)
    es = []
    for c in num_cols:
        es.append({'feature': c,
                   'easy_mean': F.loc[F.bucket=='easy', c].mean(),
                   'hard_mean': F.loc[F.bucket=='hard', c].mean(),
                   'cohens_d': effect_size(F.loc[F.bucket=='easy', c].to_numpy(),
                                           F.loc[F.bucket=='hard', c].to_numpy())})
    effects = pd.DataFrame(es).sort_values('cohens_d', ascending=False)
    return {'per_matrix': F, 'summary': summary, 'effects': effects}

# Option A: global 5% split
out = characterize_easy_hard()
# Option B: stratify by config (recommended to control confounding)
out = characterize_easy_hard(by=('generation_type','distribution','city_size'))

F = out['per_matrix']      # features per matrix (only easy/hard)
S = out['summary']         # mean(feature) per bucket
E = out['effects']         # easy vs hard effect sizes
print(S.head())
print(E.head(12))

"""
Triangle violation rate & severity: hard non-metric instances often “break” greedy structure; conversely, earlier observation was that the hardest TSP found had few violations—this feature will surface that contrast cleanly.

Asymmetry index: ATSP hardness often correlates with strong directionality; this gives a scalar for it.

Gini / entropy / unique_ratio: heavy-tailed or overly quantized edge sets behave differently for branching and pruning.

Nearest-neighbor / epsilon-density / MST: capture clusteriness and bottlenecks—good tour structure vs scattered outliers.

Reduction LB: a cheap proxy for how much “free” lower bound we get from one pass of reductions (good for B&B).


Difference in Lital iterations
Difference in easiest and hardest matrices
Standard deviation and mean matrices, and compare the easiest and hardest with those metrics
"""