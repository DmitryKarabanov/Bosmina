import re
import math
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.signal import lfilter

try:
    import allel
    ALLEL_AVAILABLE = True
except ImportError:
    ALLEL_AVAILABLE = False
    print("Warning: scikit-allel library not found. Install it via: pip install scikit-allel")

# ==============================================================================
# 1. SETTINGS
# ==============================================================================
NEXUS_FILE = "bosmina_popart.nex"
OUTPUT_CSV = "bosmina_full_popgen_stats.csv"
PAIRWISE_CSV = "bosmina_pairwise_phi_st.csv"
N_PERMUTATIONS = 1000   # permutations (AMOVA) / bootstrap replicates (SSD) per test
MIN_N_BOOT = 10         # minimal N for the parametric SSD bootstrap of a group

# ==============================================================================
# 2. NEXUS FILE PARSING
# ==============================================================================
def parse_nexus():
    print("Reading NEXUS file...")
    with open(NEXUS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    matrix_match = re.search(r'BEGIN DATA;.*?MATRIX\s*(.*?)\s*;\s*END;', content, re.DOTALL | re.IGNORECASE)
    seqs = {}
    seq_names_list = []
    for line in matrix_match.group(1).strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            seq_names_list.append(parts[0])
            seqs[parts[0]] = parts[1].upper()

    traits_match = re.search(r'BEGIN TRAITS;.*?Matrix\s*(.*?)\s*;\s*END;', content, re.DOTALL | re.IGNORECASE)
    trait_labels_match = re.search(r'TraitLabels\s+(.*?);', content, re.IGNORECASE)
    trait_labels = trait_labels_match.group(1).strip().split()

    seq_traits = {}
    for line in traits_match.group(1).strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            seq_name = parts[0]
            counts = [int(x) for x in parts[1].split(',')]
            seq_traits[seq_name] = dict(zip(trait_labels, counts))

    sequences = list(seqs.values())
    L = len(sequences[0])
    print(f"Loaded {len(sequences)} sequences of length {L} bp.")
    return sequences, seq_names_list, seq_traits, L

# ==============================================================================
# 3. HELPER FUNCTIONS AND IUPAC
# ==============================================================================
VALID = set('ACGT')
IUPAC = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
    'R': {'A','G'}, 'Y': {'C','T'}, 'S': {'G','C'}, 'W': {'A','T'},
    'K': {'G','T'}, 'M': {'A','C'}, 'B': {'C','G','T'}, 'D': {'A','G','T'},
    'H': {'A','C','T'}, 'V': {'A','C','G'}, 'N': {'A','C','G','T'},
}

def is_informative_site(seqs_list, pos):
    counts = defaultdict(int)
    for s in seqs_list:
        if s[pos] in VALID: counts[s[pos]] += 1
    return sum(1 for c in counts.values() if c >= 2) >= 2

def is_singleton_site(seqs_list, pos):
    counts = defaultdict(int)
    for s in seqs_list:
        if s[pos] in VALID: counts[s[pos]] += 1
    return len([c for c in counts.values() if c == 1]) == 1 and sum(counts.values()) > 0

def calc_distance_matrix(seq_list, L):
    """Calculates a simple p-distance matrix (IUPAC-aware)."""
    n = len(seq_list)
    dist_mat = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            d = sum(1 for pos in range(L)
                    if IUPAC.get(seq_list[i][pos], set()) and
                       IUPAC.get(seq_list[j][pos], set()) and
                       IUPAC.get(seq_list[i][pos], set()).isdisjoint(IUPAC.get(seq_list[j][pos], set())))
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat

def calc_tajima_d_allel(seq_list, L):
    """Calculates reference Tajima's D via scikit-allel."""
    if not ALLEL_AVAILABLE:
        return float('nan')

    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = []
    for pos in range(L):
        col = [base_map.get(seq[pos], -1) for seq in seq_list]
        if -1 not in col and len(set(col)) > 1:
            data.append(col)

    if not data:
        return float('nan')

    seq_array = np.array(data, dtype=np.int8)
    haps = allel.HaplotypeArray(seq_array)
    ac = haps.count_alleles()
    return float(allel.tajima_d(ac))

# ==============================================================================
# 3b. MISMATCH MODEL (Rogers & Harpending 1992) - module level, multiprocessing-safe
# ==============================================================================
def rogers_mismatch_pdf(x_vals, tau, theta_0):
    """
    Expected mismatch distribution under the sudden expansion model:
        P(x) = 1/(1+th) * sum_{i<=x} r^i * Poisson(x-i; tau),  r = th/(1+th)
    Computed via the IIR recursion A(x) = pois(x) + r*A(x-1) (scipy.signal.lfilter).
    """
    x = np.asarray(x_vals, dtype=float)
    theta_0 = max(float(theta_0), 0.0)
    r = theta_0 / (1.0 + theta_0)
    if tau <= 0:                                   # degenerate limit: geometric
        return (r ** x) / (1.0 + theta_0)
    pois = poisson.pmf(x, tau)
    A = lfilter([1.0], [1.0, -r], pois)            # A(x) = pois(x) + r*A(x-1)
    return A / (1.0 + theta_0)


def fit_rogers_model(P_obs):
    """
    Least-squares (OLS) fit of the sudden expansion model to a mismatch
    histogram. Multi-start for robustness.
    Returns (tau_hat, theta0_hat, P_exp, SSD).
    """
    P_obs = np.asarray(P_obs, dtype=float)
    if P_obs.size < 2 or P_obs.sum() <= 0:
        return float('nan'), float('nan'), None, float('nan')
    x = np.arange(P_obs.size, dtype=float)

    def objective(params):
        t, th = params
        return float(np.sum((P_obs - rogers_mismatch_pdf(x, t, th)) ** 2))

    k_bar = float(np.sum(x * P_obs))
    starts = [(max(k_bar, 0.5), 0.1)]
    if k_bar > 0:
        starts += [(k_bar / 2.0, 1.0), (k_bar * 2.0, 0.01)]

    best = None
    for s0 in starts:
        try:
            res = minimize(objective, s0, method='Nelder-Mead',
                           bounds=[(0.0, None), (0.0, None)],
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-12})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    if best is None:
        return float('nan'), float('nan'), None, float('nan')

    tau_hat = float(max(best.x[0], 0.0))
    theta_hat = float(max(best.x[1], 0.0))
    P_exp = rogers_mismatch_pdf(x, tau_hat, theta_hat)
    ssd = float(np.sum((P_obs - P_exp) ** 2))
    return tau_hat, theta_hat, P_exp, ssd

# ==============================================================================
# 3c. PARAMETRIC BOOTSTRAP: coalescent simulation under the fitted model
# ==============================================================================
def simulate_expansion_pairwise(n, tau, theta_anc, rng):
    """
    Coalescent simulation of n sequences under the fitted sudden-expansion
    model (parametric bootstrap):
      * present -> scaled time tau: no coalescence (population large after
        expansion; the same assumption as in the fitted pdf);
      * tau -> MRCA: standard coalescent with ancestral size theta_anc
        (coalescence rate k(k-1)/(2*theta_anc) per unit of scaled time);
      * infinite-sites mutations at rate 1/2 per lineage per unit time
        (=> E[pairwise diffs] = tau + theta_anc, exactly consistent with the
        fitted pdf, whose ancestral component is geometric with mean theta_0).
    Returns an (n x n) int32 matrix of pairwise differences.
    """
    theta_anc = max(float(theta_anc), 1e-8)
    tau = max(float(tau), 0.0)

    lineages = []                 # [tips_mask, birth_time]
    for i in range(n):
        m = np.zeros(n, dtype=bool)
        m[i] = True
        lineages.append([m, 0.0])
    branches = []                 # (length, tips_mask); MRCA branch excluded
    s = tau
    k = n
    while k > 1:
        rate = k * (k - 1) / (2.0 * theta_anc)
        s += rng.exponential(1.0 / rate)
        i, j = (int(v) for v in rng.choice(k, size=2, replace=False))
        for t in (i, j):
            m, b = lineages[t]
            branches.append((s - b, m))
        m_new = lineages[i][0] | lineages[j][0]
        hi, lo = (i, j) if i > j else (j, i)
        lineages.pop(hi)
        lineages.pop(lo)
        lineages.append([m_new, s])
        k -= 1

    if not branches:
        return np.zeros((n, n), dtype=np.int32)
    lengths = np.array([b[0] for b in branches], dtype=float)
    T = lengths.sum()
    if T <= 0:
        return np.zeros((n, n), dtype=np.int32)
    M = int(rng.poisson(T / 2.0))
    if M == 0:
        return np.zeros((n, n), dtype=np.int32)

    masks = np.stack([b[1] for b in branches]).astype(np.float32)   # (B, n)
    chosen = rng.choice(len(branches), size=M, p=lengths / T)

    # accumulate per-individual mutation counts and co-occurrence in chunks
    # (keeps memory at O(n * chunk) even for very long trees)
    pop = np.zeros(n, dtype=np.float64)
    shared = np.zeros((n, n), dtype=np.float32)
    CH = 4096
    for start in range(0, M, CH):
        sel = masks[chosen[start:start + CH]]       # (chunk, n)
        pop += sel.sum(axis=0)
        shared += sel.T @ sel                       # mutations shared by both members
    D = pop[:, None] + pop[None, :] - 2.0 * shared
    return D.astype(np.int32)


def worker_mismatch_parametric(args):
    """One bootstrap replicate: simulate under the fitted model, refit, return SSD."""
    n, tau, theta0, seed = args
    rng = np.random.default_rng(seed)
    D = simulate_expansion_pairwise(n, tau, theta0, rng)
    diffs = D[np.triu_indices(n, k=1)]
    counts = np.bincount(diffs)
    P_sim = counts / counts.sum()
    _, _, _, ssd_sim = fit_rogers_model(P_sim)
    return ssd_sim

# ==============================================================================
# 3d. AMOVA (Excoffier et al. 1992) - module level, multiprocessing-safe
# ==============================================================================
def assign_regions(names_list, traits_dict):
    """Assign every specimen (row) to the region with the largest trait count."""
    region_indices = defaultdict(list)
    for idx, name in enumerate(names_list):
        if name in traits_dict and traits_dict[name]:
            dom_region = max(traits_dict[name], key=traits_dict[name].get)
            region_indices[dom_region].append(idx)
        else:
            region_indices['Unknown'].append(idx)
    return region_indices


def _amova_components(D, labels, sizes, n_c):
    """
    Core of the one-level distance AMOVA (Excoffier et al. 1992).
      D      : (m, m) float64 symmetric, zero diagonal; for nucleotide
               difference counts d^2 == d in the one-hot encoding, so raw
               distances are the correct squared Euclidean distances;
      labels : (m,) int group id per individual;
      sizes  : (K,) group sizes (fixed; define df and n_c);
      n_c    : correction for unequal group sizes.
    Returns (Phi_ST, sigma2_among, sigma2_within).
    Note: sigma2_among (and hence Phi_ST) may be slightly negative if the
    groups are LESS differentiated than expected under random grouping -
    this is normal and is not clamped to zero.
    """
    m = D.shape[0]
    K = len(sizes)
    Z = np.zeros((m, K), dtype=np.float64)
    Z[np.arange(m), labels] = 1.0
    DZ = D @ Z
    Wg = np.sum(Z * DZ, axis=0)                 # ordered within-group distance sums
    ssd_w = 0.5 * float(np.sum(Wg / sizes))     # sum_g (1/n_g) * sum_{i<j in g} d_ij
    ssd_t = 0.5 * float(D.sum()) / m            # (1/N) * sum_{i<j} d_ij
    ssd_a = ssd_t - ssd_w
    ms_w = ssd_w / (m - K)                      # df within = N - K
    ms_a = ssd_a / (K - 1)                      # df among  = K - 1
    sigma_w = ms_w
    sigma_a = (ms_a - ms_w) / n_c
    sigma_t = sigma_a + sigma_w
    phi = sigma_a / sigma_t if sigma_t > 0 else float('nan')
    return phi, sigma_a, sigma_w


def amova_phi_st(dist_mat, group_indices):
    """Observed one-level AMOVA on a distance matrix for the given grouping."""
    n = dist_mat.shape[0]
    K = len(group_indices)
    if K < 2 or n - K < 1:
        return float('nan'), float('nan'), float('nan')
    sizes = np.array([len(g) for g in group_indices], dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    for gi, g in enumerate(group_indices):
        labels[np.asarray(g, dtype=np.int64)] = gi
    D = np.asarray(dist_mat, dtype=np.float64)
    N = float(n)
    n_c = (N * N - float(np.sum(sizes.astype(np.float64) ** 2))) / (N * (K - 1))
    return _amova_components(D, labels, sizes, n_c)


def init_worker_amova(dist_mat):
    global WORKER_D
    WORKER_D = np.asarray(dist_mat, dtype=np.float64)


def worker_amova_perm(args):
    """One permutation of individuals among regions (global K-group AMOVA)."""
    sizes, n_c, seed = args
    rng = np.random.default_rng(seed)
    K = len(sizes)
    labels = np.repeat(np.arange(K, dtype=np.int64), sizes)
    rng.shuffle(labels)                          # fixed group sizes, random assignment
    phi, _, _ = _amova_components(WORKER_D, labels, sizes, n_c)
    return phi


def init_worker_pairs(dist_mat, pair_specs):
    global WORKER_D, PAIR_MATS, PAIR_META
    WORKER_D = np.asarray(dist_mat, dtype=np.float64)
    PAIR_MATS = [WORKER_D[np.ix_(p['idx'], p['idx'])] for p in pair_specs]
    PAIR_META = [(p['n1'], p['n2'], p['n_c']) for p in pair_specs]


def worker_pair_phi_perm(args):
    """One permutation of individuals between two regions (pairwise Phi_ST)."""
    pair_i, seed = args
    rng = np.random.default_rng(seed)
    n1, n2, n_c = PAIR_META[pair_i]
    sizes = np.array([n1, n2], dtype=np.int64)
    labels = np.repeat(np.arange(2, dtype=np.int64), sizes)
    rng.shuffle(labels)
    phi, _, _ = _amova_components(PAIR_MATS[pair_i], labels, sizes, n_c)
    return phi

# ==============================================================================
# 4. MAIN STATISTICS CALCULATION
# ==============================================================================
def calc_advanced_stats(seq_list, names_list, traits_dict, L, label="Dataset", dist_mat=None):
    n = len(seq_list)
    if n < 2:
        return None

    if dist_mat is None:
        dist_mat = calc_distance_matrix(seq_list, L)

    unique_haps = len(set(seq_list))
    Tajima_D_allel = float('nan')

    # 4.1. Segregating sites
    S, S_singleton, S_inf = 0, 0, 0
    for pos in range(L):
        bases = [s[pos] for s in seq_list if s[pos] in VALID]
        if len(set(bases)) > 1:
            S += 1
            if is_singleton_site(seq_list, pos): S_singleton += 1
            if is_informative_site(seq_list, pos): S_inf += 1

    # 4.2. Pairwise distances
    n_pairs = n * (n - 1) // 2
    total_diff = float(np.sum(dist_mat[np.triu_indices(n, k=1)], dtype=np.int64))
    k_mean = total_diff / n_pairs if n_pairs > 0 else 0
    pi = k_mean / L if L > 0 else 0

    # 4.3. Haplotype diversity (h)
    hap_counts = defaultdict(int)
    for s in seq_list: hap_counts[s] += 1
    sum_fi2 = sum((c / n) ** 2 for c in hap_counts.values())
    h = (n / (n - 1)) * (1 - sum_fi2) if n > 1 else 0

    # 4.4. Theta and neutrality
    a1 = sum(1.0 / i for i in range(1, n))
    a2 = sum(1.0 / (i * i) for i in range(1, n))
    theta_w = (S / a1) / L if a1 > 0 and L > 0 else 0
    theta_pi = pi

    # --- Tajima's D ---
    D_stat = float('nan')
    if S > 0 and n > 2:
        theta_w_locus = S / a1
        b1 = (n + 1) / (3.0 * (n - 1))
        b2 = 2.0 * (n * n + n + 3) / (9.0 * n * (n - 1))
        c1 = b1 - 1.0 / a1
        c2 = b2 - (n + 2.0) / (a1 * n) + a2 / (a1 * a1)
        e1 = c1 / a1
        e2 = c2 / (a1 * a1 + a2)
        var_D = e1 * S + e2 * S * (S - 1)
        if var_D > 0:
            D_stat = (k_mean - theta_w_locus) / math.sqrt(var_D)
        Tajima_D_allel = calc_tajima_d_allel(seq_list, L)

    # --- Fu's Fs (Fu 1997; log-odds form, as in Arlequin/DnaSP) + p-value ---
    Fs, p_Fs = float('nan'), float('nan')
    theta_locus = k_mean          # theta estimated by pi (per-locus, as in Arlequin)
    if n >= 10 and S > 0 and theta_locus > 0:
        # Ewens/CRP recursion: distribution of the number of haplotypes K
        P = [0.0] * (n + 1)
        P[1] = 1.0
        for i in range(2, n + 1):
            new_P = [0.0] * (n + 1)
            denom = theta_locus + i - 1
            for kk in range(1, i + 1):
                term1 = P[kk - 1] * theta_locus / denom if kk > 1 else 0.0
                term2 = P[kk] * (i - 1) / denom if kk <= i - 1 else 0.0
                new_P[kk] = term1 + term2
            P = new_P
        f = sum(P[unique_haps:n + 1])          # f = P(K >= observed) under neutrality
        if f <= 0.0:
            Fs, p_Fs = float('-inf'), 0.0
        elif f >= 1.0:
            Fs, p_Fs = float('inf'), 1.0
        else:
            Fs = math.log(f / (1.0 - f))       # Fs < 0: excess of haplotypes (expansion)
            p_Fs = f                           # one-sided p for the EXPANSION direction

    # 4.5. Mismatch Distribution
    mismatch_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            mismatch_counts[dist_mat[i, j]] += 1
    max_diff = max(mismatch_counts.keys()) if mismatch_counts else 0
    P_obs = np.array([mismatch_counts[i] / n_pairs for i in range(max_diff + 1)])

    raggedness, ssd = float('nan'), float('nan')
    tau_hat, theta_hat = float('nan'), float('nan')
    if len(P_obs) >= 2:
        tau_hat, theta_hat, P_exp, ssd = fit_rogers_model(P_obs)
        if not math.isnan(ssd):
            # Harpending's raggedness: squared deviation of the OBSERVED histogram
            # from the EXPECTED (Rogers) one; identical to SSD under the OLS fit.
            raggedness = ssd

    # 4.6. AMOVA (1-level), only for the pooled dataset
    Phi_ST, var_among, var_within, var_total = (float('nan'), float('nan'),
                                                float('nan'), float('nan'))
    if label == "ALL SAMPLES" and len(traits_dict) > 0:
        region_indices = assign_regions(names_list, traits_dict)
        valid_regions = {k2: v for k2, v in sorted(region_indices.items()) if len(v) >= 2}
        if len(valid_regions) >= 2:
            Phi_ST, var_among, var_within = amova_phi_st(dist_mat, list(valid_regions.values()))
            var_total = var_among + var_within

    return {
        'label': label, 'N': n, 'nhap': unique_haps, 'S': S, 'S_single': S_singleton,
        'S_inf': S_inf, 'h': h, 'pi': pi, 'k': k_mean, 'theta_w': theta_w,
        'theta_pi': theta_pi, 'Tajima_D': D_stat, 'Tajima_D_allel': Tajima_D_allel,
        'Fu_Fs': Fs, 'p_Fs': p_Fs, 'tau': tau_hat, 'theta0': theta_hat,
        'raggedness': raggedness, 'SSD': ssd, 'Phi_ST': Phi_ST,
        'var_among_pct': (var_among / var_total * 100) if not math.isnan(var_total) and var_total > 0 else float('nan'),
        'var_within_pct': (var_within / var_total * 100) if not math.isnan(var_total) and var_total > 0 else float('nan'),
        'p_Phi_ST': float('nan'), 'p_raggedness': float('nan'), 'p_SSD': float('nan')
    }

# ==============================================================================
# 5. PERMUTATION / BOOTSTRAP TESTS
# ==============================================================================
def run_global_amova_perm(dist_mat, region_indices, phi_obs, N_PERM):
    total_cores = os.cpu_count() or 4
    n_cores = max(1, total_cores - 1)
    valid_regions = {k: v for k, v in sorted(region_indices.items()) if len(v) >= 2}
    if len(valid_regions) < 2:
        print("  Global AMOVA permutations skipped (fewer than 2 regions with N >= 2).")
        return float('nan')
    sizes = np.array([len(v) for v in valid_regions.values()], dtype=np.int64)
    n = int(sizes.sum())
    K = len(sizes)
    n_c = (float(n) * n - float(np.sum(sizes.astype(np.float64) ** 2))) / (float(n) * (K - 1))
    args_list = [(sizes, n_c, 100_000 + i) for i in range(N_PERM)]
    print(f"  Global AMOVA permutations (K={K}, N={n}, {N_PERM} replicates)... ", end="", flush=True)
    with ProcessPoolExecutor(max_workers=n_cores, initializer=init_worker_amova,
                             initargs=(dist_mat,)) as executor:
        results = list(executor.map(worker_amova_perm, args_list))
    results = [r for r in results if not math.isnan(r)]
    if not results:
        print("done! (all replicates degenerate)")
        return float('nan')
    count_extreme = sum(1 for r in results if r >= phi_obs)
    p = (count_extreme + 1) / (len(results) + 1)
    print(f"done! p(Phi_ST) = {p:.4g}")
    return p


def run_pairwise_phi_st(dist_mat, region_indices, N_PERM):
    """Observed + permutation p-values for Phi_ST between all pairs of regions."""
    total_cores = os.cpu_count() or 4
    n_cores = max(1, total_cores - 1)
    region_names = sorted(k for k in region_indices if len(region_indices[k]) >= 2)
    pairs = []
    for a in range(len(region_names)):
        for b in range(a + 1, len(region_names)):
            na, nb = region_names[a], region_names[b]
            ia, ib = list(region_indices[na]), list(region_indices[nb])
            n1, n2 = len(ia), len(ib)
            m = n1 + n2
            idx = np.asarray(ia + ib, dtype=np.int64)
            n_c = (float(m) * m - float(n1 * n1) - float(n2 * n2)) / float(m)   # K = 2
            pairs.append({'a': na, 'b': nb, 'idx': idx, 'n1': n1, 'n2': n2, 'n_c': n_c})
    if not pairs:
        print("  Pairwise Phi_ST skipped (fewer than 2 regions with N >= 2).")
        return []

    for p in pairs:
        sub = dist_mat[np.ix_(p['idx'], p['idx'])]
        groups = [np.arange(p['n1']), np.arange(p['n1'], p['n1'] + p['n2'])]
        p['phi_obs'], _, _ = amova_phi_st(sub, groups)

    args_list = [(pair_i, 300_000 + pair_i * N_PERM + j)
                 for pair_i in range(len(pairs)) for j in range(N_PERM)]
    print(f"  Pairwise Phi_ST permutations ({len(pairs)} pairs x {N_PERM} replicates)... ",
          end="", flush=True)
    with ProcessPoolExecutor(max_workers=n_cores, initializer=init_worker_pairs,
                             initargs=(dist_mat, pairs)) as executor:
        results = list(executor.map(worker_pair_phi_perm, args_list))

    counts = [0] * len(pairs)
    valid = [0] * len(pairs)
    for (pair_i, _), phi_perm in zip(args_list, results):
        if not math.isnan(phi_perm):
            valid[pair_i] += 1
            if phi_perm >= pairs[pair_i]['phi_obs']:
                counts[pair_i] += 1
    for i, p in enumerate(pairs):
        if valid[i] == 0 or math.isnan(p['phi_obs']):
            p['p_val'] = float('nan')
        else:
            p['p_val'] = (counts[i] + 1) / (valid[i] + 1)
    print("done!")
    return pairs


def run_ssd_bootstraps(results, N_PERM, min_n=MIN_N_BOOT):
    """Parametric bootstrap p-values for SSD/raggedness of every eligible group."""
    total_cores = os.cpu_count() or 4
    n_cores = max(1, total_cores - 1)
    tasks, task_group = [], []
    for gi, res in enumerate(results):
        n, tau, th0 = res['N'], res['tau'], res['theta0']
        if n >= min_n and not math.isnan(tau) and not math.isnan(res['SSD']):
            for j in range(N_PERM):
                tasks.append((n, tau, th0, 500_000 + gi * 1_000_000 + j))
                task_group.append(gi)
        else:
            reason = f"N={n} < MIN_N_BOOT={min_n}" if n < min_n else "mismatch model fit failed"
            print(f"  SSD bootstrap skipped for '{res['label']}' ({reason}).")
    if not tasks:
        return
    n_groups = len(set(task_group))
    print(f"  Parametric bootstrap for SSD ({n_groups} groups x {N_PERM} replicates)... ",
          end="", flush=True)
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results_boot = list(executor.map(worker_mismatch_parametric, tasks))
    counts, valid = defaultdict(int), defaultdict(int)
    for gi, ssd_sim in zip(task_group, results_boot):
        if ssd_sim is not None and not math.isnan(ssd_sim):
            valid[gi] += 1
            if ssd_sim >= results[gi]['SSD']:
                counts[gi] += 1
    print("done!")
    for gi in sorted(valid):
        if valid[gi] > 0:
            p = (counts[gi] + 1) / (valid[gi] + 1)
            results[gi]['p_SSD'] = p
            results[gi]['p_raggedness'] = p
            print(f"     {results[gi]['label']}: p(SSD) = p(raggedness) = {p:.4g}")

# ==============================================================================
# 6. MAIN EXECUTION BLOCK (REQUIRED FOR WINDOWS!)
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("FULL POPULATION GENETIC ANALYSIS (p-distances + Rogers SSD) - v3")
    print("=" * 80)
    sequences, seq_names_list, seq_traits, L = parse_nexus()
    results = []

    print("Calculating observed values for the ENTIRE SAMPLE...")
    dist_mat_full = calc_distance_matrix(sequences, L)
    stats_all = calc_advanced_stats(sequences, seq_names_list, seq_traits, L,
                                    "ALL SAMPLES", dist_mat=dist_mat_full)
    if stats_all:
        results.append(stats_all)

    region_seqs, region_names = defaultdict(list), defaultdict(list)
    for name, seq in zip(seq_names_list, sequences):
        if name in seq_traits:
            for region, count in seq_traits[name].items():
                for _ in range(count):
                    region_seqs[region].append(seq)
                    region_names[region].append(name)

    for region in sorted(region_seqs.keys()):
        if len(region_seqs[region]) >= 2:
            print(f"Calculating for region: {region}...")
            stats = calc_advanced_stats(region_seqs[region], region_names[region], {}, L,
                                        f"REGION: {region}")
            if stats:
                results.append(stats)

    # ------------------- significance tests -------------------
    region_indices = assign_regions(seq_names_list, seq_traits)

    if stats_all and not math.isnan(stats_all.get('Phi_ST', float('nan'))):
        stats_all['p_Phi_ST'] = run_global_amova_perm(dist_mat_full, region_indices,
                                                      stats_all['Phi_ST'], N_PERMUTATIONS)
    else:
        print("  Global AMOVA skipped (not computable for this dataset).")

    pairs = run_pairwise_phi_st(dist_mat_full, region_indices, N_PERMUTATIONS)

    run_ssd_bootstraps(results, N_PERMUTATIONS, min_n=MIN_N_BOOT)

    # ------------------- output -------------------
    def print_stats(s):
        print(f"\n{s['label']} (N={s['N']}, hap={s['nhap']})")
        print(f"  {'-' * 70}")
        print(f"  Polymorphic sites (S):         {s['S']} (singletons: {s['S_single']}, pars-inf: {s['S_inf']})")
        print(f"  Haplotype diversity (h):       {s['h']:.4f}")
        print(f"  Nucleotide diversity (pi):     {s['pi'] * 1000:.4f} x 10^-3 (p-distance)")
        print(f"  Tajima's D (our script):       {s['Tajima_D']:.4f}")
        if not math.isnan(s.get('Tajima_D_allel', float('nan'))):
            print(f"  Tajima's D (scikit-allel):     {s['Tajima_D_allel']:.4f} *")
        pF = s.get('p_Fs', float('nan'))
        pF_str = f"{pF:.4g}" if not math.isnan(pF) else "n/a"
        print(f"  Fu's Fs:                       {s['Fu_Fs']:.4f}   (p_expansion = {pF_str})")
        if not math.isnan(s.get('tau', float('nan'))):
            fit_mean = s['tau'] + s['theta0']
            print(f"  Mismatch fit:                  tau={s['tau']:.2f}, theta0={s['theta0']:.2f}  "
                  f"(fit mean {fit_mean:.1f} vs observed k = {s['k']:.1f})")
        print(f"  Mismatch Raggedness (r):       {s['raggedness']:.4f}")
        print(f"  Mismatch SSD:                  {s['SSD']:.4f}")
        if not math.isnan(s.get('p_SSD', float('nan'))):
            print(f"  +-- p(SSD = raggedness):       {s['p_SSD']:.4g}  (> 0.05: expansion model not rejected)")
        if s['label'] == "ALL SAMPLES" and not math.isnan(s.get('Phi_ST', float('nan'))):
            print(f"  +-- AMOVA (1-level):")
            print(f"  |  Variation Among Regions:   {s['var_among_pct']:.2f}%")
            print(f"  |  Variation Within Regions:  {s['var_within_pct']:.2f}%")
            print(f"  +-- Phi_ST:                    {s['Phi_ST']:.4f}  (p={s.get('p_Phi_ST', float('nan')):.4g})")

    for res in results:
        print_stats(res)

    if pairs:
        print("\n" + "-" * 80)
        print("PAIRWISE Phi_ST (permutation test, individuals permuted between regions):")
        for p in pairs:
            print(f"  {p['a']:<12} vs {p['b']:<12} (N={p['n1']}+{p['n2']}):  "
                  f"Phi_ST = {p['phi_obs']:.4f}   p = {p['p_val']:.4g}")
        with open(PAIRWISE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['region_a', 'n_a', 'region_b', 'n_b', 'Phi_ST', 'p_value'])
            for p in pairs:
                writer.writerow([p['a'], p['n1'], p['b'], p['n2'],
                                 f"{p['phi_obs']:.4f}", f"{p['p_val']:.4g}"])
        print(f"\nPairwise Phi_ST saved to '{PAIRWISE_CSV}'")

    fieldnames = ['label', 'N', 'nhap', 'S', 'S_single', 'S_inf', 'h', 'pi', 'k',
                  'theta_w', 'theta_pi', 'Tajima_D', 'Tajima_D_allel', 'Fu_Fs', 'p_Fs',
                  'tau', 'theta0', 'raggedness', 'SSD', 'p_raggedness', 'p_SSD',
                  'Phi_ST', 'p_Phi_ST', 'var_among_pct', 'var_within_pct']
    p_cols = {'p_Fs', 'p_SSD', 'p_raggedness', 'p_Phi_ST'}

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {}
            for k, v in res.items():
                if isinstance(v, (float, np.floating)):
                    row[k] = f"{v:.4g}" if k in p_cols else f"{v:.4f}"
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"\nResults successfully saved to '{OUTPUT_CSV}'")
    print("=" * 80)

    # Legend
    legend = {
        'label': 'Name of the analyzed group',
        'N': 'Sample size (number of sequences)',
        'nhap': 'Number of unique haplotypes',
        'S': 'Number of polymorphic sites',
        'S_single': 'Number of singleton (single-copy) segregating sites',
        'S_inf': 'Parsimony-informative sites',
        'h': 'Haplotype diversity',
        'pi': 'Nucleotide diversity (p-distance, per site)',
        'k': 'Mean absolute number of pairwise differences',
        'theta_w': "Watterson's Theta (per site)",
        'theta_pi': "Tajima's Theta (per site)",
        'Tajima_D': "Tajima's D-statistic (our calculation)",
        'Tajima_D_allel': "Tajima's D-statistic (reference scikit-allel)",
        'Fu_Fs': "Fu's Fs (log-odds form; negative = excess of haplotypes / expansion)",
        'p_Fs': ("One-sided P-value for Fu's Fs: P(K >= observed haplotypes) under Ewens "
                 "neutrality with theta = pi. Small = significant haplotype excess "
                 "(expansion); values near 1.0 indicate a deficit instead (bottleneck/"
                 "structure); for the deficit direction use 1 - p_Fs."),
        'tau': 'Mismatch fit: time since sudden expansion (units of pairwise differences)',
        'theta0': 'Mismatch fit: ancestral population theta',
        'raggedness': ("Harpending's raggedness (squared deviation of observed histogram "
                       "from the expected Rogers spectrum; identical to SSD under OLS fit)"),
        'SSD': 'Sum of squared deviations from the fitted Rogers expansion model',
        'p_raggedness': ('Parametric-bootstrap P-value (equals p_SSD; coalescent simulation '
                         'under the fitted model with refitting, computed per group)'),
        'p_SSD': ('Parametric-bootstrap P-value for SSD under the fitted sudden-expansion '
                  'model (simulated with fitted tau/theta0, model refitted per replicate)'),
        'Phi_ST': ('Fixation index (1-level AMOVA, Excoffier et al. 1992; normalized '
                   'SSD_T = (1/N)*sum d_ij, SSD_W = sum_g (1/n_g)*sum d_ij; can be '
                   'slightly negative if regions are less differentiated than random)'),
        'p_Phi_ST': 'P-value for Phi_ST (permutation of individuals among regions)',
        'var_among_pct': '% of variance among regions (sigma2_among / sigma2_total)',
        'var_within_pct': '% of variance within regions'
    }
    legend_file = "bosmina_columns_legend.csv"
    with open(legend_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Column", "Description"])
        for col, desc in legend.items():
            writer.writerow([col, desc])
    print(f"Legend saved to '{legend_file}'")