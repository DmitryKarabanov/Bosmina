# -*- coding: utf-8 -*-
"""
FULL POPULATION GENETIC ANALYSIS (p-distances + Rogers mismatch model)

Corrected version:
  * Fu's Fs: proper log-odds formula Fs = ln(f/(1-f)), where f = P(K >= k_obs)
  * Harpending's raggedness: deviations of the observed histogram from the
    EXPECTED (Rogers) spectrum; under an OLS fit it coincides with SSD
  * SSD significance: parametric bootstrap via coalescent simulation under the
    fitted sudden-expansion model, with model refitting in every replicate
  * AMOVA Phi_ST: Excoffier et al. (1992) with df and the unequal group-size
    correction n_c; significance via permutation of individuals among regions

References:
  Fu Y.X. (1997) Genetics 147:915-925
  Rogers A.R., Harpending H. (1992) Mol Biol Evol 9:552-569
  Harpending H.C. (1994) Genetics 139:1117-1123
  Excoffier L., Smouse P.E., Quattro J.M. (1992) Genetics 131:479-491
"""

import re
import math
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

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
N_PERMUTATIONS = 1000

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
    Computed via the stable recursion A(x) = pois(x) + r*A(x-1).
    """
    x = np.asarray(x_vals, dtype=float)
    theta_0 = max(float(theta_0), 0.0)
    r = theta_0 / (1.0 + theta_0)
    if tau <= 0:                                   # degenerate limit: geometric
        return (r ** x) / (1.0 + theta_0)
    pois = poisson.pmf(x, tau)
    A = np.empty_like(pois)
    acc = 0.0
    for i in range(pois.size):
        acc = pois[i] + r * acc
        A[i] = acc
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
        expansion, same assumption as in the fitted pdf);
      * tau -> MRCA: standard coalescent with ancestral size theta_anc
        (coalescence rate k(k-1)/(2*theta_anc) per unit of scaled time);
      * infinite-sites mutations at rate 1/2 per lineage per unit time
        (=> E[pairwise diffs] = tau + theta_anc, consistent with the pdf).
    Returns an (n x n) int32 matrix of pairwise differences.
    """
    theta_anc = max(float(theta_anc), 1e-8)
    tau = max(float(tau), 0.0)

    lineages = []          # [tips_mask, birth_time]
    for i in range(n):
        m = np.zeros(n, dtype=bool)
        m[i] = True
        lineages.append([m, 0.0])
    branches = []          # (length, tips_mask)
    s = tau                # all n lineages reach time tau unchanged
    k = n
    while k > 1:
        rate = k * (k - 1) / (2.0 * theta_anc)
        s += rng.exponential(1.0 / rate)
        i, j = (int(v) for v in rng.choice(k, size=2, replace=False))
        for t in (i, j):                       # close branches of merged pair
            m, b = lineages[t]
            branches.append((s - b, m))
        m_new = lineages[i][0] | lineages[j][0]
        hi, lo = (i, j) if i > j else (j, i)
        lineages.pop(hi)
        lineages.pop(lo)
        lineages.append([m_new, s])
        k -= 1
    # the remaining lineage is the MRCA; mutations on it affect all sequences
    # equally and contribute nothing to pairwise differences -> ignored

    lengths = np.array([b[0] for b in branches], dtype=float)
    T = lengths.sum()
    if T <= 0:
        return np.zeros((n, n), dtype=np.int32)
    M = int(rng.poisson(T / 2.0))              # total number of mutations
    if M == 0:
        return np.zeros((n, n), dtype=np.int32)

    chosen = rng.choice(len(branches), size=M, p=lengths / T)
    bits = np.zeros((n, M), dtype=np.float32)  # mutation profiles per sequence
    for mi, br in enumerate(chosen):
        bits[branches[br][1], mi] = 1.0
    pop = bits.sum(axis=1)
    shared = bits @ bits.T                     # mutations shared by both members
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
def amova_phi_st(dist_mat, group_indices):
    """
    One-level distance-based AMOVA.
    Returns (Phi_ST, sigma2_among, sigma2_within).
    Note: sigma2_among (and hence Phi_ST) may be slightly negative if the
    groups are LESS differentiated than expected under random grouping -
    this is normal, it is not clamped to zero.
    """
    n = dist_mat.shape[0]
    K = len(group_indices)
    sizes = [len(g) for g in group_indices]
    if K < 2 or n - K < 1:
        return float('nan'), float('nan'), float('nan')

    SSD_T = float(np.sum(dist_mat[np.triu_indices(n, k=1)], dtype=np.int64))
    SSD_W = 0.0
    for g in group_indices:
        g = np.asarray(g)
        m = g.size
        if m > 1:
            SSD_W += float(np.sum(dist_mat[np.ix_(g, g)][np.triu_indices(m, k=1)], dtype=np.int64))
    SSD_A = SSD_T - SSD_W

    MS_W = SSD_W / (n - K)                     # df within = n - K
    MS_A = SSD_A / (K - 1)                     # df among  = K - 1
    N = float(n)
    n_c = (N * N - sum(float(sk) ** 2 for sk in sizes)) / (N * (K - 1))

    sigma_w = MS_W
    sigma_a = (MS_A - MS_W) / n_c              # correction for unequal sizes
    sigma_t = sigma_a + sigma_w
    Phi = sigma_a / sigma_t if sigma_t > 0 else float('nan')
    return Phi, sigma_a, sigma_w

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
    Tajima_D_allel = float('nan')  # in case of no polymorphism

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
    D_stat, var_D = float('nan'), float('nan')
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

        # Reference calculation via allel
        Tajima_D_allel = calc_tajima_d_allel(seq_list, L)

    # --- Fu's Fs (Fu 1997; log-odds form, as in Arlequin/DnaSP) ---
    Fs = float('nan')
    theta_locus = k_mean   # theta estimated by pi (mean pairwise differences)
    if n >= 10 and S > 0 and theta_locus > 0:
        # P[k] = P(number of haplotypes == k) in a sample of size n
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
        f = sum(P[unique_haps:n + 1])          # f = P(K >= observed)
        if f <= 0.0:
            Fs = float('-inf')
        elif f >= 1.0:
            Fs = float('inf')
        else:
            Fs = math.log(f / (1.0 - f))       # Fs < 0: excess of haplotypes (expansion)

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
            # Harpending's raggedness = sum of squared deviations of the OBSERVED
            # histogram from the EXPECTED (Rogers) one. Under an OLS fit this is
            # mathematically identical to SSD (in Arlequin they differ only via
            # the GLS weighting), so both columns will carry the same value.
            raggedness = ssd

    # 4.6. AMOVA (1-level)
    Phi_ST, var_among, var_within, var_total = float('nan'), float('nan'), float('nan'), float('nan')
    if label == "ALL SAMPLES" and len(traits_dict) > 0:
        region_indices = defaultdict(list)
        for idx, name in enumerate(names_list):
            if name in traits_dict and traits_dict[name]:
                dom_region = max(traits_dict[name], key=traits_dict[name].get)
                region_indices[dom_region].append(idx)
            else:
                region_indices['Unknown'].append(idx)

        valid_regions = {k2: v for k2, v in region_indices.items() if len(v) >= 2}
        if len(valid_regions) >= 2:
            Phi_ST, var_among, var_within = amova_phi_st(dist_mat, list(valid_regions.values()))
            var_total = var_among + var_within

    return {
        'label': label, 'N': n, 'nhap': unique_haps, 'S': S, 'S_single': S_singleton,
        'S_inf': S_inf, 'h': h, 'pi': pi, 'k': k_mean, 'theta_w': theta_w,
        'theta_pi': theta_pi, 'Tajima_D': D_stat, 'Tajima_D_allel': Tajima_D_allel,
        'Fu_Fs': Fs, 'tau': tau_hat, 'theta0': theta_hat,
        'raggedness': raggedness, 'SSD': ssd, 'Phi_ST': Phi_ST,
        'var_among_pct': (var_among / var_total * 100) if not math.isnan(var_total) and var_total > 0 else float('nan'),
        'var_within_pct': (var_within / var_total * 100) if not math.isnan(var_total) and var_total > 0 else float('nan'),
        'p_Phi_ST': float('nan'), 'p_raggedness': float('nan'), 'p_SSD': float('nan')
    }

# ==============================================================================
# 5. PERMUTATION / BOOTSTRAP TESTS
# ==============================================================================
def init_worker(dist_mat):
    global WORKER_DIST_MAT
    WORKER_DIST_MAT = dist_mat

def worker_amova_perm(args):
    sizes, n, n_c, seed = args
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    K = len(sizes)
    pos, SSW = 0, 0.0
    for size in sizes:
        chunk = perm[pos:pos + size]
        pos += size
        m = chunk.size
        if m > 1:
            SSW += float(np.sum(WORKER_DIST_MAT[np.ix_(chunk, chunk)][np.triu_indices(m, k=1)], dtype=np.int64))
    SSA = float(np.sum(WORKER_DIST_MAT[np.triu_indices(n, k=1)], dtype=np.int64)) - SSW
    MS_W = SSW / (n - K)
    MS_A = SSA / (K - 1)
    sigma_a = (MS_A - MS_W) / n_c
    sigma_t = sigma_a + MS_W
    return sigma_a / sigma_t if sigma_t > 0 else float('nan')

def run_permutations_fast(stats_all, sequences, seq_names_list, seq_traits, L, N_PERM):
    total_cores = os.cpu_count() or 4
    n_cores = max(1, total_cores - 1)
    print(f"Starting parallel calculations ({N_PERM} replicates on {n_cores} of {total_cores} available cores)...")
    n = len(sequences)
    dist_mat_for_workers = calc_distance_matrix(sequences, L)
    p_AMOVA, p_Ragged, p_SSD = float('nan'), float('nan'), float('nan')

    # --- AMOVA: permutation of individuals among regions ---
    if stats_all and not math.isnan(stats_all.get('Phi_ST', float('nan'))):
        region_indices = defaultdict(list)
        for idx, name in enumerate(seq_names_list):
            if name in seq_traits and seq_traits[name]:
                region_indices[max(seq_traits[name], key=seq_traits[name].get)].append(idx)
            else:
                region_indices['Unknown'].append(idx)
        valid_regions = {k2: v for k2, v in region_indices.items() if len(v) >= 2}

        if len(valid_regions) >= 2:
            group_sizes = [len(v) for v in valid_regions.values()]
            K = len(group_sizes)
            n_c = (n * n - sum(sk * sk for sk in group_sizes)) / (n * (K - 1))
            args_list = [(group_sizes, n, n_c, 100_000 + i) for i in range(N_PERM)]
            print("  AMOVA permutations... ", end="", flush=True)
            with ProcessPoolExecutor(max_workers=n_cores, initializer=init_worker,
                                     initargs=(dist_mat_for_workers,)) as executor:
                results = list(executor.map(worker_amova_perm, args_list))
            results = [r for r in results if not math.isnan(r)]
            count_extreme = sum(1 for r in results if r >= stats_all['Phi_ST'])
            p_AMOVA = (count_extreme + 1) / (len(results) + 1)
            print(f"done! p(Phi_ST) = {p_AMOVA:.4f}")

    # --- Mismatch: PARAMETRIC bootstrap under the fitted expansion model ---
    if stats_all and n >= 10 and not math.isnan(stats_all.get('tau', float('nan'))):
        args_list = [(n, stats_all['tau'], stats_all['theta0'], 200_000 + i) for i in range(N_PERM)]
        print("  Parametric bootstrap (mismatch SSD/raggedness)... ", end="", flush=True)
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(worker_mismatch_parametric, args_list))
        results = [r for r in results if not math.isnan(r)]
        count_extreme = sum(1 for r in results if r >= stats_all['SSD'])
        p_SSD = (count_extreme + 1) / (len(results) + 1)
        p_Ragged = p_SSD   # raggedness == SSD under the OLS fit -> same null distribution
        print(f"done! p(SSD) = p(raggedness) = {p_SSD:.4f}")

    return p_AMOVA, p_Ragged, p_SSD

# ==============================================================================
# 6. MAIN EXECUTION BLOCK (REQUIRED FOR WINDOWS!)
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("FULL POPULATION GENETIC ANALYSIS (p-distances + Rogers SSD) - corrected")
    print("=" * 80)
    sequences, seq_names_list, seq_traits, L = parse_nexus()
    results = []

    print("Calculating observed values for the ENTIRE SAMPLE...")
    stats_all = calc_advanced_stats(sequences, seq_names_list, seq_traits, L, "ALL SAMPLES")
    if stats_all: results.append(stats_all)

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
            stats = calc_advanced_stats(region_seqs[region], region_names[region], {}, L, f"REGION: {region}")
            if stats: results.append(stats)

    p_AMOVA, p_Ragged, p_SSD = run_permutations_fast(stats_all, sequences, seq_names_list, seq_traits, L, N_PERMUTATIONS)

    if stats_all:
        stats_all['p_Phi_ST'] = p_AMOVA
        stats_all['p_raggedness'] = p_Ragged
        stats_all['p_SSD'] = p_SSD

    def print_stats(s):
        print(f"\n{s['label']} (N={s['N']}, hap={s['nhap']})")
        print(f"  {'-' * 70}")
        print(f"  Polymorphic sites (S):         {s['S']} (singletons: {s['S_single']}, pars-inf: {s['S_inf']})")
        print(f"  Haplotype diversity (h):       {s['h']:.4f}")
        print(f"  Nucleotide diversity (pi):     {s['pi'] * 1000:.4f} x 10^-3 (p-distance)")
        print(f"  Tajima's D (our script):       {s['Tajima_D']:.4f}")
        if not math.isnan(s.get('Tajima_D_allel', float('nan'))):
            print(f"  Tajima's D (scikit-allel):     {s['Tajima_D_allel']:.4f} *")
        print(f"  Fu's Fs:                       {s['Fu_Fs']:.4f}   (< 0 = excess of haplotypes / expansion)")
        if not math.isnan(s.get('tau', float('nan'))):
            print(f"  Mismatch fit:                  tau={s['tau']:.4f}, theta0={s['theta0']:.4f}")
        print(f"  Mismatch Raggedness (r):       {s['raggedness']:.4f}")
        print(f"  Mismatch SSD:                  {s['SSD']:.4f}")

        if s['label'] == "ALL SAMPLES" and not math.isnan(s.get('Phi_ST', float('nan'))):
            print(f"  +-- AMOVA (1-level):")
            print(f"  |  Variation Among Regions:   {s['var_among_pct']:.2f}%")
            print(f"  |  Variation Within Regions:  {s['var_within_pct']:.2f}%")
            print(f"  +-- Phi_ST:                    {s['Phi_ST']:.4f}  (p={s.get('p_Phi_ST', float('nan')):.4f})")

        if not math.isnan(s.get('p_SSD', float('nan'))):
            print(f"  +-- p-value (SSD = raggedness): {s['p_SSD']:.4f}  (> 0.05: expansion model not rejected)")

    for res in results:
        print_stats(res)

    fieldnames = ['label', 'N', 'nhap', 'S', 'S_single', 'S_inf', 'h', 'pi', 'k',
                  'theta_w', 'theta_pi', 'Tajima_D', 'Tajima_D_allel', 'Fu_Fs',
                  'tau', 'theta0', 'raggedness', 'SSD', 'p_raggedness', 'p_SSD',
                  'Phi_ST', 'p_Phi_ST', 'var_among_pct', 'var_within_pct']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {k3: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k3, v in res.items()}
            writer.writerow(row)

    print(f"\nResults successfully saved to '{OUTPUT_CSV}'")
    print("=" * 80)

    # Legend
    legend = {
        'label': 'Name of the analyzed group',
        'N': 'Sample size',
        'nhap': 'Number of unique haplotypes',
        'S': 'Number of polymorphic sites',
        'S_single': 'Number of singleton (single-copy) segregating sites',
        'S_inf': 'Parsimony-informative sites',
        'h': 'Haplotype diversity',
        'pi': 'Nucleotide diversity (p-distance)',
        'k': 'Mean absolute number of pairwise differences',
        'theta_w': "Watterson's Theta (per site)",
        'theta_pi': "Tajima's Theta (per site)",
        'Tajima_D': "Tajima's D-statistic (our calculation)",
        'Tajima_D_allel': "Tajima's D-statistic (reference scikit-allel)",
        'Fu_Fs': "Fu's Fs (log-odds form; negative = haplotype excess / expansion signal)",
        'tau': 'Mismatch fit: time since sudden expansion (units of mean pairwise differences)',
        'theta0': 'Mismatch fit: ancestral population theta',
        'raggedness': "Harpending's raggedness (deviation from expected spectrum; = SSD under OLS fit)",
        'SSD': 'Sum of squared deviations from the fitted Rogers expansion model',
        'p_raggedness': 'P-value for raggedness (parametric bootstrap; equals p_SSD under OLS)',
        'p_SSD': 'P-value for SSD (parametric bootstrap under the fitted expansion model)',
        'Phi_ST': 'Fixation index (1-level AMOVA, Excoffier et al. 1992)',
        'p_Phi_ST': 'P-value for Phi_ST (permutation of individuals among regions)',
        'var_among_pct': '% of variance among regions (may be slightly negative if Phi_ST < 0)',
        'var_within_pct': '% of variance within regions'
    }
    legend_file = "bosmina_columns_legend.csv"
    with open(legend_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Column", "Description"])
        for col, desc in legend.items():
            writer.writerow([col, desc])
    print(f"Legend saved to '{legend_file}'")