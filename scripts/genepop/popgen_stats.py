import re
import math
import csv
import random
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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
random.seed(42)

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
    """Calculates a simple p-distance matrix."""
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
# 4. MAIN STATISTICS CALCULATION
# ==============================================================================
def calc_advanced_stats(seq_list, names_list, traits_dict, L, label="Dataset", dist_mat=None):
    n = len(seq_list)
    if n < 2:
        return None
        
    if dist_mat is None:
        dist_mat = calc_distance_matrix(seq_list, L)
        
    unique_haps = len(set(seq_list))
    Tajima_D_allel = float('nan') # Initialization in case of no polymorphism

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
    total_diff = np.sum(dist_mat[np.triu_indices(n, k=1)])
    k_mean = total_diff / n_pairs if n_pairs > 0 else 0
    pi = k_mean / L if L > 0 else 0

    # 4.3. Haplotype diversity (h)
    hap_counts = defaultdict(int)
    for s in seq_list: hap_counts[s] += 1
    sum_fi2 = sum((c/n)**2 for c in hap_counts.values())
    h = (n / (n - 1)) * (1 - sum_fi2) if n > 1 else 0

    # 4.4. Theta and neutrality
    a1 = sum(1.0/i for i in range(1, n))
    a2 = sum(1.0/(i*i) for i in range(1, n))
    theta_w = (S / a1) / L if a1 > 0 and L > 0 else 0
    theta_pi = pi

    # --- Tajima's D ---
    D, var_D = float('nan'), float('nan')
    if S > 0 and n > 2:
        theta_w_locus = S / a1 
        b1 = (n + 1) / (3.0 * (n - 1))
        b2 = 2.0 * (n*n + n + 3) / (9.0 * n * (n - 1))
        c1 = b1 - 1.0/a1
        c2 = b2 - (n + 2.0)/(a1*n) + a2/(a1* a1)
        e1 = c1 / a1
        e2 = c2 / (a1*a1 + a2)
        var_D = e1*S + e2*S*(S-1)
        if var_D > 0:
            D = (k_mean - theta_w_locus) / math.sqrt(var_D)
            
        # Reference calculation via allel
        Tajima_D_allel = calc_tajima_d_allel(seq_list, L)

    # --- Fu's Fs ---
    Fs = float('nan')
    if n >= 10 and unique_haps >= 1 and S > 0:
        theta_locus = k_mean  
        P = [0.0] * (n + 1)
        P[1] = 1.0  
        for i in range(2, n + 1):
            new_P = [0.0] * (n + 1)
            for k in range(1, i + 1):
                 term1 = P[k-1] * theta_locus / (theta_locus + i - 1) if k > 1 else 0.0
                 term2 = P[k] * (i - 1) / (theta_locus + i - 1) if k <= i - 1 else 0.0
                 new_P[k] = term1 + term2
            P = new_P
            
        f = sum(P[unique_haps:n+1])
        epsilon = 1e-15
        f = max(epsilon, min(1.0 - epsilon, f))
        Fs = math.log(f / (1.0 - f))
        
    # 4.5. Mismatch Distribution
    mismatch_counts = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            mismatch_counts[dist_mat[i, j]] += 1
            
    max_diff = max(mismatch_counts.keys()) if mismatch_counts else 0
    P_obs = np.array([mismatch_counts[i] / n_pairs for i in range(max_diff + 1)])

    raggedness, ssd = 0.0, 0.0
    if len(P_obs) >= 2:
        raggedness = float(np.sum((P_obs[:-1] - P_obs[1:])**2))
        
        def rogers_mismatch_pdf(x_vals, tau, theta_0):
            if tau < 0 or theta_0 < 0: return np.zeros_like(x_vals, dtype=float)
            P = np.zeros_like(x_vals, dtype=float)
            factor = 1.0 / (1.0 + theta_0) if theta_0 > 0 else 1.0
            ratio = theta_0 / (1.0 + theta_0) if theta_0 > 0 else 0.0
            for idx, val in enumerate(x_vals):
                prob_sum = 0.0
                for i in range(val + 1):
                    p_poisson = poisson.pmf(val - i, tau) if tau > 0 else (1.0 if val == i else 0.0)
                    prob_sum += (ratio**i) * p_poisson
                P[idx] = factor * prob_sum
            return P

        def objective(params):
            t, th = params
            P_exp = rogers_mismatch_pdf(np.arange(len(P_obs)), t, th)
            return np.sum((P_obs - P_exp)**2)

        initial_tau = max(k_mean, 1.0)
        initial_theta = 0.1
        try:
            result = minimize(objective, [initial_tau, initial_theta], 
                              method='Nelder-Mead', bounds=[(0, None), (0, None)], options={'maxiter': 500})
            best_tau, best_theta = result.x
            if best_tau < 0: best_tau = 0
            if best_theta < 0: best_theta = 0
        except Exception:
            best_tau, best_theta = initial_tau, initial_theta
            
        P_exp = rogers_mismatch_pdf(np.arange(len(P_obs)), best_tau, best_theta)
        ssd = float(np.sum((P_obs - P_exp)**2))

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
        
        valid_regions = {k: v for k, v in region_indices.items() if len(v) >= 2}
        if len(valid_regions) >= 2:
            SST = total_diff
            SSW = 0
            N_within = 0
            for indices in valid_regions.values():
                k = len(indices)
                N_within += k * (k - 1) // 2
                for i_idx in range(k):
                    for j_idx in range(i_idx + 1, k):
                        SSW += dist_mat[indices[i_idx], indices[j_idx]]
         
            SSA = SST - SSW
            N_tot = n * (n - 1) // 2
            N_among = N_tot - N_within
            
            MSW = SSW / N_within if N_within > 0 else 0
            MSA = SSA / N_among if N_among > 0 else 0
            
            var_within = MSW
            var_among = max(0, MSA - MSW)
            var_total = var_among + var_within
            Phi_ST = var_among / var_total if var_total > 0 else float('nan')

    return {
        'label': label, 'N': n, 'nhap': unique_haps, 'S': S, 'S_inf': S_inf,
        'h': h, 'pi': pi, 'k': k_mean, 'theta_w': theta_w, 'theta_pi': theta_pi,
        'Tajima_D': D, 'Tajima_D_allel': Tajima_D_allel, 'Fu_Fs': Fs,
        'raggedness': raggedness, 'SSD': ssd, 'Phi_ST': Phi_ST, 
        'var_among_pct': (var_among / var_total * 100) if not math.isnan(var_total) and var_total > 0 else float('nan'),
        'var_within_pct': (var_within / var_total * 100) if not math.isnan(var_total) and var_total > 0 else float('nan'),
        'p_Phi_ST': float('nan'), 'p_raggedness': float('nan'), 'p_SSD': float('nan')
    }

# ==============================================================================
# 5. FAST PERMUTATION TESTS
# ==============================================================================
def init_worker(dist_mat):
    global WORKER_DIST_MAT
    WORKER_DIST_MAT = dist_mat

def worker_amova_perm(args):
    group_sizes, n = args
    perm_indices = np.random.permutation(n)
    pos = 0
    p_SSW = 0
    p_N_within = 0
    for size in group_sizes:
        chunk = perm_indices[pos:pos+size]
        pos += size
        p_N_within += size * (size - 1) // 2
        for i in range(len(chunk)):
            for j in range(i + 1, len(chunk)):
                p_SSW += WORKER_DIST_MAT[chunk[i], chunk[j]]
    p_SSA = np.sum(WORKER_DIST_MAT[np.triu_indices(n, k=1)]) - p_SSW
    p_N_among = (n * (n - 1) // 2) - p_N_within
    p_MSW = p_SSW / p_N_within if p_N_within > 0 else 0
    p_MSA = p_SSA / p_N_among if p_N_among > 0 else 0
    p_var_among = max(0, p_MSA - p_MSW)
    p_var_total = p_var_among + p_MSW
    return p_var_among / p_var_total if p_var_total > 0 else 0

def worker_mismatch_bootstrap(args):
    n = args
    boot_indices = np.random.choice(n, size=n, replace=True)
    boot_mismatch = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            d = WORKER_DIST_MAT[boot_indices[i], boot_indices[j]]
            boot_mismatch[d] += 1
    b_max = max(boot_mismatch.keys()) if boot_mismatch else 0
    if b_max == 0: return 0.0, 0.0
    
    b_P = np.array([boot_mismatch[i] / (n * (n - 1) // 2) for i in range(b_max + 1)])
    if len(b_P) >= 2:
        b_ragged = float(np.sum((b_P[:-1] - b_P[1:])**2))
        tau_boot = np.sum(np.arange(len(b_P)) * b_P)
        if tau_boot <= 0: tau_boot = 1e-5
        b_P_exp = poisson.pmf(np.arange(len(b_P)), tau_boot)
        b_ssd = float(np.sum((b_P - b_P_exp)**2))
        return b_ragged, b_ssd
    return 0.0, 0.0

def run_permutations_fast(stats_all, sequences, seq_names_list, seq_traits, L, N_PERM):
    total_cores = os.cpu_count() or 4
    n_cores = max(1, total_cores - 1)
    print(f"Starting parallel calculations ({N_PERM} replicates on {n_cores} of {total_cores} available cores)...")
    n = len(sequences)
    dist_mat_for_workers = calc_distance_matrix(sequences, L)
    p_AMOVA, p_Ragged, p_SSD = float('nan'), float('nan'), float('nan')
    
    if stats_all and not math.isnan(stats_all['Phi_ST']):
        region_indices = defaultdict(list)
        for idx, name in enumerate(seq_names_list):
            if name in seq_traits and seq_traits[name]:
                region_indices[max(seq_traits[name], key=seq_traits[name].get)].append(idx)
            else:
                region_indices['Unknown'].append(idx)
        valid_regions = {k: v for k, v in region_indices.items() if len(v) >= 2}
        
        if len(valid_regions) >= 2:
            group_sizes = [len(indices) for indices in valid_regions.values()]
            args_list = [(group_sizes, n)] * N_PERM
            print("  AMOVA permutations... ", end="", flush=True)
            with ProcessPoolExecutor(max_workers=n_cores, initializer=init_worker, initargs=(dist_mat_for_workers,)) as executor:
                results = list(executor.map(worker_amova_perm, args_list))
            count_extreme = sum(1 for r in results if r >= stats_all['Phi_ST'])
            p_AMOVA = (count_extreme + 1) / (N_PERM + 1)
            print(f"done! p(Phi_ST) = {p_AMOVA:.4f}")

    if n >= 10:
        args_list = [n] * N_PERM
        print("  Mismatch bootstrap... ", end="", flush=True)
        with ProcessPoolExecutor(max_workers=n_cores, initializer=init_worker, initargs=(dist_mat_for_workers,)) as executor:
             results = list(executor.map(worker_mismatch_bootstrap, args_list))
        
        r_results = [r[0] for r in results]
        ssd_results = [r[1] for r in results]
        count_r_extreme = sum(1 for r in r_results if r >= stats_all['raggedness'])
        p_Ragged = (count_r_extreme + 1) / (N_PERM + 1)
        count_ssd_extreme = sum(1 for s in ssd_results if s >= stats_all['SSD'])
        p_SSD = (count_ssd_extreme + 1) / (N_PERM + 1)
        print(f"done! p(r) = {p_Ragged:.4f}, p(SSD) = {p_SSD:.4f}")
        
    return p_AMOVA, p_Ragged, p_SSD

# ==============================================================================
# 6. MAIN EXECUTION BLOCK (REQUIRED FOR WINDOWS!)
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("FULL POPULATION GENETIC ANALYSIS (p-distances + Rogers SSD)")
    print("="*80)
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
        print(f"  {'-'*70}")
        print(f"  Polymorphic sites (S):         {s['S']} (parsimony-inf: {s['S_inf']})")
        print(f"  Haplotype diversity (h):       {s['h']:.4f}")
        print(f"  Nucleotide diversity (pi):     {s['pi']*1000:.4f} x 10^-3 (p-distance)")
        print(f"  Tajima's D (our script):       {s['Tajima_D']:.4f}")
        if not math.isnan(s.get('Tajima_D_allel', float('nan'))):
            print(f"  Tajima's D (scikit-allel):     {s['Tajima_D_allel']:.4f} *")
        print(f"  Fu's Fs:                       {s['Fu_Fs']:.4f}")
        print(f"  Mismatch Raggedness (r):       {s['raggedness']:.4f}")
        print(f"  Mismatch SSD:                  {s['SSD']:.4f}")
        
        if s['label'] == "ALL SAMPLES" and not math.isnan(s.get('Phi_ST', float('nan'))):
            print(f"  +-- AMOVA (1-level):")
            print(f"  |  Variation Among Regions:   {s['var_among_pct']:.2f}%")
            print(f"  |  Variation Within Regions:  {s['var_within_pct']:.2f}%")
            print(f"  +-- Phi_ST:                    {s['Phi_ST']:.4f}  (p={s.get('p_Phi_ST', float('nan')):.4f})")
            
        if not math.isnan(s.get('p_raggedness', float('nan'))):
             print(f"  +-- p-value (Raggedness):      {s['p_raggedness']:.4f}")
        if not math.isnan(s.get('p_SSD', float('nan'))):
             print(f"  +-- p-value (SSD):             {s['p_SSD']:.4f}")

    for res in results:
        print_stats(res)

    fieldnames = ['label', 'N', 'nhap', 'S', 'S_inf', 'h', 'pi', 'k', 'theta_w', 'theta_pi', 
                  'Tajima_D', 'Tajima_D_allel', 'Fu_Fs', 'raggedness', 'p_raggedness', 'SSD', 'p_SSD',
                  'Phi_ST', 'p_Phi_ST', 'var_among_pct', 'var_within_pct']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}
            writer.writerow(row)

    print(f"\nResults successfully saved to '{OUTPUT_CSV}'")
    print("="*80)

    # Legend
    legend = {
        'label': 'Name of the analyzed group',
        'N': 'Sample size',
        'nhap': 'Number of unique haplotypes',
        'S': 'Number of polymorphic sites',
        'S_inf': 'Parsimony-informative sites',
        'h': 'Haplotype diversity',
        'pi': 'Nucleotide diversity (p-distance)',
        'k': 'Mean absolute number of pairwise differences',
        'theta_w': "Watterson's Theta",
        'theta_pi': "Tajima's Theta",
        'Tajima_D': "Tajima's D-statistic (our calculation)",
        'Tajima_D_allel': "Tajima's D-statistic (reference scikit-allel)",
        'Fu_Fs': "Fu's Fs-statistic",
        'raggedness': "Harpending's raggedness index",
        'p_raggedness': 'P-value for raggedness index',
        'SSD': 'Sum of squared deviations',
        'p_SSD': 'P-value for SSD',
        'Phi_ST': 'Fixation index (AMOVA)',
        'p_Phi_ST': 'P-value for Phi_ST',
        'var_among_pct': '% of variance among regions',
        'var_within_pct': '% of variance within regions'
    }
    legend_file = "bosmina_columns_legend.csv"
    with open(legend_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Column", "Description"])
        for col, desc in legend.items():
            writer.writerow([col, desc])
    print(f"Legend saved to '{legend_file}'")