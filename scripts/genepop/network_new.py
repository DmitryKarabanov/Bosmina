import re
import math
import base64
from collections import defaultdict
from pyvis.network import Network
from functools import lru_cache
import networkx as nx

# ==========================================
# 1. SETTINGS
# ==========================================
NEXUS_FILE = "bosmina_popart.nex"
OUTPUT_HTML = "Bosmina_TCS_MedianJoining.html"
TCS_CONFIDENCE = 0.90
MJ_EPSILON = 1.0
MAX_MEDIANS = 48
MAX_MJ_ITERATIONS = 1000

IUPAC = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'}, 'U': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'}, 'W': {'A', 'T'},
    'K': {'G', 'T'}, 'M': {'A', 'C'}, 'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'},
    '-': set(), '?': set(), '.': set(), '*': set()
}

print("Reading NEXUS file...")
with open(NEXUS_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# ==========================================
# 2. NEXUS PARSING
# ==========================================
matrix_match = re.search(r'BEGIN DATA;.*?MATRIX\s*(.*?)\s*;\s*END;', content, re.DOTALL | re.IGNORECASE)
seqs = {}
for line in matrix_match.group(1).strip().split('\n'):
    parts = line.strip().split()
    if len(parts) >= 2:
        seqs[parts[0]] = parts[1]

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

print(f"Found {len(seqs)} sequences.")

# ==========================================
# 3. GENETICALLY CORRECT DISTANCE CALCULATION
# ==========================================
def calc_dna_dist(seq1, seq2):
    dist = 0
    min_len = min(len(seq1), len(seq2))
    for i in range(min_len):
        c1, c2 = seq1[i].upper(), seq2[i].upper()
        set1 = IUPAC.get(c1, set())
        set2 = IUPAC.get(c2, set())
        if not set1 or not set2:
            continue
        if set1.isdisjoint(set2):
            dist += 1
    return dist

# ==========================================
# 4. COLLAPSING INTO UNIQUE HAPLOTYPES
# ==========================================
haplotypes = defaultdict(list)
for name, seq in seqs.items():
    haplotypes[seq].append(name)

unique_haps = list(haplotypes.keys())
hap_names = {seq: f"Hap_{i+1:03d}" for i, seq in enumerate(unique_haps)}
print(f"Collapsed to {len(unique_haps)} unique haplotypes.")

# ==========================================
# FAST DISTANCE CALCULATION WITH CACHING
# ==========================================
@lru_cache(maxsize=None)
def calc_dna_dist_cached(seq1, seq2):
    if seq1 > seq2:
        seq1, seq2 = seq2, seq1
    dist = 0
    min_len = min(len(seq1), len(seq2))
    for i in range(min_len):
        c1, c2 = seq1[i].upper(), seq2[i].upper()
        set1 = IUPAC.get(c1, set())
        set2 = IUPAC.get(c2, set())
        if not set1 or not set2:
            continue
        if set1.isdisjoint(set2):
            dist += 1
    return dist

# ==========================================
# 5. TCS THRESHOLD CALCULATION
# ==========================================
print("Calculating probabilistic TCS threshold...")

def poisson_pmf(k, lam):
    try:
        return (lam**k) * math.exp(-lam) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0

def tcs_connection_limit(n_haplotypes, confidence=0.95):
    if n_haplotypes <= 1:
        return 1
    lam = 2.0 * math.log(n_haplotypes)
    cumul = 0.0
    for k in range(1, 1000):
        cumul += poisson_pmf(k, lam)
        if cumul >= (1.0 - confidence):
            return k
    return 1

TCS_LIMIT = tcs_connection_limit(len(unique_haps), TCS_CONFIDENCE)
print(f"Automatic TCS threshold: <= {TCS_LIMIT} mutations.")

# ==========================================
# 6. MEDIAN-JOINING NETWORK
# ==========================================
print("Building Minimum Spanning Network...")

n = len(unique_haps)
dist_matrix = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(i + 1, n):
        d = calc_dna_dist(unique_haps[i], unique_haps[j])
        dist_matrix[i][j] = d
        dist_matrix[j][i] = d

def prim_mst(n_nodes, dist_mat):
    in_mst = [False] * n_nodes
    in_mst[0] = True
    edges = []
    for _ in range(n_nodes - 1):
        min_w = float('inf')
        best_edge = None
        for u in range(n_nodes):
            if not in_mst[u]: continue
            for v in range(n_nodes):
                if in_mst[v]: continue
                if dist_mat[u][v] < min_w:
                    min_w = dist_mat[u][v]
                    best_edge = (u, v, min_w)
        if best_edge:
            u, v, w = best_edge
            edges.append((u, v, w))
            in_mst[v] = True
    return edges

mst_edges = prim_mst(n, dist_matrix)
print(f"MST built: {len(mst_edges)} edges.")

adj = defaultdict(set)
for u, v, w in mst_edges:
    adj[u].add(v)
    adj[v].add(u)

def compute_single_median(s1, s2, s3):
    if not (len(s1) == len(s2) == len(s3)):
        return None
    median_chars = []
    for i in range(len(s1)):
        c1, c2, c3 = s1[i], s2[i], s3[i]
        if c1 == c2 or c1 == c3:
            median_chars.append(c1)
        elif c2 == c3:
            median_chars.append(c2)
        else:
            median_chars.append(c1)
    return ''.join(median_chars)

print("Searching for median vectors...")
all_sequences = set(unique_haps)
all_seq_list = list(unique_haps)
median_flags = [False] * len(unique_haps)
iteration = 0
changed = True
medians_added = 0

def find_triplets(adj_dict):
    triplets = set()
    for u in adj_dict:
        neighbors_u = list(adj_dict[u])
        for i in range(len(neighbors_u)):
            for j in range(i+1, len(neighbors_u)):
                v, w = neighbors_u[i], neighbors_u[j]
                triplets.add(tuple(sorted([u, v, w])))
    return list(triplets)

while changed and iteration < MAX_MJ_ITERATIONS and medians_added < MAX_MEDIANS:
    iteration += 1
    changed = False
    triplets = find_triplets(adj)
    print(f"  Iteration {iteration}: analyzing {len(triplets)} triplets...")

    candidates = []
    for triplet in triplets:
        u, v, w = triplet
        s1, s2, s3 = all_seq_list[u], all_seq_list[v], all_seq_list[w]
        
        median_seq = compute_single_median(s1, s2, s3)
        if median_seq is None or median_seq in all_sequences:
            continue
        
        cost = (calc_dna_dist_cached(median_seq, s1) + 
                calc_dna_dist_cached(median_seq, s2) + 
                calc_dna_dist_cached(median_seq, s3))
        candidates.append((median_seq, cost, triplet))

    if not candidates:
        print(f"  No new medians found. Stopping.")
        break

    candidates.sort(key=lambda x: x[1])
    top_candidates = candidates[:5]

    added_this_iter = 0
    for median_seq, cost, triplet in top_candidates:
        if medians_added >= MAX_MEDIANS:
            break
        if median_seq in all_sequences:
            continue
            
        new_idx = len(all_seq_list)
        all_seq_list.append(median_seq)
        all_sequences.add(median_seq)
        median_flags.append(True)
        
        u, v, w = triplet
        for neighbor in [u, v, w]:
            adj[neighbor].add(new_idx)
            adj[new_idx].add(neighbor)
        
        medians_added += 1
        added_this_iter += 1
        changed = True

    if added_this_iter == 0:
        break

print(f"Added {medians_added} median vectors in {iteration} iterations.")

# ==========================================
# 7. GRAPH CONSTRUCTION (NETWORKX)
# ==========================================
print("Constructing NetworkX graph for reliable edge labels...")
COLORS = {
    'NAmer': '#1f77b4', 'SAmer': '#d62728', 'Eur': '#8c564b',
    'Asia': '#f1c40f', 'Austral': '#2ca02c', 'Unknown': '#cccccc'
}

def make_pie_svg(proportions, size=64):
    total = sum(proportions.values())
    cx, cy, r = size/2, size/2, size/2 - 2
    if total == 0:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">' \
              f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#000000" ' \
              f'stroke-width="2" stroke-dasharray="3,2"/></svg>'
        return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

    paths = []
    start_angle = 0
    for region, count in proportions.items():
        frac = count / total
        end_angle = start_angle + frac * 360
        color = COLORS.get(region, '#cccccc')
         
        if frac >= 0.999:
            paths.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
            break
            
        x1 = cx + r * math.cos(math.radians(start_angle - 90))
        y1 = cy + r * math.sin(math.radians(start_angle - 90))
        x2 = cx + r * math.cos(math.radians(end_angle - 90))
        y2 = cy + r * math.sin(math.radians(end_angle - 90))
        large_arc = 1 if frac > 0.5 else 0
        
        paths.append(f'<path d="M {cx} {cy} L {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2} Z" fill="{color}"/>')
        start_angle = end_angle

    paths.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#222" stroke-width="2"/>')
    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">' + ''.join(paths) + '</svg>'
    return f"data:image/svg+xml;base64,{base64.b64encode(svg_content.encode()).decode()}"

G = nx.Graph()
median_counter = 0

for idx, seq in enumerate(all_seq_list):
    if median_flags[idx] is None:
        continue
    is_median = median_flags[idx] == True

    if is_median:
        median_counter += 1
        node_id = f"mv_{median_counter:02d}"
        node_label = f"mv{median_counter}"
        total_traits = {}
        freq = 0
    else:
        hap_id = f"Hap_{idx+1:03d}"
        node_id = hap_id
        ids = haplotypes[seq]
        freq = len(ids)
        node_label = str(freq)
        
        total_traits = defaultdict(int)
        for name in ids:
            if name in seq_traits:
                for region, count in seq_traits[name].items():
                    total_traits[region] += count

    pie_img = make_pie_svg(total_traits, size=64)
    
    if is_median:
        tooltip = (
            f"MEDIAN VECTOR: {node_id}\n"
            f"(Not detected in the sample,\n"
            f"presumed ancestral haplotype)"
        )
        node_size = 10
        font_size = 12
    else:
        region_lines = [f"* {r}: {c}" for r, c in total_traits.items()]
        region_text = "\n".join(region_lines) if region_lines else "* No data"
        
        seq_lines = [f"* {sid}" for sid in haplotypes[seq][:15]]
        if len(haplotypes[seq]) > 15:
            seq_lines.append(f"...and {len(haplotypes[seq]) - 15} more")
        seq_text = "\n".join(seq_lines)
        
        tooltip = (
            f"Haplotype: {node_id}\n"
            f"Frequency: {freq} sequences\n"
            f"{'-' * 25}\n"
            f"Regions:\n{region_text}\n"
            f"{'-' * 25}\n"
            f"Sequences:\n{seq_text}"
        )

        node_size = 10 + (freq * 2.0)
        font_size = 24

    G.add_node(
        node_id,
        label=" ",
        title=tooltip,
        shape='circularImage',
        image=pie_img,
        size=node_size,
        borderWidth=2 if is_median else 0,
        borderColor='#000000',
        font={'size': font_size, 'color': '#000000', 'strokeWidth': 0, 'face': 'Arial'}
    )

# ==========================================
# 8. EDGES CONSTRUCTION
# ==========================================
print(f"Adding edges (TCS <= {TCS_LIMIT} mutations)...")

MAX_CONNECTIONS_PER_NODE = 4

def get_node_id(idx):
    if median_flags[idx]:
        median_num = sum(1 for k in range(idx + 1) if median_flags[k])
        return f"mv_{median_num:02d}"
    else:
        return f"Hap_{idx+1:03d}"

n_total = len(all_seq_list)

potential_edges = []
for i in range(n_total):
    if median_flags[i] is None: continue
    for j in range(i + 1, n_total):
        if median_flags[j] is None: continue
        
        d = calc_dna_dist_cached(all_seq_list[i], all_seq_list[j])
        if 0 < d <= TCS_LIMIT:
            potential_edges.append((d, i, j))

potential_edges.sort(key=lambda x: x[0])
node_degrees = {i: 0 for i in range(n_total)}
tcs_edges_count = 0

drawn_tcs_edges = []
for d, i, j in potential_edges:
    if node_degrees[i] < MAX_CONNECTIONS_PER_NODE and node_degrees[j] < MAX_CONNECTIONS_PER_NODE:
        id_i = get_node_id(i)
        id_j = get_node_id(j)
        
        edge_width = max(1, 10.0 - d * 1.0)
        G.add_edge(
            id_i, id_j,
            # ИЗМЕНЕНО: Убрали label=str(d) — теперь число мутаций только во всплывающей подсказке
            title=f"TCS: {d} mutations",
            color={'color': '#b0b0b0', 'highlight': '#333333'},
            value=edge_width,
            width=edge_width,
            smooth={'type': 'continuous', 'roundness': 0.1}
        )
        node_degrees[i] += 1
        node_degrees[j] += 1
        drawn_tcs_edges.append((i, j))
        tcs_edges_count += 1

print(f"Created {tcs_edges_count} cleaned TCS connections.")

print("Searching for VISUAL components...")
def find_visual_components(n_nodes, edges):
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
        
    visited = set()
    components = []
    for i in range(n_nodes):
        if median_flags[i] is None or i in visited:
            continue
        
        comp = set()
        queue = [i]
        visited.add(i)
        
        while queue:
            curr = queue.pop(0)
            comp.add(curr)
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components

components = find_visual_components(n_total, drawn_tcs_edges)
print(f"Found {len(components)} VISUAL components.")

print("Building MST between VISUAL clusters...")
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

component_edges = []
K = len(components)
for i in range(K):
    for j in range(i + 1, K):
        min_dist = float('inf')
        best_pair = None
        for idx_a in components[i]:
            for idx_b in components[j]:
                d = calc_dna_dist_cached(all_seq_list[idx_a], all_seq_list[idx_b])
                if d < min_dist:
                    min_dist = d
                    best_pair = (idx_a, idx_b)
        if best_pair:
            component_edges.append((min_dist, i, j, best_pair[0], best_pair[1]))

component_edges.sort(key=lambda x: x[0])
uf = UnionFind(K)
mst_bridges = []
for dist, c_i, c_j, idx_a, idx_b in component_edges:
    if uf.union(c_i, c_j):
        mst_bridges.append((idx_a, idx_b, dist))
        if len(mst_bridges) == K - 1:
            break

bridge_edges_count = 0
for idx_a, idx_b, dist in mst_bridges:
    id_a = get_node_id(idx_a)
    id_b = get_node_id(idx_b)
    G.add_edge(
        id_a, id_b,
        # ИЗМЕНЕНО: Убрали label=str(dist)
        title=f"MST bridge: {dist} mutations\n(minimal connection between clusters)",
        color={'color': '#ff7f0e', 'highlight': '#d62728'},
        value=5.0,
        width=5.0,
        dashes=True,
        smooth={'type': 'curvedCW', 'roundness': 0.4}
    )
    bridge_edges_count += 1

print(f"Added {bridge_edges_count} MST bridges.")

# ==========================================
# 9. PYVIS CONVERSION & PHYSICS
# ==========================================
print("Converting to PyVis and applying physics...")
net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#000000", directed=False)

net.from_nx(G, show_edge_weights=False)

net.barnes_hut(
    gravity=-50000,          
    central_gravity=0.001,    
    spring_length=1200,       
    spring_strength=0.001,    
    damping=0.20             
)

net.set_options("""
{
  "edges": {
    "scaling": {
      "min": 1,
      "max": 10
    },
    "smooth": {
      "type": "continuous",
      "roundness": 0.1
    }
  },
  "interaction": {
   "dragNodes": true,
   "dragView": true,
   "zoomView": true,
   "hover": true
 },
 "physics": {
   "stabilization": {
     "enabled": true,
     "iterations": 1000
   }
 }
}
""")

# ==========================================
# 10. HTML GENERATION
# ==========================================
print("Generating final HTML...")
legend_html = f"""
<div style="position: fixed; top: 20px; right: 20px; background: rgba(255,255,255,0.97); 
border: 1px solid #ccc; padding: 18px 22px; border-radius: 10px; font-family: Arial, sans-serif; 
box-shadow: 2px 4px 12px rgba(0,0,0,0.2); z-index: 1000; font-size: 18px; color: #222; min-width: 260px; line-height: 1.4;">
    
    <b style="font-size: 22px; display: block; margin-bottom: 10px; line-height: 1.0;">Regions:</b>
    
    <div style="line-height: 0.75; margin-bottom: 4px;">
        <span style="color:#1f77b4; font-size: 48px; vertical-align: middle;">&#9679;</span> 
        <span style="vertical-align: middle;">North America</span><br>
        <span style="color:#d62728; font-size: 48px; vertical-align: middle;">&#9679;</span> 
        <span style="vertical-align: middle;">South America</span><br>
        <span style="color:#8c564b; font-size: 48px; vertical-align: middle;">&#9679;</span> 
        <span style="vertical-align: middle;">Europe</span><br>
        <span style="color:#f1c40f; font-size: 48px; vertical-align: middle;">&#9679;</span> 
        <span style="vertical-align: middle;">Asia</span><br>
        <span style="color:#2ca02c; font-size: 48px; vertical-align: middle;">&#9679;</span> 
        <span style="vertical-align: middle;">Australia</span>
    </div>

    <hr style="margin: 12px 0; border: 0; border-top: 2px solid #ddd;">
    
    <div style="font-size: 15px; line-height: 1.5;">
        <b>Algorithm:</b> Median-Joining<br>
        <b>Threshold:</b> TCS {int(TCS_CONFIDENCE*100)}%<br>
        <b>Parsimony limit:</b> {TCS_LIMIT} mut.<br>
        <b>Unique haplotypes:</b> {len(unique_haps)}<br>
        <b>Median vectors:</b> {median_counter}<br>
    </div>

    <hr style="margin: 12px 0; border: 0; border-top: 1px solid #ddd;">
    
    <div style="font-size: 15px; line-height: 1.5;">
        <b>Size</b> = frequency<br>
        <b>Line width</b> = mutations (thicker = fewer)<br>
        <b style="color:#ff7f0e;">┄ Dashed orange</b> = MST bridge<br>
        <b style="color:#666;">&#9711; Dashed circle</b> = median vector<br>
        <i style="font-size:13px; color:#888;">(ancestral haplotype)</i>
    </div>
</div>
"""

html_content = net.generate_html()
html_content = html_content.replace("</body>", legend_html + "\n</body>")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Done! File '{OUTPUT_HTML}' is ready.")
print(f"Summary: {len(unique_haps)} real haplotypes + {median_counter} median vectors.")