# Interactive Visualizations

This directory contains all interactive HTML visualizations supporting the integrative species delimitation analysis of *Bosmina* (Cladocera: Bosminidae).
Each visualization is a self-contained HTML file that can be opened directly in any modern web browser. No special software or internet connection is required.

---

##  Available Visualizations

| # | Visualization | Main file | Description |
|---|---------------|-----------|-------------|
| 1 | **Net** | `Bosmina_TCS_MedianJoining.html` and `subgenus` networks | Haplotype networks for the entire genus and each subgenus |
| 2 | **bGMYC4** | `bGMYC_interactive_heatmap.html` + resource directory | bGMYC4 results with phylogenetic tree and heatmap |
| 3 | **Summary** | `Delimitation_heatmap_bgmyc_tree.html` + resource directory | Comparison charts of all delimitation methods |
| 4 | **Agreement** | `Agreement_Matrix.html` + resource directory | Interactive heatmap showing congruence between delimitation methods |

---

##  How to Use

### Option 1: View on GitHub Pages (recommended)

[**Bosmina Integrative Delimitation**](https://dmitrykarabanov.github.io/Bosmina/)

### Option 2: Open locally
1. Download the `.html` file and the additional directory
2. Double-click to open in your web browser
3. Use mouse to **zoom**, **pan**, and **hover** for details

---

## Haplotype Networks

Interactive haplotype networks are available for the entire genus *Bosmina* and for each recognized subgenus. Nodes are sized by haplotype frequency, edges represent mutational steps.

### Whole Genus

| File | Description |
|------|-------------|
| `Bosmina_TCS_MedianJoining.html` | Median-Joining network for the full dataset — overall genetic structure and relationships between subgenera |

### By Subgenus

| File | Subgenus |
|------|----------|
| `Bosmina_TCS.html` | *Bosmina* s.str. |
| `Liederobosmina_TCS.html` | *Liederobosmina* |
| `Sinobosmina_TCS.html` | *Sinobosmina* |
| `Eubosmina_TCS.html` | *Eubosmina* |
| `Lunobosmina_TCS.html` | *Lunobosmina* |

### How to Read the Networks

- **Node size** — proportional to haplotype frequency in the sample
- **Edge length / ticks** — number of mutational steps between haplotypes
- **Colors** — geographic or taxonomic grouping (see legend in each file)
- **Hover** — move cursor over any node to see haplotype details

---

##  Featured Visualization

Our main interactive figure combines two synchronized views:

###  Left Panel: Clade-Colored Phylogeny
- **Branches colored** by major clades 
- **Hover** any branch to see taxon name / clade assignment / branch length

###  Right Panel: Agreement Matrix
- **Heatmap** showing pairwise agreement between delimitation method(s)
- **Color scale** from 0% agreement to 100% agreement
- **Hover** any cell to see taxon names / agreement percentage 

###  Synchronization
Both panels share the same Y-axis, so each row in the matrix corresponds exactly to a tip on the tree.

---

##  Troubleshooting

**Q: The visualization doesn't load.**  
A: Try a different browser or clear your cache. Some corporate networks block large HTML files.

**Q: Hover tooltips don't appear.**  
A: Make sure JavaScript is enabled in your browser settings.

**Q: The file is too large to view on GitHub.**  
A: Download the file locally — GitHub has size limits for inline HTML rendering.

---

##  Related Resources

-  **[Data](../data/)** — Input files used to generate these visualizations
-  **[Scripts](../scripts/)** — Source code for reproducing all figures
-  **[Results](../results/)** — Static tables and statistical outputs


[← Back to Main Page](../README.md)
