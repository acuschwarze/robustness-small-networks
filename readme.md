# Robustness of 'small' networks

This is the code repository accompanying the paper **"Robustness of 'small' networks"** available at [https://arxiv.org/abs/2509.23670](https://arxiv.org/abs/2509.23670).

## Authors

- Jessica Jiang
- Allison C. Zhuang
- Petter Holme
- Peter J. Mucha
- Alice C. Schwarze

## About

This repository contains the code and data used to analyze the robustness of small networks under node removal. We derive expected values for the largest connected component of small G(N,p) random graphs from which nodes are either removed uniformly at random or targeted by highest degree, and compare these values with existing percolation theory.

## Requirements

### Python Dependencies
- NumPy
- SciPy
- Matplotlib
- Pandas
- NetworkX
- Pickle

### C++ Compiler
Some calculations use C++ for performance optimization. The compiled executables are included, but you may need to recompile for your system.

## Reproducing Figures

To recreate the figures from the paper, use the scripts in the `scripts-figures` folder:

1. **To recreate all figures at once:**
   ```bash
   python scripts-figures/run-all.py
   ```

2. **To recreate individual figures:**
   ```bash
   python scripts-figures/figure-[name].py
   ```
   
   Replace `[name]` with the specific figure you want to generate (e.g., `approximation`, `heatmaps`, `intro`, `lcc_over_f_large-attack`, `lcc_over_f_large-random`, `lcc_over_f_small`, `voronoi`).

All generated figures will be saved in the `figures` folder.

## Data Files

### Overview

The repository includes pre-computed data in several folders:

- **`cache-combinatorics`**: Contains cached values for recursive calculations, including probability values and degree distributions
- **`cache-figures`**: Contains aggregated data used for generating heatmap and Voronoi plots
- **`data-real`**: Contains results from node-removal simulations on real networks
- **`data-synthetic`**: Contains S(f) data for G(N,p) networks calculated using three different methods:
  - Recursive calculations (infRelSCurve files)
  - Percolation theory (relSCurve files)  
  - Simulations (simRelSCurve files)

### Complete Repository Structure Explained

```
├─ LICENSE  
├─ cache-combinatorics  
│  # cache of connectivity values for recursive calculations
|  ├─ Pvalues.p  
│  │  # cache of probability values for recursive calculations
│  ├─ exact_degree_distributions_n[4...8]_p0.20.txt  
│  │  # degree distributions for graphs for varying sizes
│  ├─ exact_degree_distributions_n8_p[0.50/0.80/0.90].txt  
│  │  # degree distributions for graphs of varying edge probabilities
│  └─ fvalues.p  
│    # cache of connectivity values for recursive calculations
├─ cache-figures  
│  # data caches for generating heatmap plots and Voronoi plots
│  ├─ heatmap_data.pkl  
│  ├─ real_network_mses_random.csv  
│  └─ real_network_mses_targeted.csv  
├─ cpp  
│  # c++ scripts for fast calculations
│  ├─ .rendered.recursion.cpp  
│  ├─ exact-distributions_v3.cpp  
│  ├─ exact-distributions_v3.exe  
│  ├─ max-degree.cpp  
│  ├─ p-recursion-float128.exe  
│  ├─ p-recursion-multi.cpp  
│  ├─ p-recursion.cpp  
│  └─ p-recursion.exe  
├─ data-real  
│  # cached results from node-removal simulations on real networks
│  └─ fulldata-[network-name].txt  
├─ data-synthetic  
|  # data collection on the size of the largest connected component S  
|  # as a function of the fraction f of removed nodes for G(N,p)
|  # networks for a grid of values of N and p
│  ├─ infRelSCurve_attack[False/True]_n[1...100].npy  
│  │  # 200 data files with S(f) data from recursive calculations
│  ├─ relSCurve_attack[False/True]_n[1...100].npy  
│  │  # 200 data files with S(f) data from percolation theory
│  └─ simRelSCurve_attack[False/True]_n[1...100].npy  
│     # 200 data files with S(f) data from simulations
├─ figures  
|  # Figures created by the figure script
│  ├─ fig_approximation.pdf  
│  ├─ fig_heatmaps.pdf  
│  ├─ fig_intro.pdf  
│  ├─ fig_lcc_over_f_large-attack.pdf  
│  ├─ fig_lcc_over_f_large-random.pdf  
│  ├─ fig_lcc_over_f_small.pdf  
│  └─ fig_voronoi.pdf  
├─ libs  
|  # Folder with function libraries
│  ├─ finiteTheory.py  
│  │  # Library of functions for recursive calculations
│  ├─ infiniteTheory.py  
│  │  # Library of functions for calculations based on percolation theory
│  ├─ performanceMeasures.py  
│  │  # Library of functions for calculating a network`s performance
│  ├─ robustnessSimulations.py  
│  │  # Library of functions to simulate node removal on sampled networks
│  ├─ utils.py  
│  │  # Library of helper functions
│  └─ visualizations.py  
│     # Library of plotting functions used in exploratory analysis
├─ scripts-calculations  
|  # Folder with scripts for data generation and analysis
│  ├─ calculate_fp_dictionaries.py  
│  │  # Script to create cached values for recursive calculations
│  ├─ heatmap_aggregator.py  
│  │  # Script to aggregate S(f) data from calculations and simulations
│  ├─ heatmap_finite.py  
│  │  # Script to calculate S(f) using our recursive calculation
│  ├─ heatmap_infinite.py  
│  │  # Script to calculate S(f) using results from percolation theory
│  ├─ heatmap_sim.py  
│  │  # Script to calculate S(f) via simulations on sampled networks
│  └─ wrapper.ps1  
│     # Wrapper script for parallelizing heatmap data generation on Windows
└─ scripts-figures  
   # Folder with scripts for generating figures
   ├─ figure-*.py  
   │  # Script to recreate a figure from the paper
   └─ run-all.py  
      # Script to recreate all figures from the paper
```

## External Data

The repository is self-contained except for the collection of small real-world networks used in the study. These can be separately obtained from [https://github.com/pholme/small](https://github.com/pholme/small).

## Running Calculations

If you want to regenerate the data from scratch:

1. **Generate cached values for recursive calculations:**
   ```bash
   python scripts-calculations/calculate_fp_dictionaries.py
   ```

2. **Calculate S(f) data using different methods:**
   ```bash
   python scripts-calculations/heatmap_finite.py    # Recursive calculation
   python scripts-calculations/heatmap_infinite.py  # Percolation theory
   python scripts-calculations/heatmap_sim.py       # Simulations
   ```

3. **Aggregate the data:**
   ```bash
   python scripts-calculations/heatmap_aggregator.py
   ```

For parallel processing on Windows, you can use the provided PowerShell wrapper:
```powershell
./scripts-calculations/wrapper.ps1
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jiang2024robustness,
  title={Robustness of 'small' networks},
  author={Jiang, Jessica and Zhuang, Allison C. and Holme, Petter and Mucha, Peter J. and Schwarze, Alice C.},
  journal={arXiv preprint arXiv:2509.23670},
  year={2024}
}
```

## License

See the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors through the information provided in the paper.