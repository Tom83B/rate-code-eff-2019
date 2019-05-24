# rate-code-eff-2019
Supporting code for The effect of inhibition on rate code efficiency indicators

Repositary structure
* `coding`
  * `computation` - files for running the simulation and generating data
    * `transition_scan.py` - generates raw simulation data
    * `transform_list_results.py` - processes raw simulation data and computes the transition rates. Creates the file `transition_rates.pkl`
  * `optimization` - contains algorithms for maximizing mutual information
  * `analysis` - plotting utilities and framework for information channels
* `analysis` - Jupyter notebooks used for creating the plots in the manuscript
