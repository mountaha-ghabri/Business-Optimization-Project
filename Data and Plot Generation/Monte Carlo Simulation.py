import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value

# ============================================================================
# 1. BASE DATA & SYSTEM PARAMETERS
# ============================================================================
SITE_DATA_MEDIAN = {
    'Sidi_Boubaker': {'P_i': 1100, 'TC_i': 1024.3, 'W_i': 0.75, 'E_i': 5.7},
    'Boughrara':     {'P_i': 280,  'TC_i': 1239.5, 'W_i': 2.88, 'E_i': 8.7},
    'Gabes':         {'P_i': 8500, 'TC_i': 733.1,  'W_i': 2.88, 'E_i': 2.1},
    'Batten_Ghazal': {'P_i': 580,  'TC_i': 1009.0, 'W_i': 1.60, 'E_i': 7.6},
    'Bouaguereb':    {'P_i': 2600, 'TC_i': 918.5,  'W_i': 2.88, 'E_i': 5.7},
    'Rjim_Maatoug':  {'P_i': 280,  'TC_i': 1309.3, 'W_i': 0.75, 'E_i': 7.6}
}

PARAMS = {'B': 750, 'W_req': 2.0, 'E_min': 6.0}
WEIGHTS = {'w1': 100, 'w2': 20, 'w3': 10, 'w4': 5} # Lexicographical Order
N_ITERATIONS = 1000

# ============================================================================
# 2. MOGP SOLVER ENGINE
# ============================================================================
def solve_mogp(data_sample):
    """
    Standard Lexicographical Goal Programming Solver
    """
    model = LpProblem("Relocation_Optimization", LpMinimize)
    sites = list(data_sample.keys())
    
    # Decision Variables
    x = {s: LpVariable(f"x_{s}", cat='Binary') for s in sites}
    
    # Deviational Variables
    d1_p = LpVariable("d1_p", lowBound=0) # Population Deviation
    d2_p = LpVariable("d2_p", lowBound=0) # Budget Overrun
    d3_n = LpVariable("d3_n", lowBound=0) # Water Shortfall
    d4_n = LpVariable("d4_n", lowBound=0) # Env. Shortfall
    
    # Objective Function
    model += WEIGHTS['w1']*d1_p + WEIGHTS['w2']*d3_n + WEIGHTS['w3']*d2_p + WEIGHTS['w4']*d4_n
    
    # Constraints
    model += lpSum([x[s] for s in sites]) == 1 # Select exactly one site
    model += lpSum([data_sample[s]['P_i'] * x[s] for s in sites]) == d1_p
    model += lpSum([data_sample[s]['TC_i'] * x[s] for s in sites]) - d2_p == PARAMS['B']
    model += lpSum([data_sample[s]['W_i'] * x[s] for s in sites]) + d3_n >= PARAMS['W_req']
    model += lpSum([data_sample[s]['E_i'] * x[s] for s in sites]) + d4_n >= PARAMS['E_min']
    
    # Solve
    model.solve(PULP_CBC_CMD(msg=0))
    
    # Extract Result
    selected = [s for s in sites if value(x[s]) == 1][0]
    return {
        'selected_site': selected,
        'Z': value(model.objective),
        'pop_dev': value(d1_p),
        'cost_over': value(d2_p),
        'water_short': value(d3_n),
        'env_short': value(d4_n)
    }

# ============================================================================
# 3. MONTE CARLO EXECUTION
# ============================================================================
print(f"Starting Monte Carlo Simulation ({N_ITERATIONS} iterations)...")
np.random.seed(42)

mc_results = []

for i in range(N_ITERATIONS):
    # STEP 1 & 2: Generate random sample (±15% variance using Normal Dist)
    sample_data = {}
    for site, vals in SITE_DATA_MEDIAN.items():
        sample_data[site] = {
            'P_i':  np.random.normal(vals['P_i'], vals['P_i'] * 0.10),
            'TC_i': np.random.normal(vals['TC_i'], vals['TC_i'] * 0.05),
            'W_i':  np.random.normal(vals['W_i'], vals['W_i'] * 0.15),
            'E_i':  np.random.normal(vals['E_i'], vals['E_i'] * 0.10)
        }
    
    # STEP 3: Solve optimization for this sample
    try:
        res = solve_mogp(sample_data)
        mc_results.append(res)
    except:
        continue # Skip infeasible samples if any

# STEP 4: Aggregate into DataFrame
df_mc = pd.DataFrame(mc_results)

# ============================================================================
# 4. STATISTICAL SUMMARY
# ============================================================================
print("\n" + "="*40)
print("MONTE CARLO SUMMARY STATISTICS")
print("="*40)

# 1. Selection Probability (Robustness)
prob = (df_mc['selected_site'].value_counts(normalize=True) * 100).round(2)
print("\nSite Selection Probability:")
print(prob)

# 2. Performance Metrics for the Modal (Best) Site
modal_site = prob.idxmax()
site_stats = df_mc[df_mc['selected_site'] == modal_site].describe().loc[['mean', 'std']]

print(f"\nExpected Performance for Modal Site ({modal_site}):")
print(f"- Avg Penalty (Z): {site_stats.at['mean', 'Z']:.2f}")
print(f"- Avg Population Exposure: {site_stats.at['mean', 'pop_dev']:.0f} inhabitants")
print(f"- Avg Budget Overrun: {site_stats.at['mean', 'cost_over']:.2f} M TND")
print(f"- Avg Water Shortfall: {site_stats.at['mean', 'water_short']:.2f} Mm3/yr")

# Export data for your external plotting script
df_mc.to_csv("monte_carlo_results.csv", index=False)
print("\n✓ Results exported to 'monte_carlo_results.csv'")