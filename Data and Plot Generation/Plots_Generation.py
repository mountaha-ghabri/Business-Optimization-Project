import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
np.random.seed(42)

# ============================================================================
# DATA FROM REPORT (Table 5.2 & Table 5.1)
# ============================================================================

# Exact data from your report
SITE_DATA = pd.DataFrame({
    'Site': ['Boughrara', 'Rjim Maatoug', 'Batten Ghazal', 'Sidi Boubaker', 'Bouaguereb', 'Gabès'],
    'Rank': [1, 2, 3, 4, 5, 6],
    'P_i': [280, 280, 580, 1100, 2600, 8500],  # Population (persons)
    'TC_i': [1240, 1309, 1009, 1024, 919, 733],  # Total cost (M TND)
    'W_i': [2.88, 0.75, 1.60, 0.75, 2.88, 2.88],  # Water (Mm³/yr)
    'E_i': [8.7, 7.6, 7.6, 5.7, 5.7, 2.1],  # Environment (0-10)
    'Z': [28000, 28000, 58900, 110240, 260190, 850000]  # Objective value
})

# System parameters
PARAMS = {
    'B': 750,  # Budget (M TND)
    'W_req': 2.0,  # Water requirement (Mm³/yr)
    'E_min': 6.0,  # Environmental threshold
}

# Weights
WEIGHTS = {'w1': 100, 'w2': 20, 'w3': 10, 'w4': 5}

# Calculate deviations for each site
SITE_DATA['d1_pop'] = SITE_DATA['P_i']  # Population deviation (target = 0)
SITE_DATA['d2_budget'] = np.maximum(0, SITE_DATA['TC_i'] - PARAMS['B'])  # Budget overrun
SITE_DATA['d3_water'] = np.maximum(0, PARAMS['W_req'] - SITE_DATA['W_i'])  # Water shortfall
SITE_DATA['d4_env'] = np.maximum(0, PARAMS['E_min'] - SITE_DATA['E_i'])  # Env shortfall

# Weighted contributions
SITE_DATA['w_pop'] = SITE_DATA['d1_pop'] * WEIGHTS['w1']
SITE_DATA['w_budget'] = SITE_DATA['d2_budget'] * WEIGHTS['w3']
SITE_DATA['w_water'] = SITE_DATA['d3_water'] * WEIGHTS['w2']
SITE_DATA['w_env'] = SITE_DATA['d4_env'] * WEIGHTS['w4']

# Verify objective calculation
SITE_DATA['Z_calc'] = (SITE_DATA['w_pop'] + SITE_DATA['w_budget'] + 
                        SITE_DATA['w_water'] + SITE_DATA['w_env'])

print("=" * 80)
print("DATA VERIFICATION")
print("=" * 80)
print("\nDeviation Analysis:")
print(SITE_DATA[['Site', 'd1_pop', 'd2_budget', 'd3_water', 'd4_env']].to_string(index=False))
print("\nObjective Verification:")
print(SITE_DATA[['Site', 'Z', 'Z_calc']].to_string(index=False))

# ============================================================================
# PLOT 1: STACKED BAR - WEIGHTED DEVIATIONS
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(SITE_DATA))
width = 0.6

# Create stacked bars
bars1 = ax1.bar(x_pos, SITE_DATA['w_pop'], width, label='Population Exposure (w₁=100)', 
                color='#E74C3C', edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x_pos, SITE_DATA['w_water'], width, bottom=SITE_DATA['w_pop'],
                label='Water Shortfall (w₂=20)', color='#3498DB', edgecolor='black', linewidth=0.8)
bars3 = ax1.bar(x_pos, SITE_DATA['w_budget'], width, 
                bottom=SITE_DATA['w_pop'] + SITE_DATA['w_water'],
                label='Budget Overrun (w₃=10)', color='#F39C12', edgecolor='black', linewidth=0.8)
bars4 = ax1.bar(x_pos, SITE_DATA['w_env'], width,
                bottom=SITE_DATA['w_pop'] + SITE_DATA['w_water'] + SITE_DATA['w_budget'],
                label='Environmental Shortfall (w₄=5)', color='#2ECC71', edgecolor='black', linewidth=0.8)

# Annotations
for i, (idx, row) in enumerate(SITE_DATA.iterrows()):
    # Total Z value on top
    total = row['Z']
    y_pos = total + 10000
    ax1.text(i, y_pos, f'{int(total):,}', ha='center', va='bottom', 
             fontweight='bold', fontsize=9)
    
    # Rank indicator
    rank_text = f"Rank {row['Rank']}"
    if row['Rank'] == 1:
        rank_text += " ✓"
    ax1.text(i, -30000, rank_text, ha='center', va='top', fontsize=8, 
             fontweight='bold', color='darkgreen' if row['Rank'] == 1 else 'black')

ax1.set_ylabel('Weighted Penalty (Lower = Better)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Candidate Sites', fontsize=13, fontweight='bold')
ax1.set_title('Goal Programming Objective Decomposition\n(Lexicographic Weights: Safety=100, Water=20, Cost=10, Environment=5)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(SITE_DATA['Site'], rotation=15, ha='right', fontsize=10)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.set_ylim(-50000, max(SITE_DATA['Z']) * 1.15)
ax1.yaxis.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=1)

# Add status quo marker
gabes_idx = SITE_DATA[SITE_DATA['Site'] == 'Gabès'].index[0]
ax1.add_patch(Rectangle((gabes_idx - width/2, -50000), width, 
                        SITE_DATA.loc[gabes_idx, 'Z'] + 50000,
                        fill=False, edgecolor='red', linewidth=3, linestyle='--'))
ax1.text(gabes_idx, max(SITE_DATA['Z']) * 1.05, 'Status Quo\n(Unacceptable)', 
         ha='center', fontsize=9, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

plt.tight_layout()
plt.savefig('fig1_weighted_deviations_stacked.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fig1_weighted_deviations_stacked.png")
plt.close()

# ============================================================================
# PLOT 2: PARETO FRONTIER - COST VS SAFETY
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Color by rank
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(SITE_DATA)))

for idx, row in SITE_DATA.iterrows():
    marker_size = 400 if row['Site'] == 'Boughrara' else (600 if row['Site'] == 'Gabès' else 300)
    marker = '*' if row['Site'] == 'Gabès' else 'o'
    
    ax2.scatter(row['TC_i'], row['P_i'], s=marker_size, c=[colors[idx]], 
               marker=marker, edgecolors='black', linewidths=2, alpha=0.85,
               label=f"{row['Site']} (Rank {row['Rank']})", zorder=10 if row['Site'] == 'Gabès' else 5)
    
    # Annotations
    offset_x = 30 if row['Site'] != 'Gabès' else 50
    offset_y = 200 if row['Site'] != 'Gabès' else 500
    ax2.annotate(row['Site'], 
                xy=(row['TC_i'], row['P_i']), 
                xytext=(row['TC_i'] + offset_x, row['P_i'] + offset_y),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5))

# Budget constraint line
ax2.axvline(x=PARAMS['B'], color='red', linestyle='--', linewidth=2.5, 
           label=f"Budget Constraint: {PARAMS['B']} M TND", alpha=0.7)

# Median population line
median_pop = SITE_DATA['P_i'].median()
ax2.axhline(y=median_pop, color='blue', linestyle=':', linewidth=2, 
           label=f"Median Population: {int(median_pop)} persons", alpha=0.5)

ax2.set_xlabel('Total Lifecycle Cost (Million TND)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Population within 3km (persons)', fontsize=13, fontweight='bold')
ax2.set_title('Pareto Frontier: Safety vs Cost Trade-off\n(Status Quo Dominated by Multiple Alternatives)', 
             fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(650, 1400)
ax2.set_ylim(-500, 9500)

# Add shaded "infeasible" region
ax2.axvspan(PARAMS['B'], 1400, alpha=0.1, color='red', label='Over Budget')

plt.tight_layout()
plt.savefig('fig2_pareto_frontier_cost_safety.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fig2_pareto_frontier_cost_safety.png")
plt.close()

# ============================================================================
# PLOT 3: RADAR CHART - BOUGHRARA VS GABÈS 
# ============================================================================

fig3 = plt.figure(figsize=(14, 12))
ax3 = fig3.add_subplot(111, projection='polar')

categories = ['Safety\n(Low Population)', 'Water\nAvailability', 
              'Cost\nEfficiency', 'Environmental\nQuality']

# Get data for comparison
boughrara = SITE_DATA[SITE_DATA['Site'] == 'Boughrara'].iloc[0]
gabes = SITE_DATA[SITE_DATA['Site'] == 'Gabès'].iloc[0]

# Normalize metrics (0-1 scale, higher = better)
def normalize_metrics(row, data_df):
    # Safety: inverse of population (lower is better)
    safety = 1 - (row['P_i'] / data_df['P_i'].max())
    
    # Water: actual vs requirement (cap at 1.0)
    water = min(1.0, row['W_i'] / PARAMS['W_req'])
    
    # Cost: inverse of cost (lower is better)
    cost = 1 - ((row['TC_i'] - data_df['TC_i'].min()) / 
                (data_df['TC_i'].max() - data_df['TC_i'].min()))
    
    # Environment: actual vs max
    env = row['E_i'] / data_df['E_i'].max()
    
    return [safety, water, cost, env]

boughrara_values = normalize_metrics(boughrara, SITE_DATA)
gabes_values = normalize_metrics(gabes, SITE_DATA)

# Add first value to close the polygon
boughrara_values += boughrara_values[:1]
gabes_values += gabes_values[:1]

# Angles for each axis
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Plot data
ax3.plot(angles, boughrara_values, 'o-', linewidth=3.5, color='#27AE60', 
        label='Boughrara (Optimal)', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax3.fill(angles, boughrara_values, alpha=0.3, color='#27AE60')

ax3.plot(angles, gabes_values, 's-', linewidth=3.5, color='#C0392B', 
        label='Gabès (Status Quo)', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax3.fill(angles, gabes_values, alpha=0.3, color='#C0392B')

# Perfect score reference
perfect = [1.0] * len(angles)
ax3.plot(angles, perfect, ':', linewidth=2, color='gray', alpha=0.5, label='Perfect Score (1.0)')

# Configure axes
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=13, fontweight='bold', y=0.08)
ax3.set_ylim(0, 1.3)  # Extended to create space
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
ax3.set_theta_offset(np.pi / 2)
ax3.set_theta_direction(-1)

# Legend with better positioning
ax3.legend(loc='upper left', bbox_to_anchor=(0.95, 1.12), fontsize=12, framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True)

# Title with more padding
ax3.set_title('Multi-Criteria Performance Comparison\nOptimal Relocation Site vs Current Status Quo\n', 
             fontsize=15, fontweight='bold', pad=40)

# Grid styling
ax3.grid(True, alpha=0.4, linewidth=1.2)
ax3.set_facecolor('#F8F9FA')

# Add value annotations with better positioning
annotation_distance = 1.18
for i, (angle, cat) in enumerate(zip(angles[:-1], categories)):
    b_val = boughrara_values[i]
    g_val = gabes_values[i]
    
    # Position text boxes to avoid overlap
    if i == 0:  # Top
        y_offset = annotation_distance + 0.02
    elif i == 1:  # Right
        y_offset = annotation_distance
    elif i == 2:  # Bottom
        y_offset = annotation_distance - 0.02
    else:  # Left
        y_offset = annotation_distance
    
    ax3.text(angle, y_offset, 
            f'Boughrara: {b_val:.2f}\nGabès: {g_val:.2f}', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', alpha=0.9, linewidth=1.5))

plt.tight_layout(pad=2.5)
plt.savefig('fig3_radar_boughrara_vs_gabes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fig3_radar_boughrara_vs_gabes.png")
plt.close()

# ============================================================================
# PLOT 4: RAW DEVIATIONS COMPARISON (NOT WEIGHTED)
# ============================================================================

fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ('d1_pop', 'Population Exposure (persons)', 'Target: 0', '#E74C3C'),
    ('d2_budget', 'Budget Overrun (M TND)', f'Target: ≤{PARAMS["B"]}', '#F39C12'),
    ('d3_water', 'Water Shortfall (Mm³/yr)', f'Target: ≥{PARAMS["W_req"]}', '#3498DB'),
    ('d4_env', 'Environmental Shortfall (points)', f'Target: ≥{PARAMS["E_min"]}', '#2ECC71')
]

for idx, (metric, title, target, color) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    data = SITE_DATA[metric].values
    bars = ax.bar(range(len(SITE_DATA)), data, color=color, edgecolor='black', 
                  linewidth=1.2, alpha=0.8)
    
    # Highlight best and worst
    best_idx = data.argmin()
    worst_idx = data.argmax()
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)
    bars[worst_idx].set_edgecolor('red')
    bars[worst_idx].set_linewidth(3)
    
    ax.set_xticks(range(len(SITE_DATA)))
    ax.set_xticklabels(SITE_DATA['Site'], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Deviation from Goal', fontsize=11, fontweight='bold')
    ax.set_title(f'{title}\n({target})', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, data)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

fig4.suptitle('Raw Goal Deviations by Site\n(Before Lexicographic Weighting)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('fig4_raw_deviations_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fig4_raw_deviations_comparison.png")
plt.close()

# ============================================================================
# PLOT 5: SUMMARY TABLE VISUALIZATION
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(14, 8))
ax5.axis('tight')
ax5.axis('off')

# Prepare table data
table_data = []
for idx, row in SITE_DATA.iterrows():
    improvement_pop = (gabes['P_i'] - row['P_i']) / gabes['P_i'] * 100
    cost_premium = row['TC_i'] - gabes['TC_i']
    
    table_data.append([
        f"{row['Rank']}",
        row['Site'],
        f"{int(row['P_i']):,}",
        f"{improvement_pop:+.0f}%",
        f"{row['TC_i']:.0f}",
        f"{cost_premium:+.0f}",
        f"{row['W_i']:.2f}",
        '✓' if row['W_i'] >= PARAMS['W_req'] else '✗',
        f"{row['E_i']:.1f}",
        '✓' if row['E_i'] >= PARAMS['E_min'] else '✗',
        f"{int(row['Z']):,}"
    ])

headers = ['Rank', 'Site', 'Pop.\n(3km)', 'vs Gabès\n(%)', 'Total\nCost\n(M TND)', 
           'Cost\nPremium\n(M TND)', 'Water\n(Mm³/yr)', 'Meet\nW?', 'Env.\nScore', 
           'Meet\nE?', 'Objective\nZ']

table = ax5.table(cellText=table_data, colLabels=headers, cellLoc='center',
                 loc='center', bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#34495E')
    cell.set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(table_data) + 1):
    # Rank column
    cell = table[(i, 0)]
    if i == 1:
        cell.set_facecolor('#2ECC71')
        cell.set_text_props(weight='bold', color='white')
    elif i == len(table_data):
        cell.set_facecolor('#E74C3C')
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor('#ECF0F1')
    
    # Color meet/not meet columns
    for j in [7, 9]:  # Meet W? and Meet E?
        cell = table[(i, j)]
        if table_data[i-1][j] == '✓':
            cell.set_facecolor('#D5F4E6')
        else:
            cell.set_facecolor('#FADBD8')

ax5.set_title('Complete Site Ranking Summary\n(Lexicographic Goal Programming Results)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('fig5_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fig5_summary_table.png")
plt.close()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\nOPTIMAL SOLUTION: {SITE_DATA.iloc[0]['Site']}")
print(f"  Objective Value: {int(SITE_DATA.iloc[0]['Z']):,}")
print(f"  Population: {int(SITE_DATA.iloc[0]['P_i']):,} persons")
print(f"  Cost: {SITE_DATA.iloc[0]['TC_i']:.0f} M TND")
print(f"  Water: {SITE_DATA.iloc[0]['W_i']:.2f} Mm³/yr (Meets requirement: {SITE_DATA.iloc[0]['W_i'] >= PARAMS['W_req']})")
print(f"  Environment: {SITE_DATA.iloc[0]['E_i']:.1f}/10 (Meets minimum: {SITE_DATA.iloc[0]['E_i'] >= PARAMS['E_min']})")

gabes_row = SITE_DATA[SITE_DATA['Site'] == 'Gabès'].iloc[0]
optimal_row = SITE_DATA.iloc[0]

print(f"\nIMPROVEMENT VS STATUS QUO (Gabès):")
print(f"  Population reduction: {(gabes_row['P_i'] - optimal_row['P_i']) / gabes_row['P_i'] * 100:.1f}%")
print(f"  Cost increase: +{optimal_row['TC_i'] - gabes_row['TC_i']:.0f} M TND ({(optimal_row['TC_i'] - gabes_row['TC_i']) / gabes_row['TC_i'] * 100:.1f}%)")
print(f"  Environmental improvement: +{optimal_row['E_i'] - gabes_row['E_i']:.1f} points")

print("\n" + "=" * 80)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 80)