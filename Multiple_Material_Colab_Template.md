# Multiple Material Simulator - Colab Template

## This is the template structure for the Multiple Material Simulator Colab notebook

### Cell 1: Installation and Setup
```python
# Install required packages for Multiple Material Simulator
!pip install plotly numpy pandas torch scikit-learn matplotlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ All packages installed successfully!")
print("🚀 Ready to run Multiple Material Simulation")
```

### Cell 2: Load Code from GitHub
```python
# Download the Multiple Material Simulator code
!wget https://raw.githubusercontent.com/into-deepth/BallisticMaterial_Simulation/main/multiplesimulations.py

# Import all classes and functions
exec(open('multiplesimulations.py').read())

print("✅ Multiple Material Simulator code loaded successfully!")
```

### Cell 3: Configure Simulation Parameters
```python
# Configuration for the enhanced ballistic simulation
print("🔧 Configuring Simulation Parameters...")

# User can modify these parameters
gun_name = "AR-15"  # Options: "Glock 17", "AR-15", "AK-47", "Barrett M82", "Smith & Wesson .357", "Desert Eagle"
bullet_caliber = "5.56 NATO"
bullet_grain = 62
distances = [50, 100, 200, 300, 500]  # meters

print(f"🎯 Simulation Configuration:")
print(f"   • Gun: {gun_name}")
print(f"   • Caliber: {bullet_caliber}")
print(f"   • Bullet Weight: {bullet_grain} grains")
print(f"   • Test Distances: {distances} meters")
print("\n🔄 You can modify these parameters in the cell above and re-run!")
```

### Cell 4: Run Enhanced Ballistic Simulation
```python
# Run the comprehensive simulation
print("🚀 Starting Enhanced Ballistic Simulation...")
print("⏱️  This may take 1-2 minutes to complete...")

# Execute the main simulation function
results = run_enhanced_ballistic_simulation(
    gun_name=gun_name,
    bullet_caliber=bullet_caliber, 
    bullet_grain=bullet_grain,
    distances=distances
)

if results:
    print("\n✅ Enhanced Simulation Successfully Completed!")
    print("🎨 3D Visualizations have been generated and displayed above")
else:
    print("❌ Simulation failed. Please check the gun name and try again.")
```

### Cell 5: Detailed Results Analysis
```python
if results:
    print("📊 COMPREHENSIVE SIMULATION ANALYSIS")
    print("=" * 80)
    
    # Extract simulation results
    sim_results = results['simulation_results']
    gun_specs = results['gun_specs']
    bullet_energy = results['bullet_energy']
    velocities = results['velocities_at_distances']
    
    # Display gun specifications
    print(f"\n🔫 Weapon Analysis: {gun_name}")
    print(f"   • Muzzle Velocity: {gun_specs['muzzle_velocity']} fps")
    print(f"   • Barrel Length: {gun_specs['barrel_length']} inches")
    print(f"   • Maximum Range: {gun_specs['max_range']} meters")
    print(f"   • Velocity Decay Rate: {gun_specs['velocity_decay_rate']}")
    
    # Display velocity degradation
    print(f"\n📏 Velocity Analysis at Distance:")
    for distance, velocity in velocities:
        velocity_loss = ((gun_specs['muzzle_velocity'] - velocity) / gun_specs['muzzle_velocity']) * 100
        print(f"   • {distance:3d}m: {velocity:6.1f} fps ({velocity_loss:4.1f}% loss)")
    
    # Display energy analysis
    print(f"\n⚡ Projectile Energy Analysis:")
    print(f"   • Bullet Weight: {bullet_grain} grains")
    print(f"   • Average Velocity: {np.mean([v[1] for v in velocities]):.1f} fps")
    print(f"   • Kinetic Energy: {bullet_energy:.2f} Joules")
    
    # Structure comparison
    print(f"\n🏗️  Material Structure Comparison:")
    for i, result in enumerate(sim_results):
        print(f"\n   Structure {i+1}: {result['structure_type']}")
        print(f"   {'─' * 50}")
        print(f"   📐 Layers: {result['n_layers']} | Thickness: {result['total_thickness_nm']:.2f} nm")
        print(f"   🎯 Energy Absorbed: {result['force_absorbed_total']:.1f} J ({result['force_absorbed_total']/bullet_energy*100:.1f}%)")
        print(f"   ⚡ Residual Energy: {result['residual_force']:.1f} J ({result['residual_force']/bullet_energy*100:.1f}%)")
        print(f"   🏁 Outcome: {result['penetration_status']}")
        
        if result['penetration_layer']:
            efficiency = (1 - result['penetration_layer']/result['n_layers']) * 100
            print(f"   🛑 Stopped at Layer: {result['penetration_layer']}/{result['n_layers']} ({efficiency:.1f}% layers unused)")
        else:
            print(f"   ⚠️  Complete Penetration: All {result['n_layers']} layers compromised")
```

### Cell 6: Performance Comparison and Recommendations
```python
if results:
    # Create performance comparison charts
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    structures = [r['structure_type'] for r in sim_results]
    absorbed_energies = [r['force_absorbed_total'] for r in sim_results]
    residual_energies = [r['residual_force'] for r in sim_results]
    thicknesses = [r['total_thickness_nm'] for r in sim_results]
    n_layers = [r['n_layers'] for r in sim_results]
    
    # 1. Energy Absorption Comparison
    x_pos = range(len(structures))
    bars1 = ax1.bar(x_pos, absorbed_energies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_xlabel('Material Structure')
    ax1.set_ylabel('Energy Absorbed (J)')
    ax1.set_title('Energy Absorption Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.replace(' ', '\n') for s in structures], fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, absorbed_energies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{value:.0f}J', ha='center', va='bottom', fontweight='bold')
    
    # 2. Thickness vs Performance
    efficiency = [(abs_e/(abs_e + res_e))*100 for abs_e, res_e in zip(absorbed_energies, residual_energies)]
    scatter = ax2.scatter(thicknesses, efficiency, s=[n*20 for n in n_layers], 
                         c=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    ax2.set_xlabel('Total Thickness (nm)')
    ax2.set_ylabel('Energy Absorption Efficiency (%)')
    ax2.set_title('Thickness vs Efficiency\n(Bubble size = Number of Layers)')
    ax2.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, struct in enumerate(structures):
        ax2.annotate(f'{struct.split()[0]}\n{struct.split()[1]}', 
                    (thicknesses[i], efficiency[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Layer Utilization
    penetration_layers = [r['penetration_layer'] if r['penetration_layer'] else r['n_layers'] for r in sim_results]
    utilization = [(pen_layer/total_layer)*100 for pen_layer, total_layer in zip(penetration_layers, n_layers)]
    
    bars3 = ax3.bar(x_pos, utilization, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax3.set_xlabel('Material Structure')
    ax3.set_ylabel('Layer Utilization (%)')
    ax3.set_title('Material Layer Utilization')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([s.replace(' ', '\n') for s in structures], fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Add value labels
    for bar, value in zip(bars3, utilization):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall Performance Radar
    from math import pi
    categories = ['Energy\nAbsorption', 'Thickness\nEfficiency', 'Layer\nUtilization', 'Overall\nProtection']
    
    # Normalize metrics for radar chart (0-100 scale)
    metrics = []
    for i, result in enumerate(sim_results):
        energy_score = (absorbed_energies[i] / max(absorbed_energies)) * 100
        thickness_score = (1 - (thicknesses[i] / max(thicknesses))) * 100  # Lower thickness is better
        utilization_score = utilization[i]
        protection_score = efficiency[i]
        metrics.append([energy_score, thickness_score, utilization_score, protection_score])
    
    # Convert to radial coordinates
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]  # Complete the circle
    
    ax4.set_theta_offset(pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (struct, metric, color) in enumerate(zip(structures, metrics, colors)):
        values = metric + metric[:1]  # Complete the circle
        ax4.plot(angles, values, 'o-', linewidth=2, label=struct.split()[0] + ' ' + struct.split()[1], color=color)
        ax4.fill(angles, values, alpha=0.25, color=color)
    
    ax4.set_ylim(0, 100)
    ax4.set_title('Overall Performance Comparison\n(Higher is Better)', y=1.08)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Performance recommendations
    print("\n🏆 PERFORMANCE RECOMMENDATIONS")
    print("=" * 60)
    
    best_absorption = max(enumerate(absorbed_energies), key=lambda x: x[1])
    best_efficiency = max(enumerate(efficiency), key=lambda x: x[1])
    most_economic = min(enumerate(thicknesses), key=lambda x: x[1])
    
    print(f"🥇 Best Energy Absorption: {structures[best_absorption[0]]}")
    print(f"   └─ Absorbs {best_absorption[1]:.1f}J ({(best_absorption[1]/bullet_energy)*100:.1f}% of total energy)")
    
    print(f"🥇 Best Efficiency: {structures[best_efficiency[0]]}")
    print(f"   └─ {best_efficiency[1]:.1f}% energy absorption efficiency")
    
    print(f"🥇 Most Economic: {structures[most_economic[0]]}")
    print(f"   └─ Only {most_economic[1]:.2f}nm thickness required")
    
    # Overall recommendation
    overall_scores = []
    for i in range(len(structures)):
        score = (efficiency[i] * 0.4) + ((absorbed_energies[i]/max(absorbed_energies)) * 100 * 0.4) + ((1 - thicknesses[i]/max(thicknesses)) * 100 * 0.2)
        overall_scores.append(score)
    
    best_overall = max(enumerate(overall_scores), key=lambda x: x[1])
    print(f"\n🏆 OVERALL RECOMMENDATION: {structures[best_overall[0]]}")
    print(f"   └─ Composite score: {best_overall[1]:.1f}/100")
    print(f"   └─ Provides optimal balance of protection, efficiency, and material usage")
    
    print(f"\n📋 SIMULATION SUMMARY")
    print(f"   • Projectile: {gun_name} firing {bullet_caliber} ({bullet_grain} grain)")
    print(f"   • Impact Energy: {bullet_energy:.1f} Joules")
    print(f"   • Materials Tested: {len(sim_results)} different structures")
    print(f"   • Best Protection: {structures[best_absorption[0]]}")
    print(f"   • Recommended: {structures[best_overall[0]]}")
```

## Instructions for Colab Usage:

1. **Open the Colab link**: https://colab.research.google.com/drive/1JBvO9RsBUiZseSaZYo7Sw-bPeJTeoxs6?usp=sharing
2. **Run cells sequentially**: Execute each cell in order for complete analysis
3. **Customize parameters**: Modify gun type, ammunition, and distances in Cell 3
4. **Interactive visualizations**: All 3D plots and charts are interactive in Colab
5. **Compare scenarios**: Re-run with different parameters to compare results
6. **Download results**: Save generated plots and analysis data

## Expected Runtime: ~3-4 minutes total

## Available Gun Options:
- "Glock 17" (9mm)
- "AR-15" (5.56 NATO)
- "AK-47" (7.62x39mm)
- "Barrett M82" (.50 BMG)
- "Smith & Wesson .357" (.357 Magnum)
- "Desert Eagle" (.50 AE)
