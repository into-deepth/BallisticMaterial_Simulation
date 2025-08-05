# Single Material Simulator - Colab Template

## This is the template structure for the Single Material Simulator Colab notebook

### Cell 1: Installation and Setup
```python
# Install required packages for Single Material Simulator
!pip install ase torch torchvision torchaudio torch-geometric
!pip install plotly networkx scikit-learn matplotlib pandas numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from ase import Atoms
from ase.visualize import view
from ase.io import write
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("✅ All packages installed successfully!")
print("🚀 Ready to run Single Material Simulation")
```

### Cell 2: Load Code from GitHub
```python
# Download the Single Material Simulator code
!wget https://raw.githubusercontent.com/into-deepth/BallisticMaterial_Simulation/main/Singlematerialsimulator.py

# Import all classes and functions
exec(open('Singlematerialsimulator.py').read())

print("✅ Single Material Simulator code loaded successfully!")
```

### Cell 3: Run Crystal Structure Generation
```python
# Initialize the crystal structure generator
generator = CrystalStructureGenerator()
visualizer = AdvancedVisualization()

print("🔬 Generating Carbon Nanotube structure...")
cnt_positions, cnt_bonds = generator.generate_carbon_nanotube(
    n_hexagons=15, 
    radius=6.0, 
    length=25.0
)

print("🧱 Creating layered structure...")
layered_positions, layered_bonds = generator.create_layered_structure(
    (cnt_positions, cnt_bonds), 
    n_layers=3, 
    layer_spacing=3.4
)

print("🎨 Generating 3D visualization...")
structure_fig = visualizer.visualize_crystal_structure_3d(layered_positions, layered_bonds)
structure_fig.show()

print(f"📊 Structure Statistics:")
print(f"   - Total atoms: {len(layered_positions)}")
print(f"   - Total bonds: {len(layered_bonds)}")
print(f"   - Structure layers: 3")
print(f"   - Material: Carbon Nanotube Enhanced")
```

### Cell 4: Run Ballistic Impact Simulation
```python
print("🎯 Running Ballistic Impact Simulation...")

# Define material properties
material_props = {
    'energy_transfer_efficiency': 0.91,
    'max_layer_absorption': 500,
    'thickness': 5
}

# Initialize simulator
simulator = BallisticImpactSimulator(material_props)

# Define ammunition types to test
bullets = {
    '9mm': {'mass': 115, 'velocity': 1150},
    '5.56 NATO': {'mass': 62, 'velocity': 3100},
    '7.62 NATO': {'mass': 147, 'velocity': 2750},
    '.338 Lapua': {'mass': 250, 'velocity': 3000}
}

print("📋 Ballistic Test Results:")
print("=" * 70)

results = {}
for bullet, specs in bullets.items():
    energy = simulator.calculate_kinetic_energy(specs['mass'], specs['velocity'])
    result = simulator.predict_penetration(energy, material_props)
    results[bullet] = result
    
    print(f"{bullet:12s}: {result['status']:18s} | "
          f"Absorbed: {result['energy_absorbed']:7.1f}J | "
          f"Residual: {result['residual_energy']:7.1f}J | "
          f"Ratio: {result['penetration_ratio']*100:5.1f}%")

print("=" * 70)
print("✅ Single Material Simulation Complete!")
```

### Cell 5: Analysis and Summary
```python
# Create summary visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Energy absorption chart
bullets_list = list(results.keys())
absorbed_energy = [results[b]['energy_absorbed'] for b in bullets_list]
residual_energy = [results[b]['residual_energy'] for b in bullets_list]

x = range(len(bullets_list))
width = 0.35

ax1.bar([i - width/2 for i in x], absorbed_energy, width, label='Absorbed Energy', color='green', alpha=0.7)
ax1.bar([i + width/2 for i in x], residual_energy, width, label='Residual Energy', color='red', alpha=0.7)
ax1.set_xlabel('Ammunition Type')
ax1.set_ylabel('Energy (Joules)')
ax1.set_title('Energy Absorption Analysis')
ax1.set_xticks(x)
ax1.set_xticklabels(bullets_list, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Penetration status pie chart
statuses = [results[b]['status'] for b in bullets_list]
status_counts = {}
for status in statuses:
    status_counts[status] = status_counts.get(status, 0) + 1

ax2.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', startangle=90)
ax2.set_title('Penetration Status Distribution')

plt.tight_layout()
plt.show()

print("\n🎯 Single Material Simulator Analysis Complete!")
print("📊 Key Findings:")
print(f"   • Most effective against: {min(results.keys(), key=lambda x: results[x]['residual_energy'])}")
print(f"   • Least effective against: {max(results.keys(), key=lambda x: results[x]['residual_energy'])}")
print(f"   • Average energy absorption: {np.mean(absorbed_energy):.1f} J")
print(f"   • Material efficiency: {np.mean([results[b]['energy_absorbed']/(results[b]['energy_absorbed']+results[b]['residual_energy']) for b in bullets_list])*100:.1f}%")
```

## Instructions for Colab Usage:

1. **Open the Colab link**: https://colab.research.google.com/drive/10QlwxymVs-UVJRCKvVcbzntaWjFZCEDp?usp=sharing
2. **Run cells sequentially**: Execute each cell in order (Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5)
3. **Interactive visualization**: The 3D structure visualization will be interactive in Colab
4. **Modify parameters**: You can adjust material properties and ammunition types in Cell 4
5. **Save results**: Use Colab's download feature to save generated plots and data

## Expected Runtime: ~2-3 minutes total
