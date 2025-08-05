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
class CrystalStructureGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_carbon_nanotube(self, n_hexagons=10, radius=5.0, length=20.0):
        positions = []
        bonds = []
        bond_length = 1.42
        
        for i in range(n_hexagons):
            for j in range(6):
                theta = j * np.pi / 3
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                z = i * length / n_hexagons
                positions.append([x, y, z])
                if i > 0:
                    bonds.append([len(positions)-1, len(positions)-7])
                if j > 0:
                    bonds.append([len(positions)-1, len(positions)-2])
                else:
                    bonds.append([len(positions)-1, len(positions)-6])
        return np.array(positions), bonds

    def create_layered_structure(self, base_structure, n_layers=3, layer_spacing=3.4):
        positions, bonds = base_structure
        layered_positions = []
        layered_bonds = []
        for layer in range(n_layers):
            layer_offset = layer * layer_spacing
            layer_positions = positions.copy()
            layer_positions[:, 2] += layer_offset
            start_idx = len(layered_positions)
            layered_positions.extend(layer_positions)
            for bond in bonds:
                layered_bonds.append([bond[0] + start_idx, bond[1] + start_idx])
        return np.array(layered_positions), layered_bonds
class BallisicProtectionGNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=4):
        super(BallisicProtectionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.energy_absorption_head = nn.Linear(hidden_dim, 1)
        self.elastic_modulus_head = nn.Linear(hidden_dim, 1)
        self.ballistic_limit_head = nn.Linear(hidden_dim, 1)
        self.penetration_resistance_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        h3 = F.relu(self.conv3(h2, edge_index))
        graph_repr = global_mean_pool(h3, batch)
        return {
            'energy_absorption': self.energy_absorption_head(graph_repr),
            'elastic_modulus': self.elastic_modulus_head(graph_repr),
            'ballistic_limit': self.ballistic_limit_head(graph_repr),
            'penetration_resistance': self.penetration_resistance_head(graph_repr)
        }
class BallisticImpactSimulator:
    def __init__(self, material_properties):
        self.properties = material_properties

    def calculate_kinetic_energy(self, mass_grains, velocity_fps):
        mass_kg = mass_grains * 0.0000648
        velocity_ms = velocity_fps * 0.3048
        return 0.5 * mass_kg * velocity_ms**2

    def simulate_energy_transfer(self, bullet_energy, material_thickness):
        absorbed_energy = []
        remaining_energy = bullet_energy
        for layer in range(int(material_thickness)):
            layer_absorption = min(
                remaining_energy * self.properties['energy_transfer_efficiency'],
                self.properties['max_layer_absorption']
            )
            absorbed_energy.append(layer_absorption)
            remaining_energy -= layer_absorption
            if remaining_energy <= 0:
                break
        return absorbed_energy, max(0, remaining_energy)

    def predict_penetration(self, bullet_energy, material_properties):
        absorbed, residual = self.simulate_energy_transfer(
            bullet_energy, material_properties['thickness']
        )
        total_absorbed = sum(absorbed)
        penetration_ratio = residual / bullet_energy
        if penetration_ratio < 0.1:
            status = "Stopped"
        elif penetration_ratio < 0.3:
            status = "Partial Penetration"
        else:
            status = "Full Penetration"
        return {
            'status': status,
            'energy_absorbed': total_absorbed,
            'residual_energy': residual,
            'penetration_ratio': penetration_ratio
        }
class AdvancedVisualization:
    def visualize_crystal_structure_3d(self, positions, bonds, atom_types=None):
        if atom_types is None:
            atom_types = ['C'] * len(positions)
        fig = go.Figure()
        for atom_type in set(atom_types):
            mask = np.array(atom_types) == atom_type
            atom_positions = positions[mask]
            fig.add_trace(go.Scatter3d(
                x=atom_positions[:, 0],
                y=atom_positions[:, 1],
                z=atom_positions[:, 2],
                mode='markers',
                marker=dict(size=8, color='gray'),
                name=f'{atom_type} atoms'
            ))
        bond_x, bond_y, bond_z = [], [], []
        for bond in bonds:
            if bond[0] < len(positions) and bond[1] < len(positions):
                pos1, pos2 = positions[bond[0]], positions[bond[1]]
                bond_x.extend([pos1[0], pos2[0], None])
                bond_y.extend([pos1[1], pos2[1], None])
                bond_z.extend([pos1[2], pos2[2], None])
        fig.add_trace(go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z, mode='lines',
            line=dict(color='black', width=2), name='Bonds'
        ))
        fig.update_layout(title="Crystal Structure", width=800, height=600)
        return fig
class AdvancedVisualization:
    def visualize_crystal_structure_3d(self, positions, bonds, atom_types=None):
        if atom_types is None:
            atom_types = ['C'] * len(positions)
        fig = go.Figure()
        for atom_type in set(atom_types):
            mask = np.array(atom_types) == atom_type
            atom_positions = positions[mask]
            fig.add_trace(go.Scatter3d(
                x=atom_positions[:, 0],
                y=atom_positions[:, 1],
                z=atom_positions[:, 2],
                mode='markers',
                marker=dict(size=8, color='gray'),
                name=f'{atom_type} atoms'
            ))
        bond_x, bond_y, bond_z = [], [], []
        for bond in bonds:
            if bond[0] < len(positions) and bond[1] < len(positions):
                pos1, pos2 = positions[bond[0]], positions[bond[1]]
                bond_x.extend([pos1[0], pos2[0], None])
                bond_y.extend([pos1[1], pos2[1], None])
                bond_z.extend([pos1[2], pos2[2], None])
        fig.add_trace(go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z, mode='lines',
            line=dict(color='black', width=2), name='Bonds'
        ))
        fig.update_layout(title="Crystal Structure", width=800, height=600)
        return fig
generator = CrystalStructureGenerator()
visualizer = AdvancedVisualization()

print("Generating CNT structure...")
cnt_positions, cnt_bonds = generator.generate_carbon_nanotube(n_hexagons=15, radius=6.0, length=25.0)
layered_positions, layered_bonds = generator.create_layered_structure((cnt_positions, cnt_bonds), n_layers=3, layer_spacing=3.4)
structure_fig = visualizer.visualize_crystal_structure_3d(layered_positions, layered_bonds)
structure_fig.show()

print("Simulating Ballistic Impact...")
material_props = {
    'energy_transfer_efficiency': 0.91,
    'max_layer_absorption': 500,
    'thickness': 5
}
simulator = BallisticImpactSimulator(material_props)
bullets = {
    '9mm': {'mass': 115, 'velocity': 1150},
    '5.56 NATO': {'mass': 62, 'velocity': 3100},
    '7.62 NATO': {'mass': 147, 'velocity': 2750},
    '.338 Lapua': {'mass': 250, 'velocity': 3000}
}
for bullet, specs in bullets.items():
    energy = simulator.calculate_kinetic_energy(specs['mass'], specs['velocity'])
    result = simulator.predict_penetration(energy, material_props)
    print(f"{bullet}: {result['status']} | Absorbed: {result['energy_absorbed']:.1f}J | Residual: {result['residual_energy']:.1f}J")
