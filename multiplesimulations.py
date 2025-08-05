# Install required libraries
!pip install plotly numpy pandas torch --quiet

# Import libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Gun Database Class
class GunBallisticDatabase:
    def __init__(self):
        self.gun_database = {
            'Glock 17': {
                'caliber': '9mm', 
                'muzzle_velocity': 1150, 
                'barrel_length': 4.49, 
                'typical_grain': 115, 
                'max_range': 50, 
                'velocity_decay_rate': 0.08
            },
            'AR-15': {
                'caliber': '5.56 NATO', 
                'muzzle_velocity': 3100, 
                'barrel_length': 16, 
                'typical_grain': 62, 
                'max_range': 500, 
                'velocity_decay_rate': 0.05
            },
            'AK-47': {
                'caliber': '7.62x39mm', 
                'muzzle_velocity': 2400, 
                'barrel_length': 16.3, 
                'typical_grain': 123, 
                'max_range': 400, 
                'velocity_decay_rate': 0.06
            },
            'Barrett M82': {
                'caliber': '.50 BMG', 
                'muzzle_velocity': 2800, 
                'barrel_length': 29, 
                'typical_grain': 660, 
                'max_range': 1800, 
                'velocity_decay_rate': 0.03
            },
            'Smith & Wesson .357': {
                'caliber': '.357 Magnum', 
                'muzzle_velocity': 1450, 
                'barrel_length': 6, 
                'typical_grain': 158, 
                'max_range': 100, 
                'velocity_decay_rate': 0.09
            },
            'Desert Eagle': {
                'caliber': '.50 AE', 
                'muzzle_velocity': 1400, 
                'barrel_length': 6, 
                'typical_grain': 300, 
                'max_range': 200, 
                'velocity_decay_rate': 0.07
            }
        }
    
    def get_gun_specs(self, gun_name):
        return self.gun_database.get(gun_name, None)
    
    def calculate_velocity_at_distance(self, gun_name, distance):
        specs = self.get_gun_specs(gun_name)
        if not specs:
            return None
        v0 = specs['muzzle_velocity']
        decay_rate = specs['velocity_decay_rate']
        air_resistance = 0.001 * distance
        velocity = v0 * np.exp(-decay_rate * distance / 100) * (1 - air_resistance)
        return max(velocity, v0 * 0.3)

# Crystal Structure Generator Class
class AdvancedCrystalStructureGenerator:
    def __init__(self):
        self.bond_length_c_c = 1.42
        self.layer_spacing = 3.4
    
    def generate_carbon_nanotube_structure(self, radius, n_hexagons, length):
        """Generate a simplified carbon nanotube structure"""
        positions = []
        bonds = []
        
        hex_height = length / n_hexagons
        atoms_per_ring = max(6, int(2 * np.pi * radius / self.bond_length_c_c))
        
        for i in range(n_hexagons):
            z_pos = i * hex_height
            for j in range(atoms_per_ring):
                theta = 2 * np.pi * j / atoms_per_ring
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                z = z_pos
                positions.append([x, y, z])
                
                current_idx = len(positions) - 1
                
                # Create bonds within ring
                if j > 0:
                    bonds.append([current_idx, current_idx - 1])
                elif atoms_per_ring > 1:
                    bonds.append([current_idx, current_idx - atoms_per_ring + 1])
                
                # Create bonds between rings
                if i > 0:
                    prev_ring_atom = current_idx - atoms_per_ring
                    if prev_ring_atom >= 0:
                        bonds.append([current_idx, prev_ring_atom])
        
        return np.array(positions), bonds
    
    def generate_graphene_layer(self, width, height, layer_z=0):
        """Generate a simplified graphene layer"""
        positions = []
        bonds = []
        
        a = self.bond_length_c_c * np.sqrt(3)
        nx = max(3, int(width / a))
        ny = max(3, int(height / (1.5 * self.bond_length_c_c)))
        
        for i in range(nx):
            for j in range(ny):
                x1 = i * a
                y1 = j * 1.5 * self.bond_length_c_c
                positions.append([x1, y1, layer_z])
                
                if j % 2 == 1:
                    x2 = x1 + a/2
                    y2 = y1
                    positions.append([x2, y2, layer_z])
        
        # Generate bonds for graphene structure
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:], i+1):
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if 1.35 <= distance <= 1.50:  # C-C bond length tolerance
                    bonds.append([i, j])
        
        return np.array(positions), bonds
    
    def create_multilayer_structure(self, base_structure, n_layers):
        """Create multi-layered crystal structure"""
        base_positions, base_bonds = base_structure
        all_positions = []
        all_bonds = []
        
        for layer in range(n_layers):
            layer_z_offset = layer * self.layer_spacing
            
            # Copy and offset positions
            layer_positions = base_positions.copy()
            layer_positions[:, 2] += layer_z_offset
            
            start_idx = len(all_positions)
            all_positions.extend(layer_positions)
            
            # Add intra-layer bonds
            for bond in base_bonds:
                all_bonds.append([bond[0] + start_idx, bond[1] + start_idx])
            
            # Add inter-layer bonds (van der Waals interactions)
            if layer > 0:
                prev_layer_start = start_idx - len(base_positions)
                
                # Create limited inter-layer bonds for stability
                for i in range(0, len(base_positions), 5):  # Every 5th atom
                    if i + prev_layer_start < len(all_positions) - len(base_positions):
                        all_bonds.append([i + prev_layer_start, i + start_idx])
        
        return np.array(all_positions), all_bonds
    
    def generate_structure_permutations(self, base_params):
        """Generate multiple structure permutations with different geometries"""
        structures = []
        
        # Permutation 1: CNT-based structure
        cnt_positions, cnt_bonds = self.generate_carbon_nanotube_structure(
            radius=base_params['radius'] * 1.0,
            n_hexagons=base_params['n_hexagons'],
            length=base_params['length']
        )
        
        n_layers_1 = max(3, int(base_params['length'] / 10))
        layered_1 = self.create_multilayer_structure((cnt_positions, cnt_bonds), n_layers_1)
        
        structures.append({
            'type': 'Carbon Nanotube Enhanced',
            'radius': base_params['radius'],
            'n_hexagons': base_params['n_hexagons'],
            'length': base_params['length'],
            'n_layers': n_layers_1,
            'positions': layered_1[0],
            'bonds': layered_1[1],
            'density': 2.2,
            'young_modulus': 1000
        })
        
        # Permutation 2: Modified CNT with larger radius
        cnt_positions_2, cnt_bonds_2 = self.generate_carbon_nanotube_structure(
            radius=base_params['radius'] * 1.3,
            n_hexagons=base_params['n_hexagons'] + 2,
            length=base_params['length'] * 1.1
        )
        
        n_layers_2 = max(4, int(base_params['length'] / 8))
        layered_2 = self.create_multilayer_structure((cnt_positions_2, cnt_bonds_2), n_layers_2)
        
        structures.append({
            'type': 'Wide CNT Enhanced',
            'radius': base_params['radius'] * 1.3,
            'n_hexagons': base_params['n_hexagons'] + 2,
            'length': base_params['length'] * 1.1,
            'n_layers': n_layers_2,
            'positions': layered_2[0],
            'bonds': layered_2[1],
            'density': 2.0,
            'young_modulus': 1200
        })
        
        # Permutation 3: Graphene-based layered structure
        graphene_positions, graphene_bonds = self.generate_graphene_layer(
            width=base_params['length'],
            height=base_params['length']
        )
        
        n_layers_3 = max(5, int(base_params['length'] / 6))
        layered_3 = self.create_multilayer_structure((graphene_positions, graphene_bonds), n_layers_3)
        
        structures.append({
            'type': 'Multilayer Graphene',
            'radius': 0,
            'n_hexagons': len(graphene_positions),
            'length': base_params['length'],
            'n_layers': n_layers_3,
            'positions': layered_3[0],
            'bonds': layered_3[1],
            'density': 2.1,
            'young_modulus': 1100
        })
        
        return structures

# Ballistic Simulator Class
class AdvancedBallisticSimulator:
    def __init__(self):
        self.energy_transfer_coefficients = {
            'Carbon Nanotube Enhanced': 0.93,
            'Wide CNT Enhanced': 0.91,
            'Multilayer Graphene': 0.88
        }
        
        self.max_layer_absorption = {
            'Carbon Nanotube Enhanced': 600,  # J per layer
            'Wide CNT Enhanced': 550,
            'Multilayer Graphene': 500
        }
    
    def calculate_kinetic_energy(self, mass_grains, velocity_fps):
        """Calculate kinetic energy with high precision"""
        mass_kg = mass_grains * 0.0000648  # Convert grains to kg
        velocity_ms = velocity_fps * 0.3048  # Convert fps to m/s
        
        kinetic_energy = 0.5 * mass_kg * velocity_ms**2
        return kinetic_energy
    
    def simulate_layer_by_layer_interaction(self, bullet_energy, structure_info):
        """Simulate detailed layer-by-layer energy transfer"""
        structure_type = structure_info['type']
        n_layers = structure_info['n_layers']
        
        transfer_coeff = self.energy_transfer_coefficients[structure_type]
        max_absorption = self.max_layer_absorption[structure_type]
        
        layer_absorbed = []
        remaining_energy = bullet_energy
        penetration_layer = None
        
        for layer_idx in range(n_layers):
            if remaining_energy <= 0:
                layer_absorbed.append(0.0)
                continue
            
            # Layer-specific absorption efficiency (decreases with depth)
            layer_efficiency = transfer_coeff * (0.95 ** layer_idx)
            
            # Material deformation energy
            deformation_energy = min(
                remaining_energy * layer_efficiency,
                max_absorption * (1 + 0.1 * layer_idx)  # Increasing absorption with depth
            )
            
            # Fracture and plastic deformation
            if remaining_energy > deformation_energy * 2:
                fracture_energy = deformation_energy * 0.3
                plastic_energy = deformation_energy * 0.7
            else:
                fracture_energy = remaining_energy * 0.2
                plastic_energy = remaining_energy - fracture_energy
            
            total_layer_absorption = min(
                deformation_energy + fracture_energy + plastic_energy,
                remaining_energy
            )
            
            layer_absorbed.append(total_layer_absorption)
            remaining_energy -= total_layer_absorption
            
            # Check if bullet stops at this layer
            if remaining_energy < bullet_energy * 0.05:
                penetration_layer = layer_idx + 1
                remaining_energy = 0
                break
        
        # Pad with zeros if needed
        while len(layer_absorbed) < n_layers:
            layer_absorbed.append(0.0)
        
        return layer_absorbed, remaining_energy, penetration_layer
    
    def predict_penetration_outcome(self, residual_energy, initial_energy):
        """Predict penetration outcome based on residual energy"""
        penetration_ratio = residual_energy / initial_energy
        
        if penetration_ratio < 0.05:
            return "Complete Stop", "No penetration, full energy absorbed"
        elif penetration_ratio < 0.15:
            return "Near Stop", "Minimal residual energy, stopped at back layers"
        elif penetration_ratio < 0.35:
            return "Partial Penetration", "Significant energy remaining, partial penetration"
        else:
            return "Full Penetration", "High residual energy, complete penetration likely"

# Enhanced Visualization Class
class EnhancedVisualization:
    def create_detailed_3d_structure_visualization(self, structure_data, simulation_result):
        """Create detailed 3D visualization with penetration analysis"""
        positions = structure_data['positions']
        bonds = structure_data['bonds']
        structure_type = structure_data['type']
        n_layers = structure_data['n_layers']
        
        # Calculate penetration depth and layer information
        layer_absorbed = simulation_result['force_dissipated_per_layer']
        penetration_layer = simulation_result['penetration_layer']
        total_thickness = n_layers * 3.4  # mm (layer spacing)
        
        fig = go.Figure()
        
        # Color atoms by layer and energy absorption
        z_coords = positions[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # Create color coding based on energy absorption and penetration
        colors = []
        color_labels = []
        
        for i, pos in enumerate(positions):
            layer_num = int(pos[2] / 3.4) if z_max > z_min else 0
            layer_num = min(layer_num, len(layer_absorbed) - 1)
            
            if penetration_layer and layer_num < penetration_layer:
                # Layers that were penetrated - Red gradient
                intensity = min(layer_absorbed[layer_num] / 400, 1.0)
                colors.append(f'rgba(255, {int(255*(1-intensity))}, {int(255*(1-intensity))}, 0.8)')
                color_labels.append(f'Penetrated Layer {layer_num+1}')
            elif layer_num < len(layer_absorbed) and layer_absorbed[layer_num] > 0:
                # Layers that absorbed energy - Blue gradient
                intensity = min(layer_absorbed[layer_num] / 400, 1.0)
                colors.append(f'rgba({int(255*(1-intensity))}, {int(255*(1-intensity))}, 255, 0.8)')
                color_labels.append(f'Energy Absorbing Layer {layer_num+1}')
            else:
                # Intact layers - Green
                colors.append('rgba(0, 200, 0, 0.6)')
                color_labels.append(f'Intact Layer {layer_num+1}')
        
        # Add atoms with detailed coloring
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                line=dict(width=0.5, color='black')
            ),
            name='Carbon Atoms',
            hovertemplate='<b>Carbon Atom</b><br>' +
                         'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f}) Å<br>' +
                         'Layer: %{text}<br>' +
                         '<extra></extra>',
            text=color_labels
        ))
        
        # Add bonds with different colors based on layer state
        bond_colors = []
        bond_x, bond_y, bond_z = [], [], []
        
        for bond in bonds:
            if bond[0] < len(positions) and bond[1] < len(positions):
                pos1, pos2 = positions[bond[0]], positions[bond[1]]
                layer1 = int(pos1[2] / 3.4)
                layer2 = int(pos2[2] / 3.4)
                
                # Determine bond color based on layers
                if penetration_layer and max(layer1, layer2) < penetration_layer:
                    bond_color = 'red'  # Damaged bonds
                elif max(layer1, layer2) < len(layer_absorbed) and (layer_absorbed[layer1] > 0 or layer_absorbed[layer2] > 0):
                    bond_color = 'blue'  # Stressed bonds
                else:
                    bond_color = 'gray'  # Normal bonds
                
                bond_x.extend([pos1[0], pos2[0], None])
                bond_y.extend([pos1[1], pos2[1], None])
                bond_z.extend([pos1[2], pos2[2], None])
                bond_colors.extend([bond_color, bond_color, bond_color])
        
        fig.add_trace(go.Scatter3d(
            x=bond_x,
            y=bond_y,
            z=bond_z,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.4)', width=2),
            name='Carbon Bonds',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add penetration indicator if bullet stopped
        if penetration_layer:
            stop_z = penetration_layer * 3.4
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[stop_z],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name=f'Bullet Stopped at Layer {penetration_layer}',
                hovertemplate=f'<b>Bullet Stopped</b><br>Layer: {penetration_layer}<br>Depth: {stop_z:.1f} Å<extra></extra>'
            ))
        
        # Calculate and display final sheet thickness
        effective_thickness = total_thickness if not penetration_layer else penetration_layer * 3.4
        
        fig.update_layout(
            title=f"{structure_type} - Detailed Ballistic Analysis<br>" +
                  f"Total Layers: {n_layers} | Total Thickness: {total_thickness:.1f} Å ({total_thickness/10:.2f} nm)<br>" +
                  f"Effective Protection: {effective_thickness:.1f} Å | Bullet Status: {'Stopped' if penetration_layer else 'Penetrated'}",
            scene=dict(
                xaxis_title="X (Ångström)",
                yaxis_title="Y (Ångström)",
                zaxis_title="Z (Ångström) - Penetration Depth",
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800,
            font=dict(size=12)
        )
        
        # Add color legend as annotations
        fig.add_annotation(
            text="<b>Color Legend:</b><br>" +
                 "🔴 Red: Penetrated/Damaged Layers<br>" +
                 "🔵 Blue: Energy Absorbing Layers<br>" +
                 "🟢 Green: Intact Layers<br>" +
                 "❌ Red X: Bullet Stop Point",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig

# Main Simulation Function
def run_enhanced_ballistic_simulation(gun_name, bullet_caliber, bullet_grain, distances):
    """Main function to run enhanced ballistic simulation"""
    print(f"🎯 Enhanced Ballistic Simulation for {gun_name}")
    print(f"📊 Caliber: {bullet_caliber} | Grain: {bullet_grain} | Distances: {distances}")
    print("=" * 80)
    
    # Initialize components
    gun_db = GunBallisticDatabase()
    structure_gen = AdvancedCrystalStructureGenerator()
    simulator = AdvancedBallisticSimulator()
    visualizer = EnhancedVisualization()
    
    # Get gun specifications
    gun_specs = gun_db.get_gun_specs(gun_name)
    if not gun_specs:
        print(f"❌ Gun '{gun_name}' not found in database")
        return None
    
    print(f"🔫 Gun Specifications:")
    print(f"   Muzzle Velocity: {gun_specs['muzzle_velocity']} fps")
    print(f"   Barrel Length: {gun_specs['barrel_length']} inches")
    print(f"   Max Range: {gun_specs['max_range']} meters")
    
    # Calculate velocities at different distances
    velocities = []
    for distance in distances:
        velocity = gun_db.calculate_velocity_at_distance(gun_name, distance)
        velocities.append(velocity)
    
    print(f"\n📏 Velocity at distances:")
    for d, v in zip(distances, velocities):
        print(f"   {d}m: {v:.1f} fps")
    
    # Use average velocity for structure optimization
    avg_velocity = np.mean(velocities)
    bullet_energy = simulator.calculate_kinetic_energy(bullet_grain, avg_velocity)
    
    print(f"\n⚡ Bullet Energy Analysis:")
    print(f"   Average Velocity: {avg_velocity:.1f} fps")
    print(f"   Kinetic Energy: {bullet_energy:.2f} J")
    
    # Generate structure parameters based on bullet energy
    base_params = {
        'radius': 5.0 + (bullet_energy / 1000),  # Scale with energy
        'n_hexagons': max(10, int(bullet_energy / 200)),
        'length': max(20, bullet_energy / 100)
    }
    
    # Generate 3 structure permutations
    structures = structure_gen.generate_structure_permutations(base_params)
    
    print(f"\n🔬 Generated {len(structures)} Structure Permutations:")
    
    simulation_results = []
    
    for i, structure in enumerate(structures):
        print(f"\n{'='*60}")
        print(f"Structure {i+1}: {structure['type']}")
        print(f"{'='*60}")
        print(f"   📐 Dimensions:")
        print(f"      Layers: {structure['n_layers']}")
        print(f"      Atoms: {len(structure['positions'])}")
        print(f"      Bonds: {len(structure['bonds'])}")
        print(f"      Total Thickness: {structure['n_layers'] * 3.4:.1f} Å ({structure['n_layers'] * 3.4 / 10:.2f} nm)")
        print(f"   🧪 Material Properties:")
        print(f"      Density: {structure['density']} g/cm³")
        print(f"      Young's Modulus: {structure['young_modulus']} GPa")
        
        # Run ballistic simulation
        layer_absorbed, residual, penetration_layer = simulator.simulate_layer_by_layer_interaction(
            bullet_energy, structure
        )
        
        total_absorbed = sum(layer_absorbed)
        penetration_status, penetration_desc = simulator.predict_penetration_outcome(
            residual, bullet_energy
        )
        
        print(f"   🎯 Ballistic Performance:")
        print(f"      Total Energy Absorbed: {total_absorbed:.2f} J ({total_absorbed/bullet_energy*100:.1f}%)")
        print(f"      Residual Energy: {residual:.2f} J ({residual/bullet_energy*100:.1f}%)")
        print(f"      Penetration Status: {penetration_status}")
        print(f"      Description: {penetration_desc}")
        
        if penetration_layer:
            stopped_thickness = penetration_layer * 3.4
            print(f"      🛑 Bullet Stopped at Layer: {penetration_layer}")
            print(f"      🛑 Effective Thickness Used: {stopped_thickness:.1f} Å ({stopped_thickness/10:.2f} nm)")
            print(f"      🛑 Protection Efficiency: {(1 - penetration_layer/structure['n_layers'])*100:.1f}% layers unused")
        else:
            print(f"      ⚠️  Bullet Penetrated All Layers")
            print(f"      ⚠️  Full Thickness Compromised: {structure['n_layers'] * 3.4:.1f} Å")
        
        print(f"   📊 Layer-by-Layer Energy Absorption:")
        for j, energy in enumerate(layer_absorbed[:10]):  # Show first 10 layers
            if energy > 0:
                print(f"      Layer {j+1}: {energy:.1f} J")
        if len(layer_absorbed) > 10:
            print(f"      ... and {len(layer_absorbed)-10} more layers")
        
        # Store results
        simulation_results.append({
            'structure_id': i + 1,
            'structure_type': structure['type'],
            'n_layers': structure['n_layers'],
            'total_thickness_angstrom': structure['n_layers'] * 3.4,
            'total_thickness_nm': structure['n_layers'] * 3.4 / 10,
            'force_absorbed_total': total_absorbed,
            'force_dissipated_per_layer': layer_absorbed,
            'residual_force': residual,
            'penetration_layer': penetration_layer,
            'effective_thickness': penetration_layer * 3.4 if penetration_layer else structure['n_layers'] * 3.4,
            'penetration_status': penetration_status,
            'penetration_description': penetration_desc,
            'structure_data': structure
        })
    
    print(f"\n🎨 Generating Enhanced 3D Visualizations...")
    
    # Create detailed visualizations for each structure
    for i, result in enumerate(simulation_results):
        print(f"\n🖼️  Displaying Structure {i+1}: {result['structure_type']}")
        fig = visualizer.create_detailed_3d_structure_visualization(
            result['structure_data'], result
        )
        fig.show()
    
    print(f"\n✅ Enhanced Simulation Complete!")
    print(f"   Generated {len(simulation_results)} detailed 3D structure visualizations")
    print(f"   Each visualization shows penetration depth, layer damage, and material thickness")
    
    return {
        'simulation_results': simulation_results,
        'gun_specs': gun_specs,
        'bullet_energy': bullet_energy,
        'velocities_at_distances': list(zip(distances, velocities))
    }

# Run the enhanced simulation
print("🚀 Enhanced Crystal Structure & Ballistic Protection Analysis System")
print("=" * 80)

try:
    gun_name = input("Enter gun name (e.g., AR-15): ") or "AR-15"
    bullet_caliber = input("Enter bullet caliber (e.g., 5.56 NATO): ") or "5.56 NATO"
    bullet_grain = float(input("Enter bullet grain (e.g., 62): ") or 62)
    distances_str = input("Enter distances as comma-separated (e.g., 50,100,200,300,500): ") or "50,100,200,300,500"
    distances = [int(d.strip()) for d in distances_str.split(',')]
except:
    # Default values if input fails
    gun_name = "AR-15"
    bullet_caliber = "5.56 NATO"
    bullet_grain = 62
    distances = [50, 100, 200, 300, 500]
    print("Using default values...")

# Run enhanced simulation
results = run_enhanced_ballistic_simulation(gun_name, bullet_caliber, bullet_grain, distances)

if results:
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    for i, result in enumerate(results['simulation_results']):
        print(f"\n🏗️  Structure {i+1}: {result['structure_type']}")
        print(f"   📏 Total Thickness: {result['total_thickness_angstrom']:.1f} Å ({result['total_thickness_nm']:.2f} nm)")
        print(f"   🎯 Total Energy Absorbed: {result['force_absorbed_total']:.2f} J")
        print(f"   ⚡ Residual Energy: {result['residual_force']:.2f} J")
        print(f"   🏁 Final Outcome: {result['penetration_status']}")
        
        if result['penetration_layer']:
            print(f"   🛑 Bullet Stopped at Layer: {result['penetration_layer']} of {result['n_layers']}")
            print(f"   📐 Effective Protection Thickness: {result['effective_thickness']:.1f} Å")
        else:
            print(f"   ⚠️  Bullet Penetrated All {result['n_layers']} Layers")
    
    print(f"\n🎨 All visualizations include:")
    print(f"   • Color-coded atoms showing damage levels")
    print(f"   • Penetration depth indicators")
    print(f"   • Layer-by-layer thickness measurements")
    print(f"   • Bullet stopping points (if applicable)")
    print(f"   • Material property annotations")
