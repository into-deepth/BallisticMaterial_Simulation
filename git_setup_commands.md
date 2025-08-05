# Git Repository Setup Commands

## Initialize and connect to remote repository

# Navigate to your project directory
cd "c:\Users\deepv\OneDrive\Desktop\Tactical Hive"

# Initialize git repository
git init

# Add remote origin
git remote add origin https://github.com/into-deepth/BallisticMaterial_Simulation.git

# Create main branch and switch to it
git checkout -b main

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Added Ballistic Material Simulation with Single and Multiple Material Simulators

- Added Singlematerialsimulator.py with crystal structure generation and GNN
- Added multiplesimulations.py with enhanced multi-material analysis
- Added comprehensive README.md with detailed documentation
- Added requirements.txt with all dependencies
- Integrated Google Colab links for easy access
- Features: Crystal structure generation, ballistic physics simulation, 3D visualization"

# Push to remote repository
git push -u origin main

## Additional commands for future updates

# To add new changes
# git add .
# git commit -m "Update: Description of changes"
# git push origin main

# To pull latest changes
# git pull origin main

# To check status
# git status

# To see commit history
# git log --oneline
