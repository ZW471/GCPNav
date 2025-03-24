# Environment setup
## pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .\BenchMARL
pip install -e .\VectorizedMultiAgentSimulator\
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-geometric
pip install wandb moviepy
## conda
conda env create -f environment.yml

# Reproduce paper results
## GCPNav
python BenchMARL/benchmarl/run.py task=vmas/equivariant_navigation algorithm=mappo experiment=experiment model=gcp_mlp task.exclude_vel_from_obs=True task.lidar_range=0.0,0.2 seed=0,42,100 --multirun
## MLPx4
python BenchMARL/benchmarl/run.py task=vmas/equivariant_navigation algorithm=mappo experiment=experiment model=layers/mlp task.comms_rendering_range=0.0,0.2 model.num_cells=[256,256,256,256] task.lidar_range=0.2 seed=0,42,100 --multirun
## MLP
python BenchMARL/benchmarl/run.py task=vmas/equivariant_navigation algorithm=mappo experiment=experiment model=layers/mlp task.comms_rendering_range=0.0,0.2 task.lidar_range=0.2 seed=0,42,100 --multirun
## GNN
python BenchMARL/benchmarl/run.py task=vmas/equivariant_navigation algorithm=mappo experiment=experiment model=gnn_mlp task.lidar_range=0.0,0.2 seed=0,42,100 --multirun
