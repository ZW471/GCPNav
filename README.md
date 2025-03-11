pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .\BenchMARL
pip install -e .\VectorizedMultiAgentSimulator\
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-geometric
pip install wandb moviepy