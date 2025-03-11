
import torch
from torch_cluster import radius_graph

# Example data
# Points in 3D space (x, y, z)
points = torch.tensor([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 1.0, 0.0]], device='cuda')  # Data is on GPU

print(torch.mm(points, points.t()))
print(torch.mm(points, points.t()).diag().sqrt())

radius = 1.5  # Radius for the neighborhood

# Create the radius graph on GPU
edge_index = radius_graph(points, r=radius)

# Result: edge_index contains pairs of indices representing edges
print("Edge Index (on GPU):")
print(edge_index)
