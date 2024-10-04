import numpy as np
from dataclasses import dataclass, field

@dataclass
class BasicPointCloud():
    points : np.array
    colors : np.array
    normals : np.array

