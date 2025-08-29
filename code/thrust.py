import numpy as np
from four_vector import FourVector as FV

def athrust(pp):
    n = len(pp)
    if n <= 2:
        raise ValueError("No thrust for 2 or fewer tracks.")
    if n > 200:
        raise ValueError("No thrust for more than 200 tracks.")
    
    # Convert input array into FourVector objects
    momenta = [FV(0, *p) for p in pp]
    spatial_momenta = np.array([p.spatial_part() for p in momenta])
    
    vmax = 0.0
    thrust_axis = np.zeros(3)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            vc = np.cross(spatial_momenta[i], spatial_momenta[j])
            
            if np.linalg.norm(vc) < 1e-15:
                continue
            
            vl = np.sum([p.spatial_part() if np.dot(p.spatial_part(), vc) >= 0 else -p.spatial_part() for p in momenta], axis=0)
            
            for sign_i in [1, -1]:
                for sign_j in [1, -1]:
                    vnew = vl + sign_i * spatial_momenta[i] + sign_j * spatial_momenta[j]
                    vnorm = np.dot(vnew, vnew)
                    if vnorm > vmax:
                        vmax = vnorm
                        thrust_axis = vnew
    
    # Iterative refinement (4 iterations max)
    for _ in range(4):
        projected_momenta = np.array([p.spatial_part() if np.dot(p.spatial_part(), thrust_axis) >= 0 else -p.spatial_part() for p in momenta])
        vnew = np.sum(projected_momenta, axis=0)
        vnorm = np.dot(vnew, vnew)
        if np.isclose(vnorm, vmax):
            break
        vmax = vnorm
        thrust_axis = vnew
    
    # Normalize
    total_momentum = sum(p.abs_space() for p in momenta)
    thrust = np.sqrt(vmax) / total_momentum if total_momentum > 0 else 0.0
    thrust_axis /= np.linalg.norm(thrust_axis) if np.linalg.norm(thrust_axis) > 1e-15 else 1.0
    
    return thrust, thrust_axis