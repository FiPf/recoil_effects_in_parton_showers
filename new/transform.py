import numpy as np
from four_vector import FourVector as FV

class Transform:
    @staticmethod
    def householder(v: FV, vp: FV) -> np.ndarray:
        """Constructs the boost matrix that transforms vp into v.

        Args:
            v (FV): First FourVector
            vp (FV): Target FourVector

        Returns:
            np.ndarray: boost matrix
        """
        diff = vp - v
        ndiff = diff.four_dot(diff)  
        
        boostm = np.identity(4) 
        
        if abs(ndiff) > 1e-10:  # Only apply the transformation if the vectors are not identical
            boostm -= (2 / ndiff) * diff.outer_product()
        
        return boostm

    @staticmethod
    def perpendicular_component(vec: FV, ref: FV, vvec: FV, M2: float):
        """Compute the perpendicular component of vec with respect to ref.

        Args:
            vec (FV): The vector whose perpendicular component is being calculated.
            ref (FV): The reference vector defining the direction to project against.
            vvec (FV): A vector used as an intermediate offset or baseline in the projection.
            M2 (float): A scalar parameter (usually squared mass or normalization factor)
                        used to scale the projection component.

        Returns:
            FV: The component of `vec` perpendicular to the specified direction.
        """
        return vec - vvec - (1 - M2 / 2) * (ref - vvec)
    
    @staticmethod
    def apply_boost_and_rotation(n1: FV, n2: FV, newvec: FV):
        """Performs the boost and rotation transformations.

        Args:
            n1 (FV): First vector from the parent dipole 
            n2 (FV): Second vector from the parent dipole 
            newvec (FV): emission FourVector

        Returns:
            FV: boosted and rotated FourVector
        """
        M2 = 2 * n1.four_dot(n2)
        M = np.sqrt(M2)
        beta = np.sqrt(1 - M2 / 4)

        n1p = FV(1, beta, 0, M / 2)
        n2p = FV(1, beta, 0, -M / 2)
        vvec = FV(1, 0, 0, 0)
        
        invboost = Transform.householder(n1, n1p)
        n2pp = n2p.matmul(invboost)

        n2pPerp = Transform.perpendicular_component(n2pp, n1, vvec, M2)
        n2Perp = Transform.perpendicular_component(n2, n1, vvec, M2)
        rotation = Transform.householder(n2Perp, n2pPerp)

        return newvec.matmul(rotation).matmul(invboost)

cutBoost = 1e-6

def boost_to_com_frame_three(p1: FV, p2: FV, p3: FV):
    """Boost the system of three four-vectors to their common COM frame.

    Args:
        p1 (FV): 1/3 vectors
        p2 (FV): 2/3 vectors
        p3 (FV): 3/3 vectors

    Returns:
        p1_com (FV): 1/3 vectors in com frame
        p2_com (FV): 2/3 vectors in com frame
        p3_com (FV): 3/3 vectors in com frame
    """
    P_total = p1 + p2 + p3
    E_total = P_total.energy()
    P_vec = P_total.spatial_part()
    P_mag = np.linalg.norm(P_vec)

    if P_mag < cutBoost:
        # Already in COM frame
        return p1, p2, p3

    beta = P_vec / E_total
    beta2 = np.dot(beta, beta)
    gamma = 1. / np.sqrt(1 - beta2)

    # Construct the Lorentz boost matrix
    boost_matrix = np.eye(4)
    boost_matrix[0, 0] = gamma
    boost_matrix[0, 1:] = -gamma * beta
    boost_matrix[1:, 0] = -gamma * beta
    for i in range(3):
        for j in range(3):
            boost_matrix[i+1, j+1] += (gamma - 1) * beta[i] * beta[j] / beta2

    # Apply the boost to each vector
    p1_com = FV(*boost_matrix @ p1.to_array())
    p2_com = FV(*boost_matrix @ p2.to_array())
    p3_com = FV(*boost_matrix @ p3.to_array())

    return p1_com, p2_com, p3_com

def householder(v: FV, vp: FV) -> np.ndarray:
    """Constructs the boost matrix that transforms vp into v.

    Args:
        v (FV): target vector
        vp (FV): start vectors

    Returns:
        np.ndarray: householder boost matrix
    """
    diff = vp - v
    ndiff = diff.four_dot(diff)  
    
    boostm = np.identity(4) 
    
    if abs(ndiff) > 1e-10:  # Only apply the transformation if the vectors are not identical
        boostm -= (2 / ndiff) * diff.outer_product()
    
    return boostm

def householder_from_normal(n: FV) -> np.ndarray:
    """Constructs a Householder matrix that reflects over the hyperplane orthogonal to n.

    Args:
        n (FV): normal vector of given hyperplane

    Returns:
        np.ndarray: householder boost matrix
    """
    ndot = n.four_dot(n)
    matrix = np.identity(4) - (2 / ndot) * n.outer_product()
    return matrix
