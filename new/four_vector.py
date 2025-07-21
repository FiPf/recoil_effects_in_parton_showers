#This class is similar to: https://github.com/MarcelBalsiger/ngl_resum/blob/master/ngl_resum/ngl_resum.py 
import numpy as np
import math

Nc = 3
cutMassive = 1e-10
cutInfRap = 1e-8
cutBoost = 1e-10
toleranceSameVector = 1e-10

class FourVector:
    def __init__(self, t: float, x: float, y: float, z: float):
        """Initialize the FourVector with its attributes

        Args:
            t (float): time of coordinate vector or energy component of momentum vector
            x (float): x coordinate of coordinate or momentum vector
            y (float): y coordinate of coordinate or momentum vector
            z (float): z coordinate of coordinate or momentum vector
        """
        self.vec = np.array([t, x, y, z], dtype=float)

    def abs_space(self): 
        """Compute the Euclidian norm of the spatial part (x,y,z).

        Returns:
            float: magnitude of the spatial 3-vector.
        """
        return np.sqrt(self.vec[1]**2 + self.vec[2]**2 + self.vec[3]**2)

    def is_massive(self):
        """check whether a FourVector is massive (non-zero norm)

        Returns:
            bool: True if massive, False else. 
        """
        return not math.isclose(self.four_dot(self), 0.0, abs_tol=cutMassive)

    def mass(self):
        """Compute the invariant mass of the Four Vector, sqrt(E^2 - |p|^2).

        Returns:
            float: The invariant mass if massive, else 0. 
        """
        if self.is_massive():
            return np.sqrt(self.vec[0]**2 - self.abs_space()**2)
        else:
            return 0.

    def beta(self):
        """Compute beta = |p|/E for a Four-Vector momentum. 

        Returns:
            float: The beta value, maximum is 1. 
        """
        if self.vec[0] > 0:
            return max(1, self.abs_space() / self.vec[0]) #sometimes, beta is slightly greater than 1 (numerical precision issue)
        else:
            return float("inf")

    def theta(self):
        """Compute the polar angle between the spatial vector and the z-axis. 

        Returns:
            float: Polar angle in radians.
        """
        if self.abs_space() > 0:
            return np.arccos(self.vec[3] / self.abs_space())
        else:
            return np.arccos(1.)

    def eT(self):
        """Compute the transverse energy E*sin(theta)

        Returns:
            float: transverse energy
        """
        if self.abs_space() > 0:
            return self.vec[0] * np.sqrt(1 - self.vec[3]**2 / self.abs_space()**2)
        else:
            return 0.

    def pT(self):
        """transverse momentum, magnitude of projection of momentum in xy plane

        Returns:
            float: transverse momentum.
        """
        return np.sqrt(self.vec[1]**2 + self.vec[2]**2)

    def pseudorapidity(self, cut_inf_rap=0):
        """approximation of rapidity for unknown or negligble mass

        Args:
            cut_inf_rap (int, optional): Cutoff to avoid singularities at theta = 0. Defaults to 0.

        Returns:
            float: pseudorapidity η, or +/-inf if undefined.
        """
        if (1 - abs(np.cos(self.theta()))) > cut_inf_rap and self.vec[0] > 0:
            return -np.log(np.tan(self.theta() / 2))
        else:
            if self.vec[3] > 0:
                return float("inf")
            else:
                return float("-inf")

    def rapidity(self):
        """compute the true rapidity for a particle.

        Returns:
            float: rapidity y, or +/-inf if undefined. 
        """
        if self.vec[0] > 0 and self.vec[3] != 0:
            return 0.5 * np.log((self.vec[0] + self.vec[3]) / (self.vec[0] - self.vec[3]))
        else:
            return float("inf") if self.vec[3] > 0 else float("-inf")

    def four_dot(self, other:"FourVector") -> float:
        """Compute the Minkowski dot product with mostly minus convention (+,-,-,-)

        Args:
            other (FourVector): second FourVector

        Returns:
            float: Minkowski scalar product
        """
        metric = np.array([1, -1, -1, -1])
        return np.sum(self.vec[i] * other.vec[i] * metric[i] for i in [0, 1, 2, 3])

    def to_array(self):
        """get the numpy array representation of the FourVector

        Returns:
            np.ndarray: the 4-vector components
        """
        return self.vec

    def tensor_prod(self, other: "FourVector"):
        """compute the tensor product with Minkowski sign convention (+,-,-,-)

        Args:
            other (FourVector): other FourVector to multiply with

        Returns:
            np.ndarray: 4x4 tensor product matrix
        """
        r = np.array([
            [self.vec[0] * other.vec[0], -self.vec[0] * other.vec[1], -self.vec[0] * other.vec[2], -self.vec[0] * other.vec[3]],
            [self.vec[1] * other.vec[0], -self.vec[1] * other.vec[1], -self.vec[1] * other.vec[2], -self.vec[1] * other.vec[3]],
            [self.vec[2] * other.vec[0], -self.vec[2] * other.vec[1], -self.vec[2] * other.vec[2], -self.vec[2] * other.vec[3]],
            [self.vec[3] * other.vec[0], -self.vec[3] * other.vec[1], -self.vec[3] * other.vec[2], -self.vec[3] * other.vec[3]]
        ])
        return r
    
    def is_massless(self):
        """check whether a vector is massless

        Returns:
            bool: True if massless, False else.
        """
        return not self.is_massive()

    def is_same(self, other: "FourVector"):
        """check if two FourVectors are approximately identical

        Args:
            other (FourVector): FourVector to compare to

        Returns:
            bool: True if they are identical within the tolerance, else False.
        """
        diff = self - other
        diff_sq = diff.vec[0]**2 + diff.vec[1]**2 + diff.vec[2]**2 + diff.vec[3]**2
        return diff_sq < toleranceSameVector

    def norm(self) -> float:
        """Compute the Minkowski norm squared. 

        Returns:
            float: Squared norm.
        """
        return self.four_dot(self)

    def spatial_part(self) -> np.ndarray:
        """return the spatial part as a numpy array (x,y,z)

        Returns:
            np.ndarray: spatial components
        """
        return self.vec[1:]

    def energy(self) -> float:
        """energy component

        Returns:
            float: energy
        """
        return self.vec[0]

    def azimuthal_angle(self):
        """compute the azimuthal angle in the xy plane.

        Returns:
            float: azimuthal angle in radians.
        """
        return np.arctan2(self.vec[2], self.vec[1])
    
    def matmul(self, matrix: np.ndarray) -> "FourVector":
        """multiply a vector by a matrix, for example to apply a Lorentz transformation.

        Args:
            matrix (np.ndarray): 4x4 matrix

        Returns:
            FourVector: Transformed FourVector.
        """
        transformed_vec = matrix @ self.vec  
        return FourVector(*transformed_vec)
    
    def spatial_cross(self, other: "FourVector"):
        """Compute the cross-product of the spatial components.

        Args:
            other (FourVector): Another FourVector.

        Returns:
            FourVector: A new four-vector with time component zero and spatial part equal to the cross product.
        """
        # Extract spatial components (x, y, z)
        x1, y1, z1 = self.vec[1], self.vec[2], self.vec[3]
        x2, y2, z2 = other.vec[1], other.vec[2], other.vec[3]

        cx = y1 * z2 - z1 * y2
        cy = z1 * x2 - x1 * z2
        cz = x1 * y2 - y1 * x2

        return FourVector(0, cx, cy, cz)
    
    def cos_theta(self, other: "FourVector"):
        """Compute the cosine of the angle between spatial parts.

        Args:
            other (FourVector): Another FourVector.

        Returns:
            float: Cosine of angle between spatial vectors. 
        """
        return np.dot(self.spatial_part(), other.spatial_part()) / (self.abs_space() * other.abs_space())
    
    def outer_product(self, other: "FourVector"=None):
        """Computes the outer product v ⊗ g·v (or v ⊗ g·other if provided) as a 4x4 matrix, applying the Minkowski metric (+---).

        Args:
            other (FourVector, optional): Other vector; if None, use self. Defaults to None.

        Returns:
            np.ndarray: The 4x4 outer product.
        """
        if other is None:
            other = self

        metric = np.diag([1, -1, -1, -1]) 
        transformed_other = metric @ other.vec  # Apply the metric to the second vector
        return np.outer(self.vec, transformed_other)

    def construct_boost(self, other: "FourVector") -> np.ndarray:
        """Construct the Lorentz boost matrix that brings 'other' into the rest frame of 'self'.
        Boosts along the direction of the spatial momentum of 'self'.

        Args:
            other (FourVector): reference FourVector to be boosted.

        Returns:
            np.ndarray: 4x4 boost matrix.
        """
        p = self.spatial_part()
        E = self.energy()
        p_norm = np.linalg.norm(p)
        
        if p_norm < cutBoost:
            return np.eye(4)  # No boost needed
        
        beta = -p / E
        gamma = 1.0 / np.sqrt(1 - np.dot(beta, beta))
        beta_outer = np.outer(beta, beta)

        boost = np.eye(4)
        boost[0, 0] = gamma
        boost[0, 1:] = gamma * beta
        boost[1:, 0] = gamma * beta
        boost[1:, 1:] += (gamma - 1) * beta_outer / np.dot(beta, beta)

        return boost
    
    def equals_approx(self, other: "FourVector", tol: float = 1e-10) -> bool:
        """Check if two FourVectors are approximately equal, element-wise within tolerance.

        Args:
            other (FourVector): Other FourVector to compare to. 
            tol (float, optional): Tolerance. Defaults to 1e-10.

        Returns:
            bool: True of all elements are close. 
        """
        return np.allclose(self.vec, other.vec, rtol=0, atol=tol)

    def transverse_momentum(self) -> float: 
        """compute the transverse momentum

        Returns:
            float: transverse momentum
        """
        return np.sqrt(self.vec[1]**2 + self.vec[2]**2)
    
    def __mul__(self, scalar: float):
        """multiply two Four-Vectors entry by entry

        Args:
            other (FourVector): second FourVector to multiply

        Returns:
            FourVector: the product self*other
        """
        return FourVector(self.vec[0] * scalar, self.vec[1] * scalar, self.vec[2] * scalar, self.vec[3] * scalar)
    
    def __rmul__(self, scalar: float):
        """Right scalar multiplication

        Args:
            scalar (float): factor to multiply with

        Returns:
            FourVector: the product self*scalar
        """
        return self * scalar  # This reuses the __mul__ method

    def __truediv__(self, scalar: float):
        """scalar division

        Args:
            scalar (float): scalar divisor

        Returns:
            FourVector: scaled FourVector
        """
        return FourVector(self.vec[0] / scalar, self.vec[1] / scalar, self.vec[2] / scalar, self.vec[3] / scalar)
    
    def __repr__(self):
        """make a string representation of the FourVector.

        Returns:
            str: string representation of the FourVector.
        """
        return f"FourVector(t={self.vec[0]}, x={self.vec[1]}, y={self.vec[2]}, z={self.vec[3]})"
    
    def __sub__(self, other: "FourVector"):
        """subtract two Four-Vectors

        Args:
            other (FourVector): second FourVector to subtract

        Returns:
            FourVector: the difference self - other
        """
        return FourVector(self.vec[0] - other.vec[0], self.vec[1] - other.vec[1], self.vec[2] - other.vec[2], self.vec[3] - other.vec[3])

    def __add__(self, other):
        """add two Four-Vectors

        Args:
            other (FourVector): second FourVector to add

        Returns:
            FourVector: the sum self + other
        """
        return FourVector(self.vec[0] + other.vec[0], self.vec[1] + other.vec[1], self.vec[2] + other.vec[2], self.vec[3] + other.vec[3])