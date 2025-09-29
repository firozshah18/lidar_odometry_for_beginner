#!/usr/bin/env python3
"""
@file      se3_utils.py
@brief     SE3 Utilities for LiDAR Odometry
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 2: SE3 Utilities
Provides SE3 (Special Euclidean Group) operations for 3D transformations.
Includes SE3 class with matrix operations, tangent space conversions,
and Lie algebra utilities for pose optimization.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class SE3:
    """SE3 class for 3D rigid body transformations (rotation + translation)"""
    
    def __init__(self, rotation_matrix=None, translation=None):
        """
        Initialize SE3 transformation
        
        Args:
            rotation_matrix: 3x3 rotation matrix (default: identity)
            translation: 3x1 translation vector (default: zero)
        """
        if rotation_matrix is None:
            self.rotation = np.eye(3)
        else:
            self.rotation = np.array(rotation_matrix, dtype=np.float64)
            self.rotation = self._normalize_rotation_matrix(self.rotation)
            
        if translation is None:
            self.translation = np.zeros(3)
        else:
            self.translation = np.array(translation, dtype=np.float64).flatten()
            
    @classmethod
    def from_matrix(cls, matrix):
        """Create SE3 from 4x4 transformation matrix"""
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        return cls(rotation, translation)
    
    @classmethod
    def from_tangent(cls, tangent):
        """Create SE3 from 6D tangent space vector [tx, ty, tz, rx, ry, rz]"""
        tangent = np.array(tangent, dtype=np.float64).flatten()
        translation = tangent[:3]
        rotation_vector = tangent[3:]
        
        # Convert rotation vector to rotation matrix using Rodrigues formula
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        
        return cls(rotation_matrix, translation)
    
    def to_matrix(self):
        """Convert to 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T
    
    def to_tangent(self):
        """Convert to 6D tangent space vector [tx, ty, tz, rx, ry, rz]"""
        rotation_vector = R.from_matrix(self.rotation).as_rotvec()
        return np.concatenate([self.translation, rotation_vector])
    
    def inverse(self):
        """Get inverse transformation"""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return SE3(R_inv, t_inv)
    
    def __mul__(self, other):
        """Compose two SE3 transformations: self * other"""
        if isinstance(other, SE3):
            new_rotation = self.rotation @ other.rotation
            new_translation = self.rotation @ other.translation + self.translation
            return SE3(new_rotation, new_translation)
        elif isinstance(other, np.ndarray):
            # Transform point: T * p
            if other.shape == (3,):
                return self.rotation @ other + self.translation
            elif other.shape == (3, 1):
                result = self.rotation @ other + self.translation.reshape(-1, 1)
                return result
            else:
                raise ValueError(f"Unsupported array shape: {other.shape}")
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
    
    def log(self):
        """Convert SE3 to tangent space using matrix logarithm"""
        return self.to_tangent()
    
    @staticmethod
    def exp(tangent):
        """Convert tangent space to SE3 using matrix exponential"""
        return SE3.from_tangent(tangent)
    
    def _normalize_rotation_matrix(self, R):
        """Normalize rotation matrix using SVD"""
        U, _, Vt = np.linalg.svd(R)
        R_normalized = U @ Vt
        
        # Ensure proper rotation (det = +1)
        if np.linalg.det(R_normalized) < 0:
            Vt[-1, :] *= -1
            R_normalized = U @ Vt
            
        return R_normalized
    
    def __str__(self):
        return f"SE3(\nR=\n{self.rotation}\nt={self.translation})"


def skew_symmetric(v):
    """Create skew-symmetric matrix from 3D vector"""
    v = v.flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def hat(xi):
    """Map 6D vector to 4x4 matrix in se3 Lie algebra"""
    rho = xi[:3]  # translation part
    phi = xi[3:]  # rotation part
    
    Phi = skew_symmetric(phi)
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = Phi
    Xi[:3, 3] = rho
    
    return Xi


def vee(Xi):
    """Map 4x4 matrix in se3 to 6D vector"""
    rho = Xi[:3, 3]
    phi = np.array([Xi[2, 1], Xi[0, 2], Xi[1, 0]])
    return np.concatenate([rho, phi])
