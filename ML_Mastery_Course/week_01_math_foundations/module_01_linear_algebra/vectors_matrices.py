"""
Module 1.1: Linear Algebra - Vectors and Matrices
=================================================

This module implements the fundamental building blocks of Machine Learning:
Vectors and Matrices. We build these structures from `scratch` to understand
the underlying operations that power libraries like NumPy and PyTorch.

**Business Context: Quantum Logistics Inc.**
In our automated fleet, every drone's state is a vector, and every
navigation maneuver is a matrix transformation.

## Mathematical Foundations
 All operations follow standard Linear Algebra rigor.

1.  **Vector**: An element of a vector space $\\mathbb{R}^n$.
    $$ \\mathbf{v} = [v_1, v_2, ..., v_n]^T $$

2.  **Dot Product**: A measure of directional alignment.
    $$ \\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^{n} a_i b_i = ||\\mathbf{a}|| ||\\mathbf{b}|| \\cos(\\theta) $$

3.  **Matrix-Vector Multiplication**: Linear transformation.
    $$ \\mathbf{Ax} = \\mathbf{b} $$
"""

import math
from typing import List, Union, Tuple

# Type Alias for clarity
Scalar = float
VectorData = List[Scalar]
MatrixData = List[List[Scalar]]

class Vector:
    r"""
    Represents a mathematical vector in $\mathbb{R}^n$.
    Example: A Drone's state vector [x_pos, y_pos, z_pos, battery_level].
    """
    def __init__(self, data: VectorData):
        self.data = data
        self.dim = len(data)

    def __repr__(self):
        return f"Vector({self.data})"

    def add(self, other: 'Vector') -> 'Vector':
        r"""
        Element-wise addition: $\mathbf{u} + \mathbf{v}$.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def scale(self, scalar: Scalar) -> 'Vector':
        r"""
        Scalar multiplication: $c \cdot \mathbf{v}$.
        """
        return Vector([x * scalar for x in self.data])

    def dot(self, other: 'Vector') -> Scalar:
        r"""
        Computes the dot product: $\mathbf{a}^T \mathbf{b}$.
        Used to determine if two drone paths are aligned.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self) -> Scalar:
        r"""
        Computes the L2 Norm (Euclidean Length): $||\mathbf{v}||_2$.
        $$ ||\mathbf{v}|| = \sqrt{\sum v_i^2} $$
        """
        return math.sqrt(self.dot(self))


class Matrix:
    r"""
    Represents a mathematical matrix in $\mathbb{R}^{m \times n}$.
    Example: A rotation matrix to adjust drone heading.
    """
    def __init__(self, data: MatrixData):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

    def __repr__(self):
        return f"Matrix({self.data})"

    def transpose(self) -> 'Matrix':
        r"""
        Computes the transpose $\mathbf{A}^T$.
        Swaps rows and columns.
        """
        new_data = [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)]
        return Matrix(new_data)

    def matmul(self, other: Union['Matrix', 'Vector']) -> Union['Matrix', 'Vector']:
        r"""
        Matrix multiplication: $\mathbf{A} \cdot \mathbf{B}$ or $\mathbf{A} \cdot \mathbf{x}$.
        
        If other is Vector(n), returns Vector(m).
        If other is Matrix(n x p), returns Matrix(m x p).
        """
        if isinstance(other, Vector):
            # Matrix-Vector Multiplication
            if self.cols != other.dim:
                raise ValueError(f"Shape mismatch: ({self.rows},{self.cols}) vs ({other.dim}, 1)")
            
            result_data = []
            for r in range(self.rows):
                row_val = sum(self.data[r][c] * other.data[c] for c in range(self.cols))
                result_data.append(row_val)
            return Vector(result_data)
        
        elif isinstance(other, Matrix):
            # Matrix-Matrix Multiplication
            if self.cols != other.rows:
                raise ValueError(f"Shape mismatch: ({self.rows},{self.cols}) vs ({other.rows}, {other.cols})")
            
            result_data = []
            for r in range(self.rows):
                row = []
                for c in range(other.cols):
                    # Dot product of self-row and other-col
                    val = sum(self.data[r][k] * other.data[k][c] for k in range(self.cols))
                    row.append(val)
                result_data.append(row)
            return Matrix(result_data)
        else:
            raise TypeError("Unsupported type for matmul")

# ==============================================================================
# Quantum Logistics Inc. - Applied Examples
# ==============================================================================

def qc_logistics_demo():
    print("--- Quantum Logistics Inc. System Initialization ---")
    
    # Scenario: Drone "Icarus-1" state [x, y, z] in normalized Grid Units
    # We ignore velocity for this simple geometric check.
    icarus_pos = Vector([10.0, 20.0, 5.0])
    print(f"Icarus-1 Position: {icarus_pos}")

    # Scenario: Wind factor acting on the drone
    wind_force = Vector([-2.0, 1.0, 0.0]) 
    print(f"Wind Force Vector: {wind_force}")

    # Calculate new approximated position after one time unit (Euler integration step)
    # New Pos = Old Pos + Wind Force
    new_pos = icarus_pos.add(wind_force)
    print(f"Projected Position (1s): {new_pos}")

    # Scenario: Check alignment with Landing Pad
    # Landing Pad is at [100, 200, 0] direction vector
    landing_vector = Vector([100.0, 200.0, 0.0])
    
    # Cosine Similarity check
    # cos(theta) = (a . b) / (|a| |b|)
    dot_prod = new_pos.dot(landing_vector)
    mag_v = new_pos.norm()
    mag_l = landing_vector.norm()
    
    cosine_sim = dot_prod / (mag_v * mag_l)
    print(f"Alignment with Landing Vector (Cosine Similarity): {cosine_sim:.4f}")
    if cosine_sim > 0.99:
        print(">> Status: ON COURSE")
    else:
        print(f">> Status: COURSE CORRECTION REQUIRED (Alignment: {cosine_sim:.4f})")

    print("\n--- Matrix Transformation Demo ---")
    # Scenario: Rotate the drone 90 degrees around Z axis to face new delivery zone
    # Rotation Matrix for 90 deg (pi/2) around Z:
    # [ 0 -1  0 ]
    # [ 1  0  0 ]
    # [ 0  0  1 ]
    rotation_data = [
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ]
    rot_matrix = Matrix(rotation_data)
    print("Executing Z-Axis Rotation Maneuver...")
    rotated_pos = rot_matrix.matmul(icarus_pos)
    print(f"Original Position: {icarus_pos}")
    print(f"Rotated Position:  {rotated_pos}")

if __name__ == "__main__":
    qc_logistics_demo()
