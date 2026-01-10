"""
Module 1.1: Linear Algebra - Eigendecomposition (Power Iteration)
=================================================================

Eigenvalues and Eigenvectors are central to many ML algorithms, including
PCA (Principal Component Analysis) and SVD (Singular Value Decomposition).

**Business Context: Quantum Logistics Inc.**
We analyze the 'connectivity matrix' of our drone swarm. The dominant
eigenvector represents the most 'central' or 'stable' configuration mode
of the swarm.

## Mathematical Foundations

1.  **Eigenvalue Equation**:
    For a square matrix $\mathbf{A}$, a non-zero vector $\mathbf{v}$ is an
    eigenvector if:
    $$ \mathbf{A} \mathbf{v} = \lambda \mathbf{v} $$
    where $\lambda$ is the eigenvalue.

2.  **Power Iteration Algorithm**:
    A numerical method to find the dominant eigenvalue (largest absolute value)
    and its corresponding eigenvector.
    
    Algorithm:
    1. Start with random vector $b_0$.
    2. Iterate:
       $$ b_{k+1} = \frac{\mathbf{A} b_k}{||\mathbf{A} b_k||} $$
    3. Converges to dominant eigenvector.
"""

import random
from vectors_matrices import Matrix, Vector

def power_iteration(matrix: Matrix, num_simulations: int = 100) -> tuple[float, Vector]:
    r"""
    Computes the dominant eigenvalue and eigenvector using Power Iteration.
    
    Args:
        matrix: A square Matrix object.
        num_simulations: Number of iterations to run.
        
    Returns:
        (eigenvalue, eigenvector) tuple.
    """
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square for eigendecomposition.")
    
    # 1. Initialize random vector b_k
    # We use a deterministic seed for reproducibility in course materials
    random.seed(42) 
    b_k = Vector([random.random() for _ in range(matrix.cols)])
    
    # Normalize initial vector
    norm = b_k.norm()
    b_k = b_k.scale(1.0 / norm)

    for _ in range(num_simulations):
        # 2. Multiply by Matrix: A * b_k
        # This stretches the vector towards the dominant characteristic direction
        product = matrix.matmul(b_k) 
        
        if not isinstance(product, Vector):
             # Should not happen given logic
             raise TypeError("Expected Vector output from matmul.")

        # 3. Normalize
        new_norm = product.norm()
        if new_norm == 0:
            return 0.0, b_k # Degenerate case
            
        b_k = product.scale(1.0 / new_norm)

    # 4. Rayleigh Quotient to find Eigenvalue lambda
    # \lambda = (b_k^T * A * b_k) / (b_k^T * b_k)
    # Since b_k is normalized, denominator is 1.
    Ab_k = matrix.matmul(b_k)
    eigenvalue = b_k.dot(Ab_k)
    
    return eigenvalue, b_k


def qc_swarm_stability_demo():
    print("--- Quantum Logistics Inc. Swarm Stability Analysis ---")
    
    # Scenario: A 3x3 Connectivity/Transition Matrix for a micro-swarm of 3 drones.
    # Represents probability or strength of signal transmission between drones.
    # Symmetric matrix implies bidirectional connection strength.
    # [ 2  1  0 ]
    # [ 1  2  1 ]
    # [ 0  1  2 ]
    data = [
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 2.0]
    ]
    connectivity_matrix = Matrix(data)
    print(f"Swarm Connectivity Matrix:\n{connectivity_matrix}")

    print("\nCalculations for Dominant Eigenmode (Power Iteration)...")
    eigen_val, eigen_vec = power_iteration(connectivity_matrix)
    
    print(f"Dominant Eigenvalue (System Stability Metric): {eigen_val:.4f}")
    print(f"Dominant Eigenvector (Principal Mode): {eigen_vec}")
    
    # Verification: A * v approx lambda * v
    print("\nVerification check:")
    Av = connectivity_matrix.matmul(eigen_vec)
    lambda_v = eigen_vec.scale(eigen_val)
    
    print(f"A * v      = {Av}")
    print(f"lambda * v = {lambda_v}")
    
    # Check error
    diff = Av.add(lambda_v.scale(-1)) # Av - lambda*v
    error = diff.norm()
    print(f"Approximation Error: {error:.8f}")
    
    if error < 1e-4:
        print(">> Analysis: SUCCESS. Principal Mode identified.")
    else:
        print(">> Analysis: WARNING. Convergence issues detected.")

if __name__ == "__main__":
    qc_swarm_stability_demo()
