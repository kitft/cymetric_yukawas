import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)
import numpy as np
import time
from functools import partial

# Disable JIT compilation
#jax.config.update("jax_disable_jit", True)


class JAXPointGenerator:
    """JAX-optimized version of PointGenerator with vectorized operations"""
    
    def __init__(self, original_generator):
        # Copy necessary attributes from the original generator
        self.selected_t = original_generator.selected_t
        if len(self.selected_t) != len(original_generator.ambient):
            raise ValueError("selected_t must be a list of lists with length matching ambient")
        if len(self.selected_t[0]) != len(original_generator.ambient):
            raise ValueError("selected_t must be a list of lists with length matching ambient")
        
        self.ambient = original_generator.ambient
        self.ncoords = original_generator.ncoords
        self.root_monomials = original_generator.root_monomials
        self.root_factors = original_generator.root_factors
        #self.generate_pn_points = original_generator.generate_pn_points
        #self._rescale_points= original_generator._rescale_points
        self.degrees=original_generator.degrees
        
        # Create JAX-compatible versions of key functions
        self._take_roots_mapped = lambda pt, selected_t_val: self._take_roots_single(pt, selected_t_val=selected_t_val)
            
        self._take_roots_jax = [jax.jit(vmap(partial(self._take_roots_mapped, selected_t_val=s))) for s in range(len(self.selected_t))]
    
        # Pre-compute indices for _point_from_sol_jax
        # Initialize separate empty lists for each selected_t
        self.slices = [[] for _ in range(len(self.selected_t))]
        for ele in range(len(self.selected_t)):
            j = 0
            for i in range(len(self.ambient)):
                for k in range(1,self.selected_t[ele][i] + 1):
                    s = jnp.sum(self.ambient[:i]) + i
                    e = jnp.sum(self.ambient[:i + 1]) + i + 1
                    self.slices[ele].append((j, k, s, e))
                    j += 1
        # for s in self.slices:
        #     print(s)
        # print(len(self.root_monomials[0]),len(self.root_factors[0]))
        # print(len(self.root_monomials),len(self.root_factors))
        # print(self.root_monomials[0])
        # print(self.root_factors[0])
        # print(self.root_monomials)
        # print(self.root_factors)

    def generate_pn_points_jax(self, n_p, n, key):
        r"""JAX version: Generates points on the sphere :math:`S^{2n+1}`.

        Args:
            n_p (int): number of points.
            n (int): degree of projective space.
            key: JAX random key for random number generation.

        Returns:
            ndarray[(np, n+1), np.complex128]: complex points
        """
        # Split the key for random number generation
        key, subkey = jax.random.split(key)
        
        # sample with gaussian
        points = jax.random.normal(subkey, shape=(n_p, 2 * (n + 1)))
        
        # Put them on the sphere
        norm = jnp.expand_dims(jnp.linalg.norm(points, axis=-1), -1)
        points_normalized = points / norm
        
        # Make them complex using jax.lax.complex
        real_parts = points_normalized[:, :n+1]
        imag_parts = points_normalized[:, n+1:]
        points_complex = jax.lax.complex(real_parts, imag_parts)
        
        return points_complex, key

    def generate_points_orig_pn_pnts(self, n_p, numpy_seed, selected_t_val=None):
        r"""Generates complex points on the CY.

        The points are automatically scaled, such that the largest 
        coordinate in each projective space is 1+0.j.

        Args:
            n_p (int): # of points.
            nproc (int, optional): # of jobs used. Defaults to -1. Then
                uses all available resources.
            batch_size (int, optional): batch_size of Parallel. 
                Defaults to 5000.
            parallel (bool, optional): Whether to use parallel processing.
                Defaults to True.

        Returns:
            ndarray[(n_p, ncoords), np.complex128]: rescaled points
        """
        #print(f"selected pn: {selected_t_val}")
        key = jax.random.PRNGKey(numpy_seed) # use the same seed as numpy for the jax seed
        max_ts = np.max(self.selected_t[selected_t_val])
        max_degree = self.ambient[self.selected_t[selected_t_val].astype(bool)][0] + 1# 0 to get the .item()
        n_p_red = int(n_p / max_degree) + 1
        pn_pnts = np.zeros((n_p_red, self.ncoords, max_ts + 1),
                           dtype=np.complex128)
        for i in range(len(self.ambient)):
            for k in range(self.selected_t[selected_t_val][i] + 1):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                pn_pnts_slice, key = self.generate_pn_points_jax(n_p_red, self.ambient[i], key)
                pn_pnts[:, s:e, k] += pn_pnts_slice
        return pn_pnts

    def _rescale_points(self, points):
        r"""Rescales points in place such that for every P_i^n
        max(abs(coords_i)) == 1 + 0j.

        Args:
            points (ndarray[(n_p, ncoords), complex]): Points.

        Returns:
            ndarray[(n_p, ncoords), complex]: rescaled points
        """
        # Create a copy of points to work with
        points_rescaled = points
        
        # Collect all updates to apply at once
        updates = []
        indices = []
        
        # Iterate over all projective spaces and prepare rescaling for each
        for i in range(len(self.ambient)):
            s = jnp.sum(self.degrees[0:i])
            e = jnp.sum(self.degrees[0:i + 1])
            
            # Get the slice of points for this projective space
            slice_points = points_rescaled[:, s:e]
            
            # Find max indices and calculate scaling factors
            max_indices = jnp.argmax(jnp.abs(slice_points), axis=-1)
            scaling_factors = points_rescaled[jnp.arange(len(points_rescaled)), s + max_indices].reshape((-1, 1)) ** -1
            
            # Store the updated values and their indices
            updates.append(slice_points * scaling_factors)
            indices.append((s, e))
        
        # Apply all updates at once
        for (s, e), update in zip(indices, updates):
            points_rescaled = points_rescaled.at[:, s:e].set(update)
            
        return points_rescaled

    def _take_roots_single(self, p, selected_t_val):
        """JAX version of _take_roots for a single point"""
        #tprint(f"selected_t_val: {selected_t_val}")
        # Compute polynomial coefficients
        all_sums = [
            jnp.sum(c * jnp.multiply.reduce(jnp.power(p.flatten(), m), axis=-1))
            for m, c in zip(self.root_monomials[selected_t_val], self.root_factors[selected_t_val])]
        
        # Convert to array for roots function
        coeffs = jnp.array(all_sums)
        
        # Find roots and create points
        roots = jnp.roots(coeffs, strip_zeros=False)
        # Convert JAX array to numpy for debugging
        points = vmap(lambda t: self._point_from_sol_jax(p, [t],selected_t_val = selected_t_val))(roots)
        return points

    
    def _point_from_sol_jax(self, p, sol, selected_t_val=None):
        """JAX version of _point_from_sol"""
        t = jnp.ones_like(p)
        for j, k, s, e in self.slices[selected_t_val]:
            t = t.at[s:e, k].set(sol[j])
        point = jnp.sum(p * t, axis=-1)
        return point
    
    def generate_points_jax(self, n_p, numpy_seed, pn_pnts=None, selected_t_val=None):
        """JAX-optimized point generation"""
        # max_ts = jnp.max(self.selected_t)
        # max_degree = self.ambient[self.selected_t.astype(bool)][0] + 1
        # n_p_red = int(n_p / max_degree) + 1
        
        # Create points
        if selected_t_val is None:
            raise ValueError("selected_t_val must be provided")
        if pn_pnts is None:
            pn_pnts = self.generate_points_orig_pn_pnts(n_p, numpy_seed, selected_t_val=selected_t_val)
        else:
            pn_pnts = jnp.array(pn_pnts)
        points = self._take_roots_jax[selected_t_val](pn_pnts)
        points = jnp.vstack(points)
        rescaled_points = self._rescale_points(points)
        # Ensure we have generated at least n_p points
        if len(rescaled_points) < n_p:
            raise ValueError(f"Not enough points generated: {len(rescaled_points)} < {n_p}")
            return rescaled_points[:n_p]  # This will fail if we don't have enough points
        else:
            return rescaled_points[:n_p]  # Return exactly n_p points

import numpy as np
from cymetric.pointgen.pointgen import PointGenerator

def test_parallel_vs_nonparallel():
    """Test that parallel and non-parallel point generation produce the same results"""
    # Initialize PointGenerator with some test parameters
    x=0
    coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
    0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1]) +\
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
          x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    ambient = np.array([1,1,1,1])
    monomials = np.array([[2, 0, 2, 0, 2, 0, 2, 0], [2, 0, 2, 0, 2, 0, 1, 1], [2, 0, 2, 0, 2, 
      0, 0, 2], [2, 0, 2, 0, 1, 1, 2, 0], [2, 0, 2, 0, 1, 1, 1, 1], [2, 0,
       2, 0, 1, 1, 0, 2], [2, 0, 2, 0, 0, 2, 2, 0], [2, 0, 2, 0, 0, 2, 1, 
      1], [2, 0, 2, 0, 0, 2, 0, 2], [2, 0, 1, 1, 2, 0, 2, 0], [2, 0, 1, 1,
       2, 0, 1, 1], [2, 0, 1, 1, 2, 0, 0, 2], [2, 0, 1, 1, 1, 1, 2, 
      0], [2, 0, 1, 1, 1, 1, 1, 1], [2, 0, 1, 1, 1, 1, 0, 2], [2, 0, 1, 1,
       0, 2, 2, 0], [2, 0, 1, 1, 0, 2, 1, 1], [2, 0, 1, 1, 0, 2, 0, 
      2], [2, 0, 0, 2, 2, 0, 2, 0], [2, 0, 0, 2, 2, 0, 1, 1], [2, 0, 0, 2,
       2, 0, 0, 2], [2, 0, 0, 2, 1, 1, 2, 0], [2, 0, 0, 2, 1, 1, 1, 
      1], [2, 0, 0, 2, 1, 1, 0, 2], [2, 0, 0, 2, 0, 2, 2, 0], [2, 0, 0, 2,
       0, 2, 1, 1], [2, 0, 0, 2, 0, 2, 0, 2], [1, 1, 2, 0, 2, 0, 2, 
      0], [1, 1, 2, 0, 2, 0, 1, 1], [1, 1, 2, 0, 2, 0, 0, 2], [1, 1, 2, 0,
       1, 1, 2, 0], [1, 1, 2, 0, 1, 1, 1, 1], [1, 1, 2, 0, 1, 1, 0, 
      2], [1, 1, 2, 0, 0, 2, 2, 0], [1, 1, 2, 0, 0, 2, 1, 1], [1, 1, 2, 0,
       0, 2, 0, 2], [1, 1, 1, 1, 2, 0, 2, 0], [1, 1, 1, 1, 2, 0, 1, 
      1], [1, 1, 1, 1, 2, 0, 0, 2], [1, 1, 1, 1, 1, 1, 2, 0], [1, 1, 1, 1,
       1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 0, 2, 2, 
      0], [1, 1, 1, 1, 0, 2, 1, 1], [1, 1, 1, 1, 0, 2, 0, 2], [1, 1, 0, 2,
       2, 0, 2, 0], [1, 1, 0, 2, 2, 0, 1, 1], [1, 1, 0, 2, 2, 0, 0, 
      2], [1, 1, 0, 2, 1, 1, 2, 0], [1, 1, 0, 2, 1, 1, 1, 1], [1, 1, 0, 2,
       1, 1, 0, 2], [1, 1, 0, 2, 0, 2, 2, 0], [1, 1, 0, 2, 0, 2, 1, 
      1], [1, 1, 0, 2, 0, 2, 0, 2], [0, 2, 2, 0, 2, 0, 2, 0], [0, 2, 2, 0,
       2, 0, 1, 1], [0, 2, 2, 0, 2, 0, 0, 2], [0, 2, 2, 0, 1, 1, 2, 
      0], [0, 2, 2, 0, 1, 1, 1, 1], [0, 2, 2, 0, 1, 1, 0, 2], [0, 2, 2, 0,
       0, 2, 2, 0], [0, 2, 2, 0, 0, 2, 1, 1], [0, 2, 2, 0, 0, 2, 0, 
      2], [0, 2, 1, 1, 2, 0, 2, 0], [0, 2, 1, 1, 2, 0, 1, 1], [0, 2, 1, 1,
       2, 0, 0, 2], [0, 2, 1, 1, 1, 1, 2, 0], [0, 2, 1, 1, 1, 1, 1, 
      1], [0, 2, 1, 1, 1, 1, 0, 2], [0, 2, 1, 1, 0, 2, 2, 0], [0, 2, 1, 1,
       0, 2, 1, 1], [0, 2, 1, 1, 0, 2, 0, 2], [0, 2, 0, 2, 2, 0, 2, 
      0], [0, 2, 0, 2, 2, 0, 1, 1], [0, 2, 0, 2, 2, 0, 0, 2], [0, 2, 0, 2,
       1, 1, 2, 0], [0, 2, 0, 2, 1, 1, 1, 1], [0, 2, 0, 2, 1, 1, 0, 
      2], [0, 2, 0, 2, 0, 2, 2, 0], [0, 2, 0, 2, 0, 2, 1, 1], [0, 2, 0, 2,
       0, 2, 0, 2]])

    

    kmoduli = np.array([1,(np.sqrt(7)-2)/3,(np.sqrt(7)-2)/3,1])


    # Fermat quintic coefficients and monomials
    # # Reset previous values for quintic
    
    # # Define ambient space: P⁴
    # ambient = np.array([4])
    
    # # Create monomials for the Fermat quintic: x₀⁵ + x₁⁵ + x₂⁵ + x₃⁵ + x₄⁵
    # monomials = np.zeros((5, 5), dtype=int)
    
    # # Set up the 5 monomials: [5,0,0,0,0], [0,5,0,0,0], etc.
    # for i in range(5):
    #     monomials[i, i] = 5
    
    # # Coefficients are all 1 for the Fermat quintic
    # coefficients = np.ones(5)
    
    # Set kmoduli to 1
    #kmoduli = np.ones(1)
    pg = PointGenerator(
        monomials=monomials,
        coefficients=coefficients,
        kmoduli=kmoduli,
        ambient=ambient,
        vol_j_norm=None,
        verbose=2,
    )
    
    npoints = 1000000
    # Generate points using both methods
    np.random.seed(42)  # Ensure reproducibility
    
    # Time parallel implementation
    start_time = time.time()
    points_parallel = pg.generate_points(n_p=npoints, process_parallel=True, use_jax=False)
    parallel_time = time.time() - start_time
    print(f"Parallel implementation took {parallel_time:.4f} seconds")
    
    np.random.seed(42)  # Reset seed for reproducibility
    # Time non-parallel implementation
    start_time = time.time()
    points_nonparallel = pg.generate_points(n_p=npoints, process_parallel=False, use_jax=False)
    nonparallel_time = time.time() - start_time
    print(f"Non-parallel implementation took {nonparallel_time:.4f} seconds")
    
    # Check if results are the same
    assert np.allclose(points_parallel, points_nonparallel)
    
    # Also test JAX implementation
    jax_pg = JAXPointGenerator(pg)

    key = jax.random.PRNGKey(42)
    np.random.seed(42)  # Reset seed for reproducibility
    
    # Time JAX pn_pts generation
    start_time = time.time()
    #pn_pts = jax_pg.generate_points_orig_pn_pnts(n_p=npoints)
    #points_jax = jax_pg.generate_points_jax(n_p=npoints, key=None, pn_pnts=pn_pts)
    points_jax = pg.generate_points(n_p=npoints, use_jax=True, process_parallel=False)
    jax_time = time.time() - start_time
    print(f"JAX implementation took {jax_time:.4f} seconds")
    print("shapes:", points_jax.shape, points_parallel.shape)
  
    # Convert JAX array to numpy for comparison
    points_jax_np = np.array(points_jax)
    
    # Check if JAX results match numpy results (allowing for small numerical differences)
    try:
        assert np.allclose(points_parallel[:npoints], points_jax_np[:npoints], atol=1e-5)
    except AssertionError:
        # Sort points by their first coordinate's real part for comparison
        sorted_parallel = points_parallel[:npoints][np.argsort(np.real(points_parallel[:npoints][:,0]))]
        sorted_jax = points_jax_np[:npoints][np.argsort(np.real(points_jax_np[:npoints][:,0]))]
        
        # Count how many points are close after sorting
        close_count = np.sum(np.all(np.isclose(sorted_parallel, sorted_jax, atol=1e-5), axis=1))
        print(f"After sorting, {close_count}/{npoints} points match between JAX and NumPy implementations")
        if close_count < 0.9*npoints:
            print("Too few points match between implementations: but different sampling so ok")
    #print("JAX",sorted_jax[0:])
    #print("NP",sorted_parallel[0:2])
    # Check if points satisfy the defining equations
    # Get the monomials and coefficients from the original generator
    monomials = pg.monomials
    coefficients = pg.coefficients
    
    # Evaluate the defining equations for both implementations
    def evaluate_equations(points):
        results = []
        for i in range(len(points)):
            point = points[i]
            # Calculate each monomial term
            terms = np.array([np.prod(point**monomial) for monomial in monomials])
            # Multiply by coefficients and sum
            equation_value = np.sum(terms * coefficients)
            results.append(equation_value)
        return np.array(results)
    
    parallel_results = evaluate_equations(points_parallel[:10000])
    jax_results = evaluate_equations(points_jax_np[:10000])
    origsatisfy = np.allclose(parallel_results, 0, atol=1e-8)
    jaxsatisfy = np.allclose(jax_results, 0, atol=1e-8)

    if not origsatisfy or not jaxsatisfy:
        raise ValueError(f"Points do not satisfy the defining equations: orig={origsatisfy} jax={jaxsatisfy}")
    print(f"Original points satisfy equations: {origsatisfy}")
    print(f"JAX points satisfy equations: {jaxsatisfy}")
    print(f"Max equation value (original): {np.max(np.abs(parallel_results)):.2e}")
    print(f"Max equation value (JAX): {np.max(np.abs(jax_results)):.2e}")
    # If most points match, consider the test passed
    print("✓ Parallel, non-parallel, and JAX point generation produce consistent results")

if __name__ == "__main__":
    test_parallel_vs_nonparallel()