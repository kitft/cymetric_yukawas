import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)
import numpy as np
import time
from functools import partial

# Disable JIT compilation
jax.config.update("jax_disable_jit", True)
from cymetric.pointgen.pointgen_jax_gmpy2 import gmpy2_poly_solver


class JAXPointGeneratorQuadratic_Torus:
    """JAX-optimized version of PointGenerator with vectorized operations"""
    
    def __init__(self, original_generator, max_iter = 10, tol = 1e-20):
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
        self.monomials = original_generator.monomials
        self.coefficients = original_generator.coefficients
        self.actual_point_generator = IntersectionSolverJAX_torus( monomials=self.monomials, coefficients=self.coefficients, use_gmpy2=True, max_iter = max_iter, tol = tol)
        self.use_gmpy2 = True


        self.lc = jnp.array(original_generator.lc, dtype=jnp.complex128)
        self.pullback_all_3_spaces_vmap = jax.vmap(lambda x, pullback_to_manifold: self.pullback_all_3_spaces(x, pullback_to_manifold, self.lc))
   
    

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

    
    def generate_points_jax(self, n_p, numpy_seed):
        """JAX-optimized point generation"""
        # max_ts = jnp.max(self.selected_t)
        # max_degree = self.ambient[self.selected_t.astype(bool)][0] + 1
        # n_p_red = int(n_p / max_degree) + 1
        
        # Create points
    
        #points = self._take_roots_jax[selected_t_val](pn_pnts)
        key = jax.random.PRNGKey(numpy_seed)
        print(f"generating {n_p//8 + 1} polynomials at a time for {n_p} points")
        points = self.actual_point_generator.solve_batch(key, batch_size=(n_p//4+1), return_residuals=False)
        rescaled_points = self._rescale_points(points)
        # Ensure we have generated at least n_p points
        if len(rescaled_points) < n_p:
            raise ValueError(f"Not enough points generated: {len(rescaled_points)} < {n_p}")
            return rescaled_points[:n_p]  # This will fail if we don't have enough points
        else:
            return rescaled_points[:n_p]  # Return exactly n_p points

    def get_weights_jax_quadratic(self, points,pullbacks):
        return self.pullback_all_3_spaces_vmap(points, pullbacks, self.lc)

    def map_p1x_p1_to_p3(self, coords):
        """
        Given coords = [x0, x1, y0, y1] in complex form,
        return [X0, X1, X2, X3] = [x0*y0, x0*y1, x1*y0, x1*y1].
        """
        x0, x1, y0, y1 = coords
        return jnp.array([x0*y0, x0*y1, x1*y0, x1*y1], dtype=coords.dtype)

    def fubini_study_p3(self, Z):
        """
        Standard homogeneous Fubini–Study metric on P^3 in 4D homogeneous coords Z.
        Returns a 4x4 matrix G(Z) (degenerate along the Euler vector), i.e.
           G_ij = (|Z|^2 * δ_ij - Z_i conj(Z_j)) / |Z|^4.
        """
        norm_sq = jnp.sum(jnp.abs(Z)**2)
        outer   = jnp.outer(jnp.conjugate(Z),Z)# as it shoudl look like (z dzbar) (z dzbar)
        eye     = jnp.eye(4, dtype=Z.dtype)
        # The homogeneous FS form in matrix form:
        return (1/jnp.pi)*(norm_sq * eye - outer) / (norm_sq**2)

    def pullback_fubini_study(self, coords):
        """
        Pull back the FS metric on P^3 via the map (x0,x1,y0,y1) -> (X0,X1,X2,X3).
        We compute dF^dagger * G * dF, where G is the 4x4 FS metric at F(coords),
        and dF is the Jacobian of F w.r.t. coords.

        Returns a 4x4 complex matrix, corresponding to partial derivatives
        wrt (x0, x1, y0, y1).
        """
        # Target point in P^3:
        Z = self.map_p1x_p1_to_p3(coords)
        # Ambient FS metric (4x4):
        G = self.fubini_study_p3(Z)

        # Jacobian dF: 4 x 4 (treating x0,x1,y0,y1 as independent complex coords).
        # We can just code the known partial derivatives explicitly:
        x0, x1, y0, y1 = coords
        dF = jnp.array([
            [      y0,    0+0j,      x0,    0+0j],
            [      y1,    0+0j,      0+0j,       x0],
            [    0+0j,      y0,      x1,    0+0j],
            [    0+0j,      y1,    0+0j,       x1]
        ], dtype=coords.dtype)# second index is the denominator! so should sum over 1st

        # Pullback = dF^\dagger * G * dF
        # In complex differential geometry, we typically form J^H G J,
        # where ^H is the conjugate-transpose.
        conjugate_dF = jnp.conjugate(dF)
        return jnp.einsum('Ia,IJ,Jb->ab', dF, G, conjugate_dF)

    def pullback_all_3_spaces(self, coords, pullback_to_manifold, lc):


        coords1 = coords[:4]

        pb_metric1 = self.pullback_fubini_study(coords1)

        pb_pb_metric1 = jnp.einsum('ai,ij,bj->ab', pullback_to_manifold[:,0:4], pb_metric1, jnp.conjugate(pullback_to_manifold[:,0:4]))
        return pb_pb_metric1[0][0]

   




import sympy
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

@jax.jit
def projective_normalize(a, b, tol=1e-15):
    """
    If |a|>=|b|>tol => (1, b/a)
    else if |b|>|a|>tol => (a/b, 1)
    else => (0,0)
    """
    #out =  jnp.array([a, b])
    #out = out /jnp.sqrt(jnp.sum(jnp.abs(out)**2,axis=-1,keepdims=True))
    #return out
    ##
    mag_a = jnp.abs(a)
    mag_b = jnp.abs(b)
    cond_a = (mag_a >= mag_b) & (mag_a > tol)
    cond_b = (mag_b > mag_a) & (mag_b > tol)
    #return jnp.array([a,b])

    def case_a(_):
        return jnp.array([1+0j, b/a])
    def case_b(_):
        return jnp.array([a/b, 1+0j])
    def case_none(_):
        return jnp.array([jnp.nan + 0j, jnp.nan + 0j])

    return jax.lax.cond(cond_a, case_a,
           lambda x: jax.lax.cond(cond_b, case_b, case_none, x),
           None)

class IntersectionSolverJAX_torus:
    """
    Pure JAX solver for:
      eqn1: c_{ij} x_i y_j = 0
      eqn2: sum over user monomials in (x,y).
    Patches:
      A => x1=1 => univariate in x0
      B => x0=1 => univariate in x1
    Each solve is fully vectorized over 'batch_size' random instances
    (only c_{ij} are randomized).
    """

    def __init__(self, monomials=None, coefficients=None, use_gmpy2=True, max_iter = 10,tol = 1e-20):
        """
        monomials: List[4-int], shape=(N,4). Each row => exponents of
                   [x0,x1,y0,y1].
        coefficients: length-N, complex array => eqn2 monomial coefficients.

        We do the symbolic expansions once (using Sympy),
        then store numeric callables for patchA, patchB polynomials
        plus eqn1 expansion for eqn-checks.
        """
        print("Compiling expressions in sympy.")
        if monomials is None:
           raise ValueError("Monomials must be provided")
        if coefficients is None:
           raise ValueError("Coefficients must be provided")

        self.monomials = monomials
        self.coefficients = coefficients
        self.monomials_jax = jnp.array(monomials)
        self.coefficients_jax = jnp.array(coefficients)

        # -------------------------------------------------
        # 1) Symbolic definitions
        # -------------------------------------------------
        x0, x1 = sympy.symbols('x0 x1', complex=True)
        y0, y1 = sympy.symbols('y0 y1', complex=True)

        c00, c01, c10, c11 = sympy.symbols('c00 c01 c10 c11', complex=True)

        # eqn1 => c_{ij} x_i y_j=0 => (y0,y1)=(b1,-a1)
        eqn1 = (c00*x0*y0 + c01*x0*y1
              + c10*x1*y0 + c11*x1*y1)
        eqn1_simpl = sympy.expand(eqn1)
        a1 = eqn1_simpl.diff(y0).subs({y0:0, y1:0})
        b1 = eqn1_simpl.diff(y1).subs({y0:0, y1:0})
        y0_expr = b1
        y1_expr = -a1

        # eqn2 => sum_{k} coefficients[k]* x0^... x1^... y0^... y1^...
        eqn2_sym = 0
        for cf, (ex0, ex1, ey0, ey1) in zip(coefficients, monomials):
            eqn2_sym += cf*( x0**ex0 * x1**ex1
                            *y0**ey0 *y1**ey1)

        eqn2_sub = eqn2_sym.subs({
            y0: y0_expr, y1: y1_expr
        })
        eqn2_simpl = sympy.simplify(eqn2_sub)

        # Patch A => x1=1 => polynomial in x0
        eqn2A = eqn2_simpl.subs(x1, 1)
        polyA = sympy.poly(eqn2A, x0)
        self.polyA = polyA
        A_coeffs = polyA.all_coeffs()  # highest-degree first

        # Patch B => x0=1 => polynomial in x1
        eqn2B = eqn2_simpl.subs(x0, 1)
        polyB = sympy.poly(eqn2B, x1)
        B_coeffs = polyB.all_coeffs()  # highest-degree first

        # -------------------------------------------------
        # 2) Lambdify expansions for eqn1 (for checks),
        #    plus patch polynomials
        # -------------------------------------------------
        from sympy.printing.numpy import JaxPrinter
        class CustomPrinter(JaxPrinter):
            def _print_Float(self, expr):
                return str(float(expr))

        from sympy.printing.numpy import NumPyPrinter
        class CustomPrinterNUMPY(NumPyPrinter):
            def _print_Float(self, expr):
                return str(float(expr))

        self.y0_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11), y0_expr, 'numpy')
        self.y1_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11), y1_expr, 'numpy')

        # Build numeric patch polynomials:
        varlist = (c00,c01,c10,c11)
        self.patchA_funs = [ jax.jit(sympy.lambdify(varlist, a,docstring_limit=0, cse=True,printer=CustomPrinter))
                             for a in A_coeffs ]
        self.patchB_funs = [ jax.jit(sympy.lambdify(varlist, b,docstring_limit=0, cse=True,printer=CustomPrinter))
                             for b in B_coeffs ]
        self.patchA_coeffs_fun = sympy.lambdify(
            (c00, c01, c10, c11),
            polyA.all_coeffs(),
            'numpy')

        self.patchA_coeffs_funNUMPY = sympy.lambdify(
            (c00, c01, c10, c11),
            polyA.all_coeffs(),
            printer=CustomPrinterNUMPY)

        self.patchA_coeffs_fun_JAXCUSTOM = sympy.lambdify(
            (c00, c01, c10, c11),
            polyB.all_coeffs(),
            printer=CustomPrinter)
        
        self.patchA_coeffs_fun_JAX = sympy.lambdify(
            (c00, c01, c10, c11),
            polyB.all_coeffs(),
            'jax')

        self.funcall_A_batch = jax.vmap(lambda x: jnp.array(self.patchA_coeffs_fun(*x)))
        self.funcall_B_batch = jax.vmap(self.patchB_poly)

        # Build patch A polynomials
        self.get_point_from_solution_batchA = jax.vmap(lambda x, c: self.get_point_from_solution(x,1+0.j, c), in_axes=(0,None))
        self.get_point_from_solution_batchA_nested = jax.vmap(self.get_point_from_solution_batchA, in_axes=(0,0))
        self.get_point_from_solution_batchB = jax.vmap(lambda x, c: self.get_point_from_solution(1+0.j,x, c), in_axes=(0,None))
        self.get_point_from_solution_batchB_nested = jax.vmap(self.get_point_from_solution_batchB, in_axes=(0,0))

        self.eqn_eval_batch = jax.vmap(self.eqn_eval, in_axes=(0,))
        self.eqn_1_eval_batch = jax.vmap(self.eqn_1_eval, in_axes=(0,None))
        self.eqn_1_eval_batch_nested = jax.vmap(self.eqn_1_eval_batch, in_axes=(0,0))

        self.use_gmpy2 = use_gmpy2
        if use_gmpy2:
            self.max_iter = max_iter
            self.tol = tol
            self.roots_solver = gmpy2_poly_solver(max_iter = max_iter, tol = tol)
        else:
            self.roots_solver = self.roots_vmap

    @jax.jit
    def patchA_poly(self, c):
        def apply_fun(fun):
            return fun(*c)
        return jnp.stack(jax.tree.map(apply_fun, self.patchA_funs))

    @jax.jit
    def patchB_poly(self, c):
        def apply_fun(fun):
            return fun(*c)
        return jnp.stack(jax.tree.map(apply_fun, self.patchB_funs))

    @jax.jit
    def get_point_from_solution(self, x0, x1, c):
        x0,x1= projective_normalize(x0,x1)
        c00, c01, c10, c11 = c[0], c[1], c[2], c[3]

        # eqn1
        y0v = self.y0_fun(x0, x1, c00, c01, c10, c11)
        y1v = self.y1_fun(x0, x1, c00, c01, c10, c11)
        ys = projective_normalize(y0v, y1v)

        return jnp.concatenate([jnp.array([x0, x1]), ys])

    def eqn_1_eval(self, point, cs):
        x0, x1, y0, y1 = point
        c00, c01, c10, c11 = cs
        eq1_val = c00*x0*y0 + c01*x0*y1 + c10*x1*y0 + c11*x1*y1
        return jnp.array([eq1_val])

    def eqn_eval(self, point):
        x0, x1, y0, y1 = point
        powers = jnp.array([
            jnp.power(x0, self.monomials_jax[:, 0]),
            jnp.power(x1, self.monomials_jax[:, 1]),
            jnp.power(y0, self.monomials_jax[:, 2]),
            jnp.power(y1, self.monomials_jax[:, 3])
        ])
        monomial_values = jnp.prod(powers, axis=0)
        eq2_val = jnp.sum(self.coefficients_jax * monomial_values)
        return jnp.array(eq2_val, dtype=jnp.complex128)

    @jax.jit
    def generate_and_format_c(self, key, batch_size):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        cs = jax.random.normal(subkey1, (batch_size, 4)) + 1j*jax.random.normal(subkey2, (batch_size, 4))
        cs /= jnp.sqrt(jnp.sum(jnp.abs(cs)**2, axis=1, keepdims=True))
        return cs

    def solve_batch(self, key, batch_size=100000, return_residuals=False):
        c_batch = self.generate_and_format_c(key, batch_size)
        polyA_batch = self.funcall_A_batch(c_batch)
        polyA_batch = polyA_batch/jnp.sqrt(jnp.sum(jnp.abs(polyA_batch)**2, axis=1, keepdims=True))
        if self.use_gmpy2:
            roots = self.roots_solver.solve_batch_of_polys(polyA_batch, max_iter = self.max_iter, tol = self.tol)
            pointsA = self.get_point_from_solution_batchA_nested(np.array(roots), c_batch).reshape(-1,4)
            if not return_residuals:
                return pointsA
            else:
                eqA_vals = self.eqn_eval_batch(pointsA.reshape(-1,4))
                eq1_Aevals = self.eqn_1_eval_batch_nested(pointsA, c_batch)
                eqA_vals_reshaped = eqA_vals.reshape(-1, 4,1)
                Aevals = np.abs(jnp.concatenate([eqA_vals_reshaped, eq1_Aevals], axis=2))
                return pointsA, Aevals
        else:
            polyB_batch = self.funcall_B_batch(c_batch)
            polyB_batch = polyB_batch/jnp.sqrt(jnp.sum(jnp.abs(polyB_batch)**2, axis=1, keepdims=True))
            rootsA = self.roots_NR_vmap(polyA_batch)
            rootsA = jnp.array(rootsA)
            pointsA = self.get_point_from_solution_batchA_nested(rootsA, c_batch)
            rootsB = self.roots_NR_vmap(polyB_batch)
            rootsB = jnp.array(rootsB)
            pointsB = self.get_point_from_solution_batchB_nested(rootsB, c_batch)
            if not return_residuals:
                return pointsA, pointsB
            else:
                eqA_vals = self.eqn_eval_batch(pointsA.reshape(-1,4))
                eqB_vals = self.eqn_eval_batch(pointsB.reshape(-1,4))
                eq1_Aevals = self.eqn_1_eval_batch_nested(pointsA, c_batch)
                eq1_Bevals = self.eqn_1_eval_batch_nested(pointsB, c_batch)
                eqA_vals_reshaped = eqA_vals.reshape(-1, 4,1)
                Aevals = np.abs(jnp.concatenate([eqA_vals_reshaped, eq1_Aevals], axis=2))
                eqB_vals_reshaped = eqB_vals.reshape(-1, 4,1)
                Bevals = np.abs(jnp.concatenate([eqB_vals_reshaped, eq1_Bevals], axis=2))
                return pointsA, pointsB, Aevals, Bevals

    def roots_vmap(self, x):
        return jax.vmap(lambda x: jnp.roots(x, strip_zeros=False))(x)
    def roots_NR_vmap(self, x, max_iter=100):
        return jax.vmap(lambda x: self.roots_NR(x, max_iter=max_iter))(x)
    
    @jax.jit
    def roots_NR(self, coeffs, tol=1e-12, max_iter=100):
        """Find polynomial roots with jnp.roots then refine with Newton-Raphson."""
        # Get initial roots
        initial_roots = jnp.roots(coeffs, strip_zeros=False)
        
        # Define polynomial and its derivative for NR
        @jax.jit
        def poly(x):
            return jnp.polyval(coeffs, x)
        
        @jax.jit
        def poly_deriv(x):
            deriv_coeffs = jnp.arange(len(coeffs)-1, 0, -1) * coeffs[:-1]
            return jnp.polyval(deriv_coeffs, x)
        
        # NR iteration
        def body_fun(i, roots):
            fx = poly(roots)
            dfx = poly_deriv(roots)
            return roots - fx / dfx
        
        # Refine roots with NR iterations
        initialvals=jnp.abs(jnp.polyval(coeffs, initial_roots))
        refined_roots = jax.lax.fori_loop(0, max_iter, body_fun, initial_roots)
        finalvals=jnp.abs(jnp.polyval(coeffs, refined_roots))
        print("mean reduction",jnp.mean(finalvals/initialvals), jnp.max(finalvals/initialvals))
        return refined_roots


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    # Suppose user supplies these monomials/coefficients for eqn4:
    user_monomials = [
        [2,0,2,0],  # x0^2*y0^2
        [0,2,0,2],  # x1^2*y1^2
        [1,1,1,1],  # x0*x1 * y0*y1
    ]
    user_coeffs = [1+0j, 1+0j, 1+0j]
    

    solver = IntersectionSolverJAX_torus(monomials=user_monomials, coefficients=user_coeffs)
    key = jax.random.PRNGKey(42)

    # Solve a batch of 3 random c/d/e instances
    pointsA, pointsB, eqA_vals, eqB_vals  = solver.solve_batch(key, batch_size=3, return_residuals=True)
    print("Patch A eqn-values max:", np.max(np.abs(eqA_vals)))  # (3, degA, 4)
    print("Patch B eqn-values max:", np.max(np.abs(eqB_vals)))  # (3, degB, 4)

    # eqA_vals[i, j] => [eq1, eq2, eq3, eq4] for the j-th root of the i-th random system
    # We'll just show system0 root0:
    #resA = eqA_vals[0,0]  # shape (4,)
    #resB = eqB_vals[0,0]  # shape (4,)
    print("System0, root0 (PatchA) eqn residuals:", np.abs(eqA_vals[:,-1]))
    print("System0, root0 (PatchB) eqn residuals:", np.abs(eqB_vals[:,-1]))
    # Typically if all is correct, eq1..eqn4 should be ~ 1e-12 or smaller.