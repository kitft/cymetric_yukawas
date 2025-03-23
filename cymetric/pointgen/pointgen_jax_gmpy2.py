import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)
import numpy as np
import time
from functools import partial

# Disable JIT compilation
jax.config.update("jax_disable_jit", True)


class JAXPointGeneratorQuadratic:
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
        self.actual_point_generator = IntersectionSolverJAX( monomials=self.monomials, coefficients=self.coefficients, use_gmpy2=True, max_iter = max_iter, tol = tol)
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
        points = self.actual_point_generator.solve_batch(key, batch_size=(n_p//8+1), return_residuals=False)
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
        coords2 = coords[2:6]
        coords3 = coords[4:]

        pb_metric1 = self.pullback_fubini_study(coords1)
        pb_metric2 = self.pullback_fubini_study(coords2)
        pb_metric3 = self.pullback_fubini_study(coords3)

        pb_pb_metric1 = jnp.einsum('ai,ij,bj->ab', pullback_to_manifold[:,0:4], pb_metric1, jnp.conjugate(pullback_to_manifold[:,0:4]))
        pb_pb_metric2 = jnp.einsum('ai,ij,bj->ab', pullback_to_manifold[:,2:6], pb_metric2, jnp.conjugate(pullback_to_manifold[:,2:6]))
        pb_pb_metric3 = jnp.einsum('ai,ij,bj->ab', pullback_to_manifold[:,4:], pb_metric3, jnp.conjugate(pullback_to_manifold[:,4:]))
        return jnp.einsum('abc,def,ad,be,cf->', lc, lc, pb_pb_metric1, pb_pb_metric2, pb_pb_metric3)

   




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

class IntersectionSolverJAX:
    """
    Pure JAX solver for:
      eqn1: c_{ij} x_i y_j = 0
      eqn2: d_{ij} y_i z_j = 0
      eqn3: e_{ij} z_i w_j = 0
      eqn4: sum over user monomials in (x,y,z,w).
    Patches:
      A => x1=1 => univariate in x0
      B => x0=1 => univariate in x1
    Each solve is fully vectorized over 'batch_size' random instances
    (only c_{ij}, d_{ij}, e_{ij} are randomized).
    """

    

    def __init__(self, monomials=None, coefficients=None, use_gmpy2=True, max_iter = 10,tol = 1e-20):
        """
        monomials: List[8-int], shape=(N,8). Each row => exponents of
                   [x0,x1,y0,y1,z0,z1,w0,w1].
        coefficients: length-N, complex array => eqn4 monomial coefficients.

        We do the symbolic expansions once (using Sympy),
        then store numeric callables for patchA, patchB polynomials
        plus eqn1..eqn3 expansions for eqn-checks.
        """
        print("Compiling expressions in sympy.")
        # Default example if none provided
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
        z0, z1 = sympy.symbols('z0 z1', complex=True)
        w0, w1 = sympy.symbols('w0 w1', complex=True)

        c00, c01, c10, c11 = sympy.symbols('c00 c01 c10 c11', complex=True)
        d00, d01, d10, d11 = sympy.symbols('d00 d01 d10 d11', complex=True)
        e00, e01, e10, e11 = sympy.symbols('e00 e01 e10 e11', complex=True)

        # eqn1 => c_{ij} x_i y_j=0 => (y0,y1)=(b1,-a1)
        eqn1 = (c00*x0*y0 + c01*x0*y1
              + c10*x1*y0 + c11*x1*y1)
        eqn1_simpl = sympy.expand(eqn1)
        a1 = eqn1_simpl.diff(y0).subs({y0:0, y1:0})
        b1 = eqn1_simpl.diff(y1).subs({y0:0, y1:0})
        y0_expr = b1
        y1_expr = -a1

        # eqn2 => d_{ij} y_i z_j=0 => (z0,z1)=(b2,-a2)
        eqn2 = d00*y0*z0 + d01*y0*z1 + d10*y1*z0 + d11*y1*z1
        a2_raw = eqn2.diff(z0).subs({z0:0, z1:0})
        b2_raw = eqn2.diff(z1).subs({z0:0, z1:0})
        a2_sub = a2_raw.subs({y0:y0_expr, y1:y1_expr})
        b2_sub = b2_raw.subs({y0:y0_expr, y1:y1_expr})
        z0_expr = b2_sub
        z1_expr = -a2_sub

        # eqn3 => e_{ij} z_i w_j=0 => (w0,w1)=(b3,-a3)
        eqn3 = e00*z0*w0 + e01*z0*w1 + e10*z1*w0 + e11*z1*w1
        a3_raw = eqn3.diff(w0).subs({w0:0,w1:0})
        b3_raw = eqn3.diff(w1).subs({w0:0,w1:0})
        a3_sub = a3_raw.subs({z0:z0_expr, z1:z1_expr})
        b3_sub = b3_raw.subs({z0:z0_expr, z1:z1_expr})
        w0_expr = b3_sub
        w1_expr = -a3_sub

        # eqn4 => sum_{k} coefficients[k]* x0^... x1^... y0^... ...
        eqn4_sym = 0
        for cf, (ex0, ex1, ey0, ey1, ez0, ez1, ew0, ew1) in zip(coefficients, monomials):
            eqn4_sym += cf*( x0**ex0 * x1**ex1
                            *y0**ey0 *y1**ey1
                            *z0**ez0 *z1**ez1
                            *w0**ew0 *w1**ew1 )

        eqn4_sub = eqn4_sym.subs({
            y0: y0_expr, y1: y1_expr,
            z0: z0_expr, z1: z1_expr,
            w0: w0_expr, w1: w1_expr
        })
        eqn4_simpl = sympy.simplify(eqn4_sub)

        # Patch A => x1=1 => polynomial in x0
        eqn4A = eqn4_simpl.subs(x1, 1)
        polyA = sympy.poly(eqn4A, x0)
        self.polyA = polyA
        A_coeffs = polyA.all_coeffs()  # highest-degree first
        #A_coeffs_simpl = [sympy.simplify(coeff) for coeff in A_coeffs]

        # Patch B => x0=1 => polynomial in x1
        eqn4B = eqn4_simpl.subs(x0, 1)
        polyB = sympy.poly(eqn4B, x1)
        B_coeffs = polyB.all_coeffs()  # highest-degree first
        #B_coeffs_simpl = [sympy.simplify(coeff) for coeff in B_coeffs]

        # -------------------------------------------------
        # 2) Lambdify expansions for eqn1..eqn3 (for checks),
        #    plus patch polynomials
        # -------------------------------------------------
        from sympy.printing.numpy import JaxPrinter #or numpyprinter
        class CustomPrinter(JaxPrinter):
            def _print_Float(self, expr):
                return str(float(expr))

        from sympy.printing.numpy import NumPyPrinter #or numpyprinter
        class CustomPrinterNUMPY(NumPyPrinter):
            def _print_Float(self, expr):
                return str(float(expr))

        self.y0_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11), y0_expr, 'numpy')
        self.y1_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11), y1_expr, 'numpy')
        self.z0_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11, d00, d01, d10, d11),
                                     z0_expr, 'numpy')
        self.z1_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11, d00, d01, d10, d11),
                                     z1_expr, 'numpy')
        self.w0_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11,
                                      d00, d01, d10, d11,
                                      e00, e01, e10, e11),
                                     w0_expr,'numpy')
        self.w1_fun = sympy.lambdify((x0, x1, c00, c01, c10, c11,
                                      d00, d01, d10, d11,
                                      e00, e01, e10, e11),
                                     w1_expr,'numpy')

        # Build numeric patch polynomials:
        varlist = (c00,c01,c10,c11,
                   d00,d01,d10,d11,
                   e00,e01,e10,e11)  # no eqn4-coeff randomization => no extra symbols needed
        self.patchA_funs = [ jax.jit(sympy.lambdify(varlist, a,docstring_limit=0, cse=True,printer=CustomPrinter))
                             for a in A_coeffs ]
        self.patchB_funs = [ jax.jit(sympy.lambdify(varlist, b,docstring_limit=0, cse=True,printer=CustomPrinter))
                             for b in B_coeffs ]
        self.patchA_coeffs_fun = sympy.lambdify(
            (c00, c01, c10, c11,
             d00, d01, d10, d11,
             e00, e01, e10, e11),
            polyA.all_coeffs(),
            'numpy')

        self.patchA_coeffs_funNUMPY = sympy.lambdify(
            (c00, c01, c10, c11,
             d00, d01, d10, d11,
             e00, e01, e10, e11),
            polyA.all_coeffs(),
            printer=CustomPrinterNUMPY)

        self.patchA_coeffs_fun_JAXCUSTOM = sympy.lambdify(
            (c00, c01, c10, c11,
             d00, d01, d10, d11,
             e00, e01, e10, e11),
            polyB.all_coeffs(),
            printer=CustomPrinter)
        
        self.patchA_coeffs_fun_JAX = sympy.lambdify(
            (c00, c01, c10, c11,
             d00, d01, d10, d11,
             e00, e01, e10, e11),
            polyB.all_coeffs(),
            'jax')
        #self.patchA_poly = jax.jit(lambda x: sympy.lambdify(varlist, sympy.Matrix(A_coeffs_simpl), 'jax', docstring_limit=0, cse= True)(*x))
        #self.patchB_poly = jax.jit(lambda x: sympy.lambdify(varlist, sympy.Matrix(A_coeffs_simpl), 'jax', docstring_limit=0, cse= True)(*x))
        #self.polyA_batch_func = jax.vmap(self.patchA_poly)  # shape (batch_size, degA+1)
        #self.funcall_A_batch = jax.vmap(self.patchA_poly)
        self.funcall_A_batch = jax.vmap(lambda x: jnp.array(self.patchA_coeffs_fun(*x)))
        #self.polyB_batch_func = jax.vmap(self.patchB_poly)  # shape (batch_size, degB+1)
        self.funcall_B_batch = jax.vmap(self.patchB_poly)
        #print(A_coeffs_simpl)





        # Build patch A polynomials
        self.get_point_from_solution_batchA = jax.vmap(lambda x, cde: self.get_point_from_solution(x,1+0.j, cde), in_axes=(0,None))
        self.get_point_from_solution_batchA_nested = jax.vmap(self.get_point_from_solution_batchA, in_axes=(0,0))
        self.get_point_from_solution_batchB = jax.vmap(lambda x, cde: self.get_point_from_solution(1+0.j,x, cde), in_axes=(0,None))
        self.get_point_from_solution_batchB_nested = jax.vmap(self.get_point_from_solution_batchB, in_axes=(0,0))

        self.eqn_eval_batch = jax.vmap(self.eqn_eval, in_axes=(0,))
        self.eqn_123_eval_batch = jax.vmap(self.eqn_123_eval, in_axes=(0,None))# vmap over solution
        self.eqn_123_eval_batch_nested = jax.vmap(self.eqn_123_eval_batch, in_axes=(0,0))# vmap over batch 
        #self.roots_NR_vmap = jax.vmap(self.roots_NR, in_axes=(0,))

        # Evaluate eqn1..eqn4 => shape (batch_size, deg, 4).
        # We'll define a batched version of root-finding in the constructor
        #self._roots_poly_batch = jax.vmap(self._roots_of_poly, in_axes=0, out_axes=0)
        self.use_gmpy2 = use_gmpy2
        if use_gmpy2:
            self.max_iter = max_iter
            self.tol = tol
            self.roots_solver = gmpy2_poly_solver(max_iter = max_iter, tol = tol)
        else:
            self.roots_solver = self.roots_vmap

    # -------------------------------------------------
    # Private numeric: build companion, root it
    # # -------------------------------------------------
    # def _build_companion(self, poly_coeffs):
    #     """
    #     poly_coeffs: shape (deg+1,) leading coeff first
    #     returns shape (deg, deg) companion matrix
    #     """
    #     lead = poly_coeffs[0]
    #     norm = poly_coeffs / lead
    #     deg = poly_coeffs.shape[0] - 1
    #     C = jnp.zeros((deg, deg), dtype=jnp.complex128)
    #     # subdiag
    #     C = C.at[1:, :-1].set(jnp.eye(deg-1, dtype=jnp.complex128))
    #     # last row
    #     last_row = -norm[1:]
    #     C = C.at[-1, :].set(last_row[::-1])
    #     return C

    # def _roots_of_poly(self, poly_coeffs):
    #     mat = self._build_companion(poly_coeffs)
    # #     vals, _ = jnp.linalg.eig(mat)
    # #     return vals
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
    
    # Vectorized version
    # -------------------------------------------------
    # Build patch polynomials
    # -------------------------------------------------
    @jax.jit
    def patchA_poly(self, c):
        """
        c => (c00,c01,c10,c11),
        d => (d00,d01,d10,d11),
        e => ...
        But we combine them as a single array for convenience below.
        """
        # c00, c01, c10, c11 = c[0], c[1], c[2], c[3]
        # d00, d01, d10, d11 = c[4], c[5], c[6], c[7]
        # e00, e01, e10, e11 = c[8], c[9], c[10], c[11]
        
        # Pre-allocate array and use JAX's functional style to avoid Python loops
        def apply_fun(fun):
            return fun(*c   )
        
        # Map the function application across all functions in patchA_funs
        return jnp.stack(jax.tree.map(apply_fun, self.patchA_funs))

    @jax.jit
    def patchB_poly(self, c):
        # c00, c01, c10, c11 = c[0], c[1], c[2], c[3]
        # d00, d01, d10, d11 = c[4], c[5], c[6], c[7]
        # e00, e01, e10, e11 = c[8], c[9], c[10], c[11]
        
        # Pre-allocate array and use JAX's functional style to avoid Python loops
        def apply_fun(fun):
            return fun(*c)
        
        # Map the function application across all functions in patchB_funs
        return jnp.stack(jax.tree.map(apply_fun, self.patchB_funs))

    # -------------------------------------------------
    # Projective normalization


    # -------------------------------------------------
    # eqn1..eqn4 check
    # -------------------------------------------------
    @jax.jit
    def get_point_from_solution(self, x0, x1, c):
        """
        Evaluate eqn1..eqn4 with expansions from eqn1..eqn3 plus user monomials.
        c => 12-length array: c[0..3]=c00..c11, c[4..7]=d00..d11, c[8..11]=e00..e11.
        monomials => shape(N,8).
        coefficients => length N.
        Returns jnp.array([eq1, eq2, eq3, eq4]).
        """
        x0,x1= projective_normalize(x0,x1)
        c00, c01, c10, c11 = c[0], c[1], c[2], c[3]
        d00, d01, d10, d11 = c[4], c[5], c[6], c[7]
        e00, e01, e10, e11 = c[8], c[9], c[10], c[11]

        # eqn1
        y0v = self.y0_fun(x0, x1, c00, c01, c10, c11)
        y1v = self.y1_fun(x0, x1, c00, c01, c10, c11)
        ys = projective_normalize(y0v, y1v)
        # eqn2
        z0v = self.z0_fun(x0, x1, c00,c01,c10,c11, d00,d01,d10,d11)
        z1v = self.z1_fun(x0, x1, c00,c01,c10,c11, d00,d01,d10,d11)
        zs = projective_normalize(z0v, z1v)

        # eqn3
        w0v = self.w0_fun(x0,x1, c00,c01,c10,c11, d00,d01,d10,d11, e00,e01,e10,e11)
        w1v = self.w1_fun(x0,x1, c00,c01,c10,c11, d00,d01,d10,d11, e00,e01,e10,e11)
        ws = projective_normalize(w0v, w1v)

        return jnp.concatenate([jnp.array([x0, x1]), ys, zs, ws])

    def eqn_123_eval(self, point, cs):
        x0, x1, y0, y1, z0, z1, w0, w1 = point
        c00, c01, c10, c11, d00, d01, d10, d11, e00, e01, e10, e11 = cs
        eq1_val = c00*x0*y0 + c01*x0*y1 + c10*x1*y0 + c11*x1*y1
        eq2_val = d00*y0*z0 + d01*y0*z1 + d10*y1*z0 + d11*y1*z1
        eq3_val = e00*z0*w0 + e01*z0*w1 + e10*z1*w0 + e11*z1*w1
        return jnp.array([eq1_val, eq2_val, eq3_val])

    def eqn_eval(self, point):
        """
        Evaluate eqn1..eqn4 with expansions from eqn1..eqn3 plus user monomials.
        c => 12-length array: c[0..3]=c00..c11, c[4..7]=d00..d11, c[8..11]=e00..e11.
        monomials => shape(N,8).
        coefficients => length N.
        Returns jnp.array([eq1, eq2, eq3, eq4]).
        """
        x0, x1, y0, y1, z0, z1, w0, w1 = point
        # c = self.c
        # c00, c01, c10, c11 = c[0], c[1], c[2], c[3]
        # d00, d01, d10, d11 = c[4], c[5], c[6], c[7]
        # e00, e01, e10, e11 = c[8], c[9], c[10], c[11]
        
        # # Calculate equation values
        # eq1_val = c00*x0*y0 + c01*x0*y1 + c10*x1*y0 + c11*x1*y1
        # eq2_val = d00*y0*z0 + d01*y0*z1 + d10*y1*z0 + d11*y1*z1
        # eq3_val = e00*z0*w0 + e01*z0*w1 + e10*z1*w0 + e11*z1*w1
        
        # Vectorized computation of eq4_val using einsum
        powers = jnp.array([
            jnp.power(x0, self.monomials_jax[:, 0]),
            jnp.power(x1, self.monomials_jax[:, 1]),
            jnp.power(y0, self.monomials_jax[:, 2]),
            jnp.power(y1, self.monomials_jax[:, 3]),
            jnp.power(z0, self.monomials_jax[:, 4]),
            jnp.power(z1, self.monomials_jax[:, 5]),
            jnp.power(w0, self.monomials_jax[:, 6]),
            jnp.power(w1, self.monomials_jax[:, 7])
        ])
        
        # Multiply all powers together for each monomial
        monomial_values = jnp.prod(powers, axis=0)
        
        # Compute the weighted sum
        eq4_val = jnp.sum(self.coefficients_jax * monomial_values)

        return jnp.array(eq4_val, dtype=jnp.complex128)

    @jax.jit
    def generate_and_format_cde(self, key, batch_size):
        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 7)
        cs = jax.random.normal(subkey1, (batch_size, 4)) + 1j*jax.random.normal(subkey2, (batch_size, 4))
        cs /= jnp.sqrt(jnp.sum(jnp.abs(cs)**2, axis=1, keepdims=True))
        ds = jax.random.normal(subkey3, (batch_size, 4)) + 1j*jax.random.normal(subkey4, (batch_size, 4))
        ds /= jnp.sqrt(jnp.sum(jnp.abs(ds)**2, axis=1, keepdims=True))
        es = jax.random.normal(subkey5, (batch_size, 4)) + 1j*jax.random.normal(subkey6, (batch_size, 4))
        es /= jnp.sqrt(jnp.sum(jnp.abs(es)**2, axis=1, keepdims=True))
        return jnp.concatenate([cs, ds, es], axis=1)
        

    # -------------------------------------------------
    # solve_batch: random c/d/e, patch polynomials, projective eval
    # -------------------------------------------------
    #@partial(jax.jit, static_argnums=(1, 2))
    def solve_batch(self, key, batch_size=100000, return_residuals=False):
        """
        1) Generate random c_{ij}, d_{ij}, e_{ij} for each batch entry. (No eqn4 randomization.)
        2) Build patchA polynomials => solve via companion matrix => shape (batch_size, degA).
        3) Build patchB polynomials => shape (batch_size, degB).
        4) For each root in each patch, projectively normalize (just like original code),
           then evaluate eqn1..eqn4. Return eqA_vals, eqB_vals => shape
           (batch_size, degA, 4) and (batch_size, degB, 4).
        """
        cde_batch = self.generate_and_format_cde(key, batch_size)
        polyA_batch = self.funcall_A_batch(cde_batch)
        polyA_batch = polyA_batch/jnp.sqrt(jnp.sum(jnp.abs(polyA_batch)**2, axis=1, keepdims=True))
        if self.use_gmpy2:
            #print("solving using gmpy2")
            roots = self.roots_solver.solve_batch_of_polys(polyA_batch,  max_iter = self.max_iter, tol = self.tol)
            pointsA = self.get_point_from_solution_batchA_nested(np.array(roots), cde_batch).reshape(-1,8)
            if not return_residuals:
                return pointsA
            else:
                eqA_vals = self.eqn_eval_batch(pointsA.reshape(-1,8))
                eq123_Aevals = self.eqn_123_eval_batch_nested(pointsA, cde_batch)
                eqA_vals_reshaped = eqA_vals.reshape(-1, 8,1)
                Aevals = np.abs(jnp.concatenate([eqA_vals_reshaped, eq123_Aevals], axis=2))
                return pointsA, Aevals
        else:
            # shape => (batch_size, 12)
            polyB_batch = self.funcall_B_batch(cde_batch)
            polyB_batch = polyB_batch/jnp.sqrt(jnp.sum(jnp.abs(polyB_batch)**2, axis=1, keepdims=True))
            rootsA = self.roots_NR_vmap(polyA_batch)#self._roots_poly_batch(polyA_batch)      # shape (batch_size, degA)
            #for i in range(batch_size):
        
            #    rootsA.append(np.roots(np.array(polyA_batch[i])))
            ##rootsA = []
            rootsA = jnp.array(rootsA)

    

            pointsA = self.get_point_from_solution_batchA_nested(rootsA, cde_batch)
            rootsB = self.roots_NR_vmap(polyB_batch)#self._roots_poly_batch(polyB_batch)      # shape (batch_size, degB)
            #r#ootsB = jnp.roots(polyB_batch)
            #for i in range(batch_size):
            #    rootsB.append(np.roots(np.array(polyB_batch[i])))
            #
            rootsB = jnp.array(rootsB)
            pointsB = self.get_point_from_solution_batchB_nested(rootsB, cde_batch)
            if not return_residuals:
                return pointsA, pointsB
            else:
                eqA_vals = self.eqn_eval_batch(pointsA.reshape(-1,8))
                eqB_vals = self.eqn_eval_batch(pointsB.reshape(-1,8))
                eq123_Aevals = self.eqn_123_eval_batch_nested(pointsA, cde_batch)
                eq123_Bevals = self.eqn_123_eval_batch_nested(pointsB, cde_batch)
                # Reshape eqA_vals and eq123_Aevals to ensure they're 2D before concatenation
                eqA_vals_reshaped = eqA_vals.reshape(-1, 8,1)
                Aevals = np.abs(jnp.concatenate([eqA_vals_reshaped, eq123_Aevals], axis=2))

                # Do the same for B values
                eqB_vals_reshaped = eqB_vals.reshape(-1, 8,1)
                Bevals = np.abs(jnp.concatenate([eqB_vals_reshaped, eq123_Bevals], axis=2))
                return pointsA, pointsB, Aevals, Bevals

        

    # def merge_points(self, pointsA, pointsB):

    # def solve_batch_and_merge(self, key, batch_size=1, return_residuals=False):
    #     pointsA, pointsB = self.solve_batch(key, batch_size, return_residuals)
    #     return self.merge_points(pointsA, pointsB)

    #     # dex

        # # Patch B => (1, root)
        # def eval_patchB(b_idx, r_idx):
        #     raw_x1 = rootsB[b_idx, r_idx]
        #     x0_norm, x1_norm = projective_normalize(1+0j, raw_x1)
        #     return self.eqn_eval(x0_norm, x1_norm, cde_batch[b_idx])

        # degB = rootsB.shape[1]
        # root_indsB = jnp.arange(degB)
        # eqB_vals = jax.vmap(
        #     lambda b: jax.vmap(lambda r: eval_patchB(b, r))(root_indsB)
        # )(batch_inds)

        # eqA_vals => shape (batch_size, degA, 4), eqB_vals => shape (batch_size, degB, 4)
import numpy as np
from functools import partial


class gmpy2_poly_solver:
    def __init__(self, max_iter = 10, tol = 1e-20):
        try:
            from high_precision_roots import find_roots_vectorized
        except ImportError:
            import subprocess
            print("Building gmpy2 extension")
            subprocess.run(["python", "setup_high_precision.py", "build_ext", "--inplace"], check=True)
            from high_precision_roots import find_roots_vectorized
        self.find_roots_vectorized = find_roots_vectorized
        self.max_iter = max_iter
        self.tol = tol
        self.vmapped_roots = jax.vmap(lambda x: jnp.roots(x, strip_zeros=False), in_axes=(0))
    
    def _process_batch(self, coeffs_batch, max_iter=None, tol=1e-20):
        """Process a single batch of polynomials"""
        max_iter = max_iter or self.max_iter
        if max_iter==0:
            roots = self.vmapped_roots(jnp.array(coeffs_batch))
        else:
            roots = self.find_roots_vectorized(np.array(coeffs_batch), use_gmpy2=True, max_iter=max_iter, tol=tol)
        return roots
    
    def solve_batch_of_polys(self, coeffs,  max_iter=None, tol=1e-20):
        """Solve a batch of polynomials, optionally using multiprocessing"""
        #if not self.use_multiprocessing:
        return self._process_batch(coeffs, max_iter, tol)
        
        # # Split into sub-batches for multiprocessing
        # n_polys = len(coeffs)
        # if batch_size is None:
        #     batch_size = max(1, n_polys // self.n_processes)
        
        # batches = [coeffs[i:i+batch_size] for i in range(0, n_polys, batch_size)]
        
        # # Process batches in parallel
        # process_func = partial(self._process_batch, max_iter=max_iter, tol=tol)
        
        # with multiprocessing.Pool(self.n_processes) as pool:
        #     results = pool.map(process_func, batches)
        
        # # # Combine results
        # # all_roots = []
        # # for batch_result in results:
        # #     all_roots.extend(batch_result)
        
        # return all_roots


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    # Suppose user supplies these monomials/coefficients for eqn4:
    user_monomials = [
        [2,0,2,0,2,0,2,0],  # x0^2*y0^2*z0^2*w0^2
        [0,2,0,2,0,2,0,2],  # x1^2*y1^2*z1^2*w1^2
        [1,1,1,1,1,1,1,1],  # x0*x1 * y0*y1 * z0*z1 * w0*w1
    ]
    user_coeffs = [1+0j, 1+0j, 1+0j]
    

    solver = IntersectionSolverJAX(monomials=user_monomials, coefficients=user_coeffs)
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