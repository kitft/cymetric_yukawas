import jax
import jax.numpy as jnp
from functools import partial
from cymetric.pointgen.pointgen import PointGenerator
import numpy as np
class WP:
    def __init__(self, pg: PointGenerator, moduli_space_directions = None):
        self.pg = pg
        self.BASIS = pg.BASIS
        self.kmoduli = pg.kmoduli
        if moduli_space_directions is None:
            print('no moduli space directions provided, using default of unitvector(40,81)')
            psivector3 = jax.nn.one_hot(40, 81)
            pg.moduli_space_directions = np.array([psivector3])#+ [jax.nn.one_hot(i, 81) for i in range(20,30)])
        else:
            pg.moduli_space_directions = moduli_space_directions
        pg.get_moduli_space_metric = True
        pg._generate_dQdz_basis()
        pg._generate_padded_dIdQZ_basis()
        pg._generate_moduli_space_basis()
        print("warning - using a hack for the inverse fubini study metric for P1^n manifolds. Easily remedied.")

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def inverse_fubini_study_metrics_jax(points, vol_js,hack = False):
        r"""Computes the FS metric at each point.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factor.
                Defaults to None.

        Returns:
            ndarray[(n_p, ncoords, ncoords), np.complex128]: g^FS
        """
        if hack:
            #gFS = jnp.zeros((len(points),8),
            #           dtype=jnp.complex128)
            gFS = jnp.concatenate([jnp.sum(jnp.abs(points[:,2*s:2*s+2])**2,axis = -1,keepdims = True)**2 for s in range(4)],axis = 1).repeat(2,axis = 1)
            #print(gFS)
            
            return gFS
                       
        gFS = jnp.zeros((len(points),8,8),
                       dtype=jnp.complex128)
        for i in range(4):
            s = 2*i
            e = 2*i + 2
            points_slice = points[:, s:e]
            invfs_metric = WP.inverse_fubini_study_n_metrics_jax(points_slice, (vol_js[i]))
            gFS = gFS.at[:, s:e, s:e].add(invfs_metric)
        #print(gFS)
        return gFS
    @staticmethod
    @jax.jit
    def inverse_fubini_study_n_metrics_jax(points, vol_j=1. + 0.j):
        r"""Computes the FS metric for a single projective space of points.

        Args:
            point (ndarray[(n_p, n), np.complex128]): Points.
            vol_j (complex): Volume factor. Defaults to 1+0.j.

        Returns:
            ndarray[(n_p, n, n), np.complex128]: g^FS
        """
        # we want to compute d_i d_j K^FS
        point_square = jnp.add.reduce(jnp.abs(points) ** 2, axis=-1)
        outer = jnp.einsum('xi,xj->xij', jnp.conj(points), points)
        gFS = jnp.einsum('x,ij->xij', point_square**0, jnp.eye(points.shape[1]))
        gFS = gFS.astype(jnp.complex128) + outer
        return jnp.einsum('xij,x->xij', gFS, (point_square)) * (1/vol_j)# * np.pi# has indices nubar mu

    @staticmethod
    @jax.jit
    def _compute_H_poly(points, DQDZB0, DQDZF0, kmoduli,good_ambient_coords):
        dq_dz = jnp.power(jnp.expand_dims(points, (1,2)),jnp.expand_dims(DQDZB0, 0))
        dq_dz = jnp.multiply.reduce(dq_dz, axis=-1)
        dq_dz = jnp.add.reduce(jnp.expand_dims(DQDZF0, 0) * dq_dz, axis=-1)
        #inverse_fubini_study  = jnp.conjugate(inverse_fubini_study_metrics_jax(points, kmoduli))# indices mu nubar
        inverse_fubini_study  = WP.inverse_fubini_study_metrics_jax(points, kmoduli,hack = True)# indices mu nubar
        dqbar_dzbar = jnp.conj(dq_dz)
        good_ambient_coords = jnp.array(good_ambient_coords,dtype = float)
       
        #H = jnp.einsum('xb,xb,xab,xa,xa->x',dqbar_dzbar, good_ambient_coords,inverse_fubini_study, good_ambient_coords,dq_dz)
        H = jnp.einsum('xa,xa,xa,xa,xa->x',dqbar_dzbar, good_ambient_coords,inverse_fubini_study, good_ambient_coords,dq_dz)
        return H, inverse_fubini_study,dq_dz,dqbar_dzbar

    @staticmethod
    @jax.jit
    def _compute_vecbeforep(points, DQDZB0, DQDZF0, kmoduli, good_ambient_coords):
        H, inverse_fubini_study,dq_dz,dqbar_dzbar = WP._compute_H_poly(points, DQDZB0, DQDZF0, kmoduli,good_ambient_coords)
       
        #vecbefore_Ip = -1* jnp.einsum('x,xiJ,xJ,xJ->xi',H**(-1),inverse_fubini_study, good_ambient_coords,dqbar_dzbar)
        vecbefore_Ip = -1* jnp.einsum('x,xJ,xJ,xJ->xJ',H**(-1),inverse_fubini_study, good_ambient_coords,dqbar_dzbar)
        vecbefore_Ip = WP.proj_mask(vecbefore_Ip, good_ambient_coords,axis=-1, ngood=4)
        return vecbefore_Ip

    @staticmethod
    @jax.jit
    def _compute_vecafterp_r_r(realpoints, DQDZB0, DQDZF0, kmoduli, good_ambient_coords):
        
        complexpoints = realpoints[:,:8] + 1j*realpoints[:,8:]
        vecbefore_Ip = WP._compute_vecbeforep(complexpoints, DQDZB0, DQDZF0, kmoduli, good_ambient_coords)
        return jnp.stack([jnp.real(vecbefore_Ip), jnp.imag(vecbefore_Ip)], axis=-1)
        
    @staticmethod
    @jax.jit
    def _compute_del_and_delbar_vecbefore_p(points, DQDZB0, DQDZF0, kmoduli, good_ambient_coords):
        """Compute both the value and gradient of _compute_vecafterp_r_r.
        
        Args:
            realpoints: Real representation of points (batch_size, 16)
            DQDZB0, DQDZF0: Basis for derivatives
            kmoduli: Kähler moduli
            proj_to_good_coords: boolean mask of good coordinates
        Returns:
            value: The output of _compute_vecafterp_r_r
            gradient: The Jacobian with respect to realpoints
        """
        # Since _compute_vecafterp_r_r returns a vector output, we need to use jacfwd
        # to get the full Jacobian matrix rather than value_and_grad
        real_points = jnp.concatenate([jnp.real(points),jnp.imag(points)],axis=-1)
        value = WP._compute_vecafterp_r_r(real_points, DQDZB0, DQDZF0, kmoduli, good_ambient_coords)
        complexvalue = value[...,0] + 1j*value[...,1]
        
        # Define the function to differentiate (real input, real output)
        def vecafterp_r_r_fn(real_pts, coordsmask):
            return WP._compute_vecafterp_r_r(real_pts, DQDZB0, DQDZF0, kmoduli, coordsmask)

        # Compute the batch Jacobian using JAX
        jacobian = jax.vmap(jax.jacfwd(lambda x, y: vecafterp_r_r_fn(jnp.array([x]), jnp.array([y]))[0]), in_axes = (0,0))(real_points, good_ambient_coords)  # shape is batch, output_dims, input_dims
        
        # Reconstruct holomorphic and anti-holomorphic components
        # For a function f(z,z̄) with z = x + iy, the holomorphic derivative is:
        # ∂f/∂z = (1/2)(∂f/∂x - i∂f/∂y)
        # And the anti-holomorphic derivative is:
        # ∂f/∂z̄ = (1/2)(∂f/∂x + i∂f/∂y)
        
        # Split the jacobian into parts for the first 8 (real) and last 8 (imaginary) components
        # Assuming jacobian shape is (batch, output_dim, input_dim)
        jac_dx_Vx = jacobian[...,0, :8] # jacobian w.r.t. real parts
        jac_dx_Vy = jacobian[...,1, :8]  # jacobian w.r.t. real parts
        jac_dy_Vx = jacobian[...,0, 8:] # jacobian w.r.t. imaginary parts
        jac_dy_Vy = jacobian[...,1, 8:] # jacobian w.r.t. imaginary parts  
        
        # Reconstruct holomorphic and anti-holomorphic components
        holo_derivative = 0.5 * (jac_dx_Vx - 1j * jac_dy_Vx) + 0.5 *1j * (jac_dx_Vy - 1j * jac_dy_Vy)
        antiholo_derivative = 0.5 * (jac_dx_Vx + 1j * jac_dy_Vx) + 0.5 *1j * (jac_dx_Vy + 1j * jac_dy_Vy)
        
        return complexvalue, holo_derivative, antiholo_derivative

    @staticmethod
    @partial(jax.jit, static_argnums=(2,3))
    def proj_mask(jacobian, proj_to_good_coords, axis=-1, ngood=3):
        """Select components along the specified axis using a boolean mask.
        
        If the mask is 1D, use its nonzero indices directly.
        If it is 2D (batched), gather indices per batch and select along the axis.
        """
        if proj_to_good_coords.ndim == 1:
            idx = jnp.nonzero(proj_to_good_coords, size = ngood)[0]
            return jnp.take(jacobian, idx, axis=axis)
        else:
            # Assume proj_to_good_coords shape is (batch, mask_length)
            idx = jax.vmap(lambda m: jnp.nonzero(m, size = ngood)[0])(proj_to_good_coords)
            idx = idx[:, :ngood]  # crop if more indices exist
            if axis != -1:
                jacobian = jnp.moveaxis(jacobian, axis, -1)
            extra_dims = (1,) * (jacobian.ndim - 2)
            idx = idx.reshape(idx.shape[0], *extra_dims, idx.shape[1])
            selected = jnp.take_along_axis(jacobian, idx, axis=-1)
            return selected if axis == -1 else jnp.moveaxis(selected, -1, axis)

    @staticmethod
    @jax.jit
    def _compute_dI_dQZ_jax_multiple_indices(points, DI_DQZB0, DI_DQZF0):
        """Compute derivatives of I,z with respect to Q"""
        # Expand points to match DI_DQZB0 shape which is (81, 8, 54, 8), where 81 can just be the number of directions under consideration
        expanded_points = jnp.expand_dims(jnp.expand_dims(points, 0), (2,3))# shape 1,N,1,1, 8
        dI_dQZ = jnp.power(expanded_points, jnp.expand_dims(DI_DQZB0, 1))#1, N,1, 1,8 and 81,N,8,54,8
        dI_dQZ = jnp.multiply.reduce(dI_dQZ, axis=-1)
        dI_dQZ = jnp.add.reduce(jnp.expand_dims(DI_DQZF0, 1) * dI_dQZ, axis=-1)
        return dI_dQZ
        
    @staticmethod
    @jax.jit
    def dholo_antiholo_theta(points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0,DI_DQZB0, DI_DQZF0, good_cy_coords, good_ambient_coords, cy_in_amb, pullbacks):
        complexvalue, holo_derivative, antiholo_derivative = WP._compute_del_and_delbar_vecbefore_p(points, DQDZB0, DQDZF0, kmoduli, good_ambient_coords)

        #complexvalue = _compute_vecbeforep(points, DQDZB0, DQDZF0, kmoduli, good_ambient_coords)

        dI_dQZ = WP._compute_dI_dQZ_jax_multiple_indices(points, DI_DQZB0, DI_DQZF0)#automatically pulled back!
        dI_p = PointGenerator._compute_dIp_jax(points, DI_DQB0, DI_DQF0)
        thetai_holo= jnp.einsum('xAb,Ix->xIAb',holo_derivative, dI_p) + jnp.einsum('xA,Ixb->xIAb',complexvalue, dI_dQZ)

        pullbackantiholoderivative = jnp.einsum('xmb,xAb->xAm',jnp.conjugate(pullbacks),antiholo_derivative) #then pull back the form
        pullbackantiholoderivative=WP.proj_mask(pullbackantiholoderivative, cy_in_amb, axis=1, ngood=3)# put vector into good cy coordinates

        iproduct = jnp.array([-1,1,-1], dtype=antiholo_derivative.dtype)
        gi_kantiholo_holoup = jnp.einsum('A,xAm,Ix->xIAm',iproduct,pullbackantiholoderivative, dI_p)#

        proj_to_inhom_ambient_thetaiholo = WP.proj_mask(thetai_holo, good_ambient_coords, axis=-1,ngood=4)
        trace_holo = -jnp.einsum('xIAA->xI',proj_to_inhom_ambient_thetaiholo)

        return gi_kantiholo_holoup, trace_holo

    @staticmethod
    @jax.jit
    def dnu_dI(points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, pullbacks, indices):
        good_ambient_coords = jnp.logical_not(jnp.isclose(points, jax.lax.complex(1.,0.),atol = 1e-8))
        good_cy_coords = jnp.copy(good_ambient_coords)
        cy_in_amb_indices =jnp.squeeze(indices)//2
        cy_in_amb = jnp.ones((indices.shape[0],4), dtype = bool)
        cy_in_amb = cy_in_amb.at[jnp.arange(indices.shape[0]), cy_in_amb_indices].set(False)
        good_cy_coords = good_cy_coords.at[jnp.arange(indices.shape[0]), indices].set(False)
        
        # Set proj_to_good_coords to False at positions specified by indices

        gi_kantiholo_holoup, trace_holo = WP.dholo_antiholo_theta(points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0,DI_DQZB0, DI_DQZF0, good_cy_coords, good_ambient_coords, cy_in_amb, pullbacks)
        #dnu_dI_holobit = trace_holo

        return gi_kantiholo_holoup, trace_holo

    @partial(jax.jit, static_argnums=(11))
    def gwP_jax(points, weights, pullbacks, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, indices = None, fullmatrix = False):
        """
        Calculates gWP and its standard error using standard numpy/jax operations.

        Args:
            points: Sample points.
            weights: Weights for each sample point (should represent vol_form / total_vol_form).
            pullbacks: Pullback metrics for each point.
            DQDZB0, DQDZF0: Basis components.
            kmoduli: Kahler moduli.
            DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0: Basis derivatives.
            indices: Indices for coordinate selection. If None, calculated dynamically.
            fullmatrix: Whether to compute the full gWP matrix or just the diagonal.

        Returns:
            A tuple containing:
                - gWP: The calculated Weil-Petersson metric (mean).
                - gWP_std_error: The standard error of the mean for gWP.
        """
        if indices is None:
            # Note: Dynamic calculation of indices might hinder JIT compilation if called without indices.
            # Consider pre-calculating indices outside if performance is critical.
            # This class method access might not work correctly inside a static method if PointGenerator isn't defined statically.
            # Assuming PointGenerator._find_max_dQ_coords_jax is accessible statically or refactored.
            # For now, let's assume it works or indices are always provided when jitted.
            # If PointGenerator is the class containing this method, self. would be needed if not static.
            # If it's a different class, PointGenerator._find_max_dQ_coords_jax might be okay if static.
            # Reverting to WP._find_max_dQ_coords_jax assuming it's defined within WP or accessible.
            # If _find_max_dQ_coords_jax is instance method, this needs refactoring.
            # Let's assume it's a static method for now.
            try:
                 # If WP is the class name
                 indices = WP._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0)
            except AttributeError:
                 # If PointGenerator is the intended class and has the static method
                 # This requires PointGenerator to be imported or accessible.
                 # from .pointgen import PointGenerator # Example import
                 # indices = PointGenerator._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0)
                 # For now, raise error if indices are needed but calculation fails
                 raise ValueError("Indices must be provided or _find_max_dQ_coords_jax must be statically accessible.")


        # Calculate the mean volume form (denominator for normalization)
        vol_omega = jnp.mean(weights)
        safe_vol_omega = jnp.where(vol_omega == 0, 1e-15, vol_omega) # Avoid division by zero

        # Calculate intermediate quantities needed for gWP
        dnu_dI_antiholobit, trace_holo = WP.dnu_dI(points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, pullbacks, indices)

        # Calculate per-sample values for int_omega_dIomega (numerator part)
        per_sample_int_omega_dIomega_num = jnp.einsum('x,xI->xI', weights, trace_holo)
        # Calculate the mean and standard error of the mean for int_omega_dIomega
        int_omega_dIomega = jnp.mean(per_sample_int_omega_dIomega_num, axis=0)
        n_samples = per_sample_int_omega_dIomega_num.shape[0]
        safe_n_sqrt = jnp.sqrt(jnp.maximum(1.0, n_samples)) # Avoid division by zero if n_samples=0

        int_omega_dIomega_real_std_error = jnp.std(jnp.real(per_sample_int_omega_dIomega_num), axis=0, ddof=0) / safe_n_sqrt # Use ddof=0 for population std used in SEM
        int_omega_dIomega_imag_std_error = jnp.std(jnp.imag(per_sample_int_omega_dIomega_num), axis=0, ddof=0) / safe_n_sqrt
        # Note: int_omega_dIomega_std_error is the error of the *numerator* mean

        # Precompute wedge product matrix
        wedge_the_two = jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)], dtype=points.dtype)

        # Calculate std error of vol_omega (mean of weights)
        vol_omega_std_error = jnp.std(weights, ddof=0) / safe_n_sqrt

        # Define common variables for error propagation
        Br = safe_vol_omega
        dBr = vol_omega_std_error
        Cr = jnp.real(int_omega_dIomega) # Mean of numerator
        Ci = jnp.imag(int_omega_dIomega) # Mean of numerator
        dCr = int_omega_dIomega_real_std_error # SEM of numerator mean
        dCi = int_omega_dIomega_imag_std_error # SEM of numerator mean

        if not fullmatrix:
            # Calculate numerators for gWP terms per sample (diagonal case)
            per_sample_term1_num = jnp.einsum('x,xI,xI->xI', weights, trace_holo, jnp.conjugate(trace_holo))
            per_sample_term2_num = jnp.einsum('x,ij,xIij,xIji->xI', weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))

            # Calculate the per-sample numerator of gWP_part1
            per_sample_gWP_part1_num = -(per_sample_term1_num + per_sample_term2_num)

            # Calculate mean and standard error of the mean for the gWP_part1 numerator
            gWP_part1_num_mean = jnp.mean(per_sample_gWP_part1_num, axis=0)
            gWP_part1_num_real_std_error = jnp.std(jnp.real(per_sample_gWP_part1_num), axis=0, ddof=0) / safe_n_sqrt
            gWP_part1_num_imag_std_error = jnp.std(jnp.imag(per_sample_gWP_part1_num), axis=0, ddof=0) / safe_n_sqrt

            # Calculate gWP_part1 by dividing the numerator mean by the mean volume
            gWP_part1 = gWP_part1_num_mean / safe_vol_omega

            # Error propagation for gWP_part1 = A / B, where A = gWP_part1_num_mean, B = vol_omega
            Ar = jnp.real(gWP_part1_num_mean)
            Ai = jnp.imag(gWP_part1_num_mean)
            dAr = gWP_part1_num_real_std_error
            dAi = gWP_part1_num_imag_std_error

            # Variance = (dAr/Br)^2 + (Ar*dBr/Br^2)^2 - assuming independence of A and B errors
            # More correctly, Var(A/B) ≈ (mean(A)/mean(B))^2 * [ (std(A)/mean(A))^2 + (std(B)/mean(B))^2 - 2*cov(A,B)/(mean(A)mean(B)) ]
            # Using simpler propagation: Var(f) ≈ (df/dA)^2 Var(A) + (df/dB)^2 Var(B)
            # Var(A) = (dAr * sqrt(N))^2, Var(B) = (dBr * sqrt(N))^2
            # Var(gWP_part1_real) = (1/Br)^2 * Var(Ar) + (-Ar/Br^2)^2 * Var(Br)
            # Var(gWP_part1_real) = (1/Br)^2 * dAr^2 * N + (Ar^2/Br^4) * dBr^2 * N
            # SEM^2(gWP_part1_real) = Var(gWP_part1_real) / N = (dAr/Br)^2 + (Ar*dBr/Br^2)^2
            gWP_part1_var_real = (dAr/Br)**2 + (Ar*dBr/Br**2)**2
            gWP_part1_var_imag = (dAi/Br)**2 + (Ai*dBr/Br**2)**2
            gWP_part1_real_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_real))
            gWP_part1_imag_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_imag))

            # Calculate gWP_part2 = <term3_num> / <vol>^2 = C*C* / B^2
            gWP_part2 = (int_omega_dIomega * jnp.conjugate(int_omega_dIomega)) / safe_vol_omega**2 # Real result

            # Error propagation for gWP_part2 = (Cr^2 + Ci^2) / Br^2
            # Var(gWP_part2) ≈ (d(gWP2)/dCr)^2 Var(Cr) + (d(gWP2)/dCi)^2 Var(Ci) + (d(gWP2)/dBr)^2 Var(Br)
            # SEM^2(gWP_part2) ≈ (d(gWP2)/dCr)^2 SEM(Cr)^2 + (d(gWP2)/dCi)^2 SEM(Ci)^2 + (d(gWP2)/dBr)^2 SEM(Br)^2
            # d(gWP2)/dCr = 2*Cr / Br^2
            # d(gWP2)/dCi = 2*Ci / Br^2
            # d(gWP2)/dBr = -2*(Cr^2 + Ci^2) / Br^3 = -2 * Real(gWP_part2) / Br
            term_Cr_sq = ( (2 * Cr / Br**2) * dCr )**2
            term_Ci_sq = ( (2 * Ci / Br**2) * dCi )**2
            term_Br_sq = ( (-2 * jnp.real(gWP_part2) / Br) * dBr )**2

            gWP_part2_var = term_Cr_sq + term_Ci_sq + term_Br_sq
            gWP_part2_real_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var))
            gWP_part2_imag_std_error = jnp.zeros_like(gWP_part2_real_std_error) # gWP_part2 is real

            # Combine the parts
            gWP = gWP_part1 + jnp.real(gWP_part2) # Ensure real addition

            # Add errors in quadrature for real and imaginary parts separately
            gWP_real_std_error = jnp.sqrt(gWP_part1_real_std_error**2 + gWP_part2_real_std_error**2)
            gWP_imag_std_error = gWP_part1_imag_std_error # gWP_part2_imag_std_error is 0

        else: # fullmatrix = True
            # Calculate numerators for gWP terms per sample (full matrix case)
            per_sample_term1_num = jnp.einsum('x,xI,xJ->xIJ', weights, trace_holo, jnp.conjugate(trace_holo))
            per_sample_term2_num = jnp.einsum('x,ij,xIij,xJji->xIJ', weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))

            # Calculate the per-sample numerator of gWP_part1
            per_sample_gWP_part1_num = -(per_sample_term1_num + per_sample_term2_num)

            # Calculate mean and standard error of the mean for the gWP_part1 numerator
            gWP_part1_num_mean = jnp.mean(per_sample_gWP_part1_num, axis=0)
            gWP_part1_num_real_std_error = jnp.std(jnp.real(per_sample_gWP_part1_num), axis=0, ddof=0) / safe_n_sqrt
            gWP_part1_num_imag_std_error = jnp.std(jnp.imag(per_sample_gWP_part1_num), axis=0, ddof=0) / safe_n_sqrt

            # Calculate gWP_part1 by dividing the numerator mean by the mean volume
            gWP_part1 = gWP_part1_num_mean / safe_vol_omega

            # Error propagation for gWP_part1 = A / B
            Ar = jnp.real(gWP_part1_num_mean)
            Ai = jnp.imag(gWP_part1_num_mean)
            dAr = gWP_part1_num_real_std_error
            dAi = gWP_part1_num_imag_std_error

            gWP_part1_var_real = (dAr/Br)**2 + (Ar*dBr/Br**2)**2
            gWP_part1_var_imag = (dAi/Br)**2 + (Ai*dBr/Br**2)**2
            gWP_part1_real_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_real))
            gWP_part1_imag_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_imag))

            # Calculate gWP_part2 = <term3_num> / <vol>^2 = C_I * C_J* / B^2
            gWP_part2 = jnp.einsum('I,J->IJ', int_omega_dIomega, jnp.conjugate(int_omega_dIomega)) / safe_vol_omega**2

            # Error propagation for gWP_part2 = C_I * C_J* / B^2
            # SEM^2(Real(gWP2_IJ)) ≈ sum_K [(dReal/dCKr*dCKr)^2 + (dReal/dCKi*dCKi)^2] + (dReal/dBr*dBr)^2
            # SEM^2(Imag(gWP2_IJ)) ≈ sum_K [(dImag/dCKr*dCKr)^2 + (dImag/dCKi*dCKi)^2] + (dImag/dBr*dBr)^2
            # Where K runs over variables C_I_real, C_I_imag, C_J_real, C_J_imag, B_real

            # Partial derivatives (evaluated at mean values)
            # dReal/dCrI = CrJ / Br^2
            # dReal/dCiI = CiJ / Br^2
            # dReal/dCrJ = CrI / Br^2
            # dReal/dCiJ = CiI / Br^2
            # dReal/dBr = -2 * Real(gWP2_IJ) / Br
            # dImag/dCrI = -CiJ / Br^2
            # dImag/dCiI = CrJ / Br^2
            # dImag/dCrJ = CiI / Br^2
            # dImag/dCiJ = -CrI / Br^2
            # dImag/dBr = -2 * Imag(gWP2_IJ) / Br

            # Variance calculation (SEM squared)
            # Term involving dBr
            term_Br_real_sq = ( (-2 * jnp.real(gWP_part2) / Br) * dBr )**2
            term_Br_imag_sq = ( (-2 * jnp.imag(gWP_part2) / Br) * dBr )**2

            # Terms involving dCrI, dCiI (derivative w.r.t. first index I)
            # (dReal/dCrI * dCrI)^2 + (dReal/dCiI * dCiI)^2
            term_I_real_sq = jnp.einsum('J,I->IJ', Cr**2 / Br**4, dCr**2) + jnp.einsum('J,I->IJ', Ci**2 / Br**4, dCi**2)
            # (dImag/dCrI * dCrI)^2 + (dImag/dCiI * dCiI)^2
            term_I_imag_sq = jnp.einsum('J,I->IJ', Ci**2 / Br**4, dCr**2) + jnp.einsum('J,I->IJ', Cr**2 / Br**4, dCi**2)

            # Terms involving dCrJ, dCiJ (derivative w.r.t. second index J)
            # (dReal/dCrJ * dCrJ)^2 + (dReal/dCiJ * dCiJ)^2
            term_J_real_sq = jnp.einsum('I,J->IJ', Cr**2 / Br**4, dCr**2) + jnp.einsum('I,J->IJ', Ci**2 / Br**4, dCi**2)
            # (dImag/dCrJ * dCrJ)^2 + (dImag/dCiJ * dCiJ)^2
            term_J_imag_sq = jnp.einsum('I,J->IJ', Ci**2 / Br**4, dCr**2) + jnp.einsum('I,J->IJ', Cr**2 / Br**4, dCi**2)

            # Total variance (SEM squared)
            gWP_part2_var_real = term_I_real_sq + term_J_real_sq + term_Br_real_sq
            gWP_part2_var_imag = term_I_imag_sq + term_J_imag_sq + term_Br_imag_sq

            # Diagonal elements need special handling (I=J) - the above double counts derivatives
            # For I=J: Real = (CrI^2 + CiI^2)/Br^2, Imag = 0
            # SEM^2(Real) ≈ (dReal/dCrI*dCrI)^2 + (dReal/dCiI*dCiI)^2 + (dReal/dBr*dBr)^2
            diag_idx = jnp.diag_indices_from(gWP_part2)
            diag_Cr = Cr[diag_idx[0]]
            diag_Ci = Ci[diag_idx[0]]
            diag_dCr = dCr[diag_idx[0]]
            diag_dCi = dCi[diag_idx[0]]
            diag_gWP_part2_real = jnp.real(gWP_part2[diag_idx])

            diag_term_Cr_sq = ( (2 * diag_Cr / Br**2) * diag_dCr )**2
            diag_term_Ci_sq = ( (2 * diag_Ci / Br**2) * diag_dCi )**2
            diag_term_Br_sq = ( (-2 * diag_gWP_part2_real / Br) * dBr )**2
            diag_gWP_part2_var_real = diag_term_Cr_sq + diag_term_Ci_sq + diag_term_Br_sq
            diag_gWP_part2_var_imag = jnp.zeros_like(diag_gWP_part2_var_real)

            # Update diagonal elements of the variance matrices
            gWP_part2_var_real = gWP_part2_var_real.at[diag_idx].set(diag_gWP_part2_var_real)
            gWP_part2_var_imag = gWP_part2_var_imag.at[diag_idx].set(diag_gWP_part2_var_imag) # Should remain zero

            # Standard error (sqrt of SEM squared)
            gWP_part2_real_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var_real))
            gWP_part2_imag_std_error = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var_imag))

            # Combine the parts
            gWP = gWP_part1 + gWP_part2

            # Add errors in quadrature for real and imaginary parts separately
            gWP_real_std_error = jnp.sqrt(gWP_part1_real_std_error**2 + gWP_part2_real_std_error**2)
            gWP_imag_std_error = jnp.sqrt(gWP_part1_imag_std_error**2 + gWP_part2_imag_std_error**2)

        # Combine into complex std error
        gWP_std_error = gWP_real_std_error + 1j * gWP_imag_std_error

        return gWP, gWP_std_error

    @staticmethod
    @partial(jax.jit, static_argnums=(11, 12, 14)) # nmoduli, fullmatrix, batch_size are static
    def gwP_jax_batched(points, weights, pullbacks, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, indices = None, fullmatrix = False, batch_size = 1000, num_batches = None, nmoduli = 1):
        """
        Calculates gWP and its standard error using online Welford algorithm for batches
        to avoid memory issues with large datasets. Uses jax.lax.fori_loop for better JIT compilation.
        Normalization strategy matches gwP_jax (divide by final mean volume at the end).

        Args:
            points: Sample points. Shape (total_points, ...).
            weights: Weights for each sample point. Shape (total_points,).
            pullbacks: Pullback metrics for each point. Shape (total_points, ...).
            DQDZB0, DQDZF0: Basis components.
            kmoduli: Kahler moduli.
            DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0: Basis derivatives.
            indices: Indices for coordinate selection. Shape (total_points, ...). Required.
            fullmatrix: Whether to compute the full gWP matrix or just the diagonal. Must be static.
            batch_size: Number of points to process in each batch. Must be static for JIT.
            num_batches: Total number of batches (optional, calculated if None).
            nmoduli: Number of moduli (dimension). Must be static for JIT.

        Returns:
            A tuple containing:
                - gWP: The calculated Weil-Petersson metric (mean).
                - gWP_std_error: The standard error of the mean for gWP.
        """
        if indices is None:
             raise ValueError("Indices must be provided for batched computation.")

        total_points = points.shape[0]
        # Calculate number of batches needed. This value will be used by fori_loop.
        num_batches_calc = (total_points + batch_size - 1) // batch_size

        # Determine the shape for gWP part 1 numerator based on fullmatrix flag
        gWP_part1_num_shape = (nmoduli, nmoduli) if fullmatrix else (nmoduli,)

        # Initialize accumulators for online mean and variance calculation (Welford's algorithm)
        # State tuple for fori_loop: Accumulate numerators for gWP part 1
        initial_state = (
            jnp.zeros((), dtype=jnp.int64), # n: Total samples processed
            jnp.zeros((), dtype=jnp.float64), # vol_omega_mean
            jnp.zeros((), dtype=jnp.float64), # vol_omega_M2
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_num_mean_real
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_num_mean_imag
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_num_M2_real
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_num_M2_imag
            jnp.zeros(gWP_part1_num_shape, dtype=jnp.float64), # gWP_part1_num_mean_real
            jnp.zeros(gWP_part1_num_shape, dtype=jnp.float64), # gWP_part1_num_mean_imag
            jnp.zeros(gWP_part1_num_shape, dtype=jnp.float64), # gWP_part1_num_M2_real
            jnp.zeros(gWP_part1_num_shape, dtype=jnp.float64)  # gWP_part1_num_M2_imag
        )

        # Define the loop body function for jax.lax.fori_loop
        def loop_body(i, current_state):
            # Unpack current state
            n, vol_omega_mean, vol_omega_M2, \
            int_omega_dIomega_num_mean_real, int_omega_dIomega_num_mean_imag, \
            int_omega_dIomega_num_M2_real, int_omega_dIomega_num_M2_imag, \
            gWP_part1_num_mean_real, gWP_part1_num_mean_imag, \
            gWP_part1_num_M2_real, gWP_part1_num_M2_imag = current_state

            # Calculate batch indices
            start_idx = i * batch_size
            current_batch_size = jnp.minimum(batch_size, total_points - start_idx)

            # Slice batch data using dynamic slices
            batch_points = jax.lax.dynamic_slice_in_dim(points, start_idx, batch_size, axis=0)
            batch_weights = jax.lax.dynamic_slice_in_dim(weights, start_idx, batch_size, axis=0).astype(jnp.float64)
            batch_pullbacks = jax.lax.dynamic_slice_in_dim(pullbacks, start_idx, batch_size, axis=0)
            batch_indices = jax.lax.dynamic_slice_in_dim(indices, start_idx, batch_size, axis=0)

            # Create a mask for valid entries in the current batch
            valid_mask = jnp.arange(batch_size) < current_batch_size

            # Calculate necessary quantities for the batch
            dnu_dI_antiholobit, trace_holo = WP.dnu_dI(batch_points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, batch_pullbacks, batch_indices)

            # --- Welford Update Logic (applied only to valid data) ---
            batch_n = current_batch_size # Number of valid samples in this batch
            new_n = n + batch_n
            new_n_float = new_n.astype(jnp.float64)

            # Helper function for safe division (avoids NaN from 0/0 or x/0)
            def safe_divide(numerator, denominator):
                # Ensure denominator is float for division
                denom_float = denominator.astype(jnp.float64)
                return jnp.where(denom_float == 0, 0.0, numerator / denom_float)

            # --- Process Volume (Weights) ---
            # Welford update for mean and M2 (sum of squares of differences from the current mean)
            # Combine M2: M2_new = M2_old + M2_batch + delta^2 * n_old * n_batch / n_new
            valid_weights = jnp.where(valid_mask, batch_weights, 0.0)
            batch_sum_weights = jnp.sum(valid_weights)
            batch_mean_weights = safe_divide(batch_sum_weights, batch_n)
            delta_vol_global = batch_mean_weights - vol_omega_mean
            new_vol_omega_mean = vol_omega_mean + delta_vol_global * safe_divide(batch_n, new_n)
            batch_M2_vol = jnp.sum(jnp.where(valid_mask, (batch_weights - batch_mean_weights)**2, 0.0))
            new_vol_omega_M2 = vol_omega_M2 + batch_M2_vol + delta_vol_global**2 * safe_divide(n * batch_n, new_n)
            vol_omega_mean = new_vol_omega_mean
            vol_omega_M2 = new_vol_omega_M2

            # --- Process int_omega_dIomega Numerator ---
            per_sample_int_omega_dIomega_num = jnp.einsum('x,xI->xI', batch_weights, trace_holo) # Shape (batch_size, nmoduli)
            valid_per_sample_int_omega_dIomega_num = jnp.where(valid_mask[:, None], per_sample_int_omega_dIomega_num, 0.0+0.0j)

            # Combine Real part
            batch_sum_real = jnp.sum(jnp.real(valid_per_sample_int_omega_dIomega_num), axis=0)
            batch_mean_real = safe_divide(batch_sum_real, batch_n)
            delta_real_global = batch_mean_real - int_omega_dIomega_num_mean_real
            new_int_omega_dIomega_num_mean_real = int_omega_dIomega_num_mean_real + delta_real_global * safe_divide(batch_n, new_n)
            batch_M2_real = jnp.sum(jnp.where(valid_mask[:, None], (jnp.real(per_sample_int_omega_dIomega_num) - batch_mean_real)**2, 0.0), axis=0)
            new_int_omega_dIomega_num_M2_real = int_omega_dIomega_num_M2_real + batch_M2_real + delta_real_global**2 * safe_divide(n * batch_n, new_n)
            int_omega_dIomega_num_mean_real = new_int_omega_dIomega_num_mean_real
            int_omega_dIomega_num_M2_real = new_int_omega_dIomega_num_M2_real

            # Combine Imaginary part
            batch_sum_imag = jnp.sum(jnp.imag(valid_per_sample_int_omega_dIomega_num), axis=0)
            batch_mean_imag = safe_divide(batch_sum_imag, batch_n)
            delta_imag_global = batch_mean_imag - int_omega_dIomega_num_mean_imag
            new_int_omega_dIomega_num_mean_imag = int_omega_dIomega_num_mean_imag + delta_imag_global * safe_divide(batch_n, new_n)
            batch_M2_imag = jnp.sum(jnp.where(valid_mask[:, None], (jnp.imag(per_sample_int_omega_dIomega_num) - batch_mean_imag)**2, 0.0), axis=0)
            new_int_omega_dIomega_num_M2_imag = int_omega_dIomega_num_M2_imag + batch_M2_imag + delta_imag_global**2 * safe_divide(n * batch_n, new_n)
            int_omega_dIomega_num_mean_imag = new_int_omega_dIomega_num_mean_imag
            int_omega_dIomega_num_M2_imag = new_int_omega_dIomega_num_M2_imag

            # --- Process gWP Part 1 Numerator ---
            wedge_the_two = jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)], dtype=jnp.float64)

            if not fullmatrix:
                per_sample_term1_num = jnp.einsum('x,xI,xI->xI', batch_weights, trace_holo, jnp.conjugate(trace_holo))
                per_sample_term2_num = jnp.einsum('x,ij,xIij,xIji->xI', batch_weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))
                mask_shape = valid_mask[:, None] # (batch_size, 1)
                axis_sum = 0
            else:
                per_sample_term1_num = jnp.einsum('x,xI,xJ->xIJ', batch_weights, trace_holo, jnp.conjugate(trace_holo))
                per_sample_term2_num = jnp.einsum('x,ij,xIij,xJji->xIJ', batch_weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))
                mask_shape = valid_mask[:, None, None] # (batch_size, 1, 1)
                axis_sum = 0

            per_sample_gWP_part1_num = -(per_sample_term1_num + per_sample_term2_num)
            valid_per_sample_gWP_part1_num = jnp.where(mask_shape, per_sample_gWP_part1_num, 0.0+0.0j)

            # Combine Real part
            batch_sum_gWP_real = jnp.sum(jnp.real(valid_per_sample_gWP_part1_num), axis=axis_sum)
            batch_mean_gWP_real = safe_divide(batch_sum_gWP_real, batch_n)
            delta_gWP_real_global = batch_mean_gWP_real - gWP_part1_num_mean_real
            new_gWP_part1_num_mean_real = gWP_part1_num_mean_real + delta_gWP_real_global * safe_divide(batch_n, new_n)
            batch_M2_gWP_real = jnp.sum(jnp.where(mask_shape, (jnp.real(per_sample_gWP_part1_num) - batch_mean_gWP_real)**2, 0.0), axis=axis_sum)
            new_gWP_part1_num_M2_real = gWP_part1_num_M2_real + batch_M2_gWP_real + delta_gWP_real_global**2 * safe_divide(n * batch_n, new_n)
            gWP_part1_num_mean_real = new_gWP_part1_num_mean_real
            gWP_part1_num_M2_real = new_gWP_part1_num_M2_real

            # Combine Imaginary part
            batch_sum_gWP_imag = jnp.sum(jnp.imag(valid_per_sample_gWP_part1_num), axis=axis_sum)
            batch_mean_gWP_imag = safe_divide(batch_sum_gWP_imag, batch_n)
            delta_gWP_imag_global = batch_mean_gWP_imag - gWP_part1_num_mean_imag
            new_gWP_part1_num_mean_imag = gWP_part1_num_mean_imag + delta_gWP_imag_global * safe_divide(batch_n, new_n)
            batch_M2_gWP_imag = jnp.sum(jnp.where(mask_shape, (jnp.imag(per_sample_gWP_part1_num) - batch_mean_gWP_imag)**2, 0.0), axis=axis_sum)
            new_gWP_part1_num_M2_imag = gWP_part1_num_M2_imag + batch_M2_gWP_imag + delta_gWP_imag_global**2 * safe_divide(n * batch_n, new_n)
            gWP_part1_num_mean_imag = new_gWP_part1_num_mean_imag
            gWP_part1_num_M2_imag = new_gWP_part1_num_M2_imag

            # Update total count only if the batch was valid
            n = jnp.where(batch_n > 0, new_n, n)

            # Pack and return updated state
            updated_state = (
                n, vol_omega_mean, vol_omega_M2,
                int_omega_dIomega_num_mean_real, int_omega_dIomega_num_mean_imag,
                int_omega_dIomega_num_M2_real, int_omega_dIomega_num_M2_imag,
                gWP_part1_num_mean_real, gWP_part1_num_mean_imag,
                gWP_part1_num_M2_real, gWP_part1_num_M2_imag
            )
            # Use jax.lax.cond to return unchanged state if batch_n is 0, preventing NaN propagation
            return jax.lax.cond(batch_n > 0,
                                lambda: updated_state,
                                lambda: current_state)


        # Run the loop
        final_state = jax.lax.fori_loop(0, num_batches_calc, loop_body, initial_state)

        # Unpack final state
        n_final, vol_omega_mean_final, vol_omega_M2_final, \
        int_omega_dIomega_num_mean_real_final, int_omega_dIomega_num_mean_imag_final, \
        int_omega_dIomega_num_M2_real_final, int_omega_dIomega_num_M2_imag_final, \
        gWP_part1_num_mean_real_final, gWP_part1_num_mean_imag_final, \
        gWP_part1_num_M2_real_final, gWP_part1_num_M2_imag_final = final_state

        # --- Final Calculations ---
        n_final_float = n_final.astype(jnp.float64)
        safe_n_minus_1 = jnp.maximum(1.0, n_final_float - 1.0)
        safe_n = jnp.maximum(1.0, n_final_float)
        zero_error_cond = n_final <= 1

        # Calculate final means and standard errors for the numerators and volume
        # Volume
        vol_omega_var = vol_omega_M2_final / safe_n_minus_1
        vol_omega_std_error = jnp.sqrt(jnp.maximum(0.0, vol_omega_var / safe_n))
        vol_omega_std_error = jnp.where(zero_error_cond, 0.0, vol_omega_std_error)

        # int_omega_dIomega Numerator
        int_omega_dIomega_num_mean = int_omega_dIomega_num_mean_real_final + 1j * int_omega_dIomega_num_mean_imag_final
        int_omega_dIomega_num_var_real = int_omega_dIomega_num_M2_real_final / safe_n_minus_1
        int_omega_dIomega_num_var_imag = int_omega_dIomega_num_M2_imag_final / safe_n_minus_1
        int_omega_dIomega_num_std_error_real = jnp.sqrt(jnp.maximum(0.0, int_omega_dIomega_num_var_real / safe_n))
        int_omega_dIomega_num_std_error_imag = jnp.sqrt(jnp.maximum(0.0, int_omega_dIomega_num_var_imag / safe_n))
        int_omega_dIomega_num_std_error_real = jnp.where(zero_error_cond, 0.0, int_omega_dIomega_num_std_error_real)
        int_omega_dIomega_num_std_error_imag = jnp.where(zero_error_cond, 0.0, int_omega_dIomega_num_std_error_imag)

        # gWP_part1 Numerator
        gWP_part1_num_mean = gWP_part1_num_mean_real_final + 1j * gWP_part1_num_mean_imag_final
        gWP_part1_num_var_real = gWP_part1_num_M2_real_final / safe_n_minus_1
        gWP_part1_num_var_imag = gWP_part1_num_M2_imag_final / safe_n_minus_1
        gWP_part1_num_std_error_real = jnp.sqrt(jnp.maximum(0.0, gWP_part1_num_var_real / safe_n))
        gWP_part1_num_std_error_imag = jnp.sqrt(jnp.maximum(0.0, gWP_part1_num_var_imag / safe_n))
        gWP_part1_num_std_error_real = jnp.where(zero_error_cond, 0.0, gWP_part1_num_std_error_real)
        gWP_part1_num_std_error_imag = jnp.where(zero_error_cond, 0.0, gWP_part1_num_std_error_imag)

        # --- Calculate gWP parts and propagate errors ---
        safe_final_vol_omega_mean = jnp.where(vol_omega_mean_final == 0, 1e-15, vol_omega_mean_final)
        Br = safe_final_vol_omega_mean
        dBr = vol_omega_std_error

        # Calculate gWP_part1 = <gWP_part1_num> / <vol>
        gWP_part1 = gWP_part1_num_mean / Br

        # Error propagation for gWP_part1 = A / B
        Ar = gWP_part1_num_mean_real_final
        Ai = gWP_part1_num_mean_imag_final
        dAr = gWP_part1_num_std_error_real
        dAi = gWP_part1_num_std_error_imag

        gWP_part1_var_real = (dAr/Br)**2 + (Ar*dBr/Br**2)**2
        gWP_part1_var_imag = (dAi/Br)**2 + (Ai*dBr/Br**2)**2
        gWP_part1_std_error_real = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_real))
        gWP_part1_std_error_imag = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_imag))

        # Calculate gWP_part2 = <int_omega_num> * <int_omega_num>* / <vol>^2
        int_omega_dIomega = int_omega_dIomega_num_mean # Mean of the numerator
        vol_denom = Br**2

        # Define inputs for error propagation for gWP_part2
        Cr = int_omega_dIomega_num_mean_real_final
        Ci = int_omega_dIomega_num_mean_imag_final
        dCr = int_omega_dIomega_num_std_error_real # Error of numerator mean
        dCi = int_omega_dIomega_num_std_error_imag # Error of numerator mean

        # Since fullmatrix is static, use Python if/else
        if fullmatrix:
            gWP_part2 = jnp.einsum('I,J->IJ', int_omega_dIomega, jnp.conjugate(int_omega_dIomega)) / vol_denom
            # Error propagation for gWP_part2 = C_I * C_J* / B^2 (same logic as in gwP_jax)
            term_Br_real_sq = ( (-2 * jnp.real(gWP_part2) / Br) * dBr )**2
            term_Br_imag_sq = ( (-2 * jnp.imag(gWP_part2) / Br) * dBr )**2
            term_I_real_sq = jnp.einsum('J,I->IJ', Cr**2 / Br**4, dCr**2) + jnp.einsum('J,I->IJ', Ci**2 / Br**4, dCi**2)
            term_I_imag_sq = jnp.einsum('J,I->IJ', Ci**2 / Br**4, dCr**2) + jnp.einsum('J,I->IJ', Cr**2 / Br**4, dCi**2)
            term_J_real_sq = jnp.einsum('I,J->IJ', Cr**2 / Br**4, dCr**2) + jnp.einsum('I,J->IJ', Ci**2 / Br**4, dCi**2)
            term_J_imag_sq = jnp.einsum('I,J->IJ', Ci**2 / Br**4, dCr**2) + jnp.einsum('I,J->IJ', Cr**2 / Br**4, dCi**2)

            gWP_part2_var_real = term_I_real_sq + term_J_real_sq + term_Br_real_sq
            gWP_part2_var_imag = term_I_imag_sq + term_J_imag_sq + term_Br_imag_sq

            # Diagonal correction
            diag_idx = jnp.diag_indices_from(gWP_part2)
            # Need to handle Cr, Ci, dCr, dCi indexing for diagonal
            diag_Cr = Cr[diag_idx[0]]
            diag_Ci = Ci[diag_idx[0]]
            diag_dCr = dCr[diag_idx[0]]
            diag_dCi = dCi[diag_idx[0]]
            diag_gWP_part2_real = jnp.real(gWP_part2[diag_idx])

            diag_term_Cr_sq = ( (2 * diag_Cr / Br**2) * diag_dCr )**2
            diag_term_Ci_sq = ( (2 * diag_Ci / Br**2) * diag_dCi )**2
            diag_term_Br_sq = ( (-2 * diag_gWP_part2_real / Br) * dBr )**2
            diag_gWP_part2_var_real = diag_term_Cr_sq + diag_term_Ci_sq + diag_term_Br_sq
            diag_gWP_part2_var_imag = jnp.zeros_like(diag_gWP_part2_var_real)

            gWP_part2_var_real = gWP_part2_var_real.at[diag_idx].set(diag_gWP_part2_var_real)
            gWP_part2_var_imag = gWP_part2_var_imag.at[diag_idx].set(diag_gWP_part2_var_imag)

            gWP_part2_std_error_real = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var_real))
            gWP_part2_std_error_imag = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var_imag))

        else: # not fullmatrix
            # Calculate diagonal elements
            diag_gWP_part2 = (int_omega_dIomega * jnp.conjugate(int_omega_dIomega)) / vol_denom # Real result

            # Error propagation: f = (Cr^2 + Ci^2) / Br^2 (same logic as in gwP_jax)
            term_Cr_sq = ( (2 * Cr / Br**2) * dCr )**2
            term_Ci_sq = ( (2 * Ci / Br**2) * dCi )**2
            term_Br_sq = ( (-2 * jnp.real(diag_gWP_part2) / Br) * dBr )**2

            diag_gWP_part2_var = term_Cr_sq + term_Ci_sq + term_Br_sq
            diag_gWP_part2_std_error_real = jnp.sqrt(jnp.maximum(0.0, diag_gWP_part2_var))
            diag_gWP_part2_std_error_imag = jnp.zeros_like(diag_gWP_part2_std_error_real)

            # Assign results with correct shape for the 'else' branch
            gWP_part2 = jnp.real(diag_gWP_part2) # Shape (nmoduli,)
            gWP_part2_std_error_real = diag_gWP_part2_std_error_real # Shape (nmoduli,)
            gWP_part2_std_error_imag = diag_gWP_part2_std_error_imag # Shape (nmoduli,)


        # Final gWP
        gWP = gWP_part1 + gWP_part2 # Shapes match due to static fullmatrix

        # Combine errors in quadrature (real and imaginary separately)
        # Shapes of std errors also match due to static fullmatrix
        gWP_std_error_real = jnp.sqrt(jnp.maximum(0.0, gWP_part1_std_error_real**2 + gWP_part2_std_error_real**2))
        gWP_std_error_imag = jnp.sqrt(jnp.maximum(0.0, gWP_part1_std_error_imag**2 + gWP_part2_std_error_imag**2))

        # Final complex standard error
        gWP_std_error = gWP_std_error_real + 1j * gWP_std_error_imag

        return gWP, gWP_std_error

    @partial(jax.jit, static_argnums=(11,)) # Ensure fullmatrix is static
    def gwP_jax_simple(points, weights, pullbacks, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, indices=None, fullmatrix=True):
        """Simple version of gwP calculation without error handling"""
        # pg is not available in static method, use WP directly if needed or pass pg instance
        # Assuming _find_max_dQ_coords_jax is accessible or redefined statically
        # if indices is None:
        #     indices = WP._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0) # Needs static context or pg instance

        # Assuming indices are provided or handled elsewhere if needed statically
        if indices is None:
             # This part cannot be dynamic inside jit if indices=None is the default path
             # It should be computed outside or passed explicitly if needed inside jit
             raise ValueError("Indices must be provided for jitted gwP_jax_simple")


        vol_omega = jnp.mean(weights)
        # Ensure dnu_dI is jittable and does not depend on external non-static state
        dnu_dI_antiholobit, trace_holo = WP.dnu_dI(points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, pullbacks, indices)

        # Calculate int_omega_dIomega
        int_omega_dIomega = jnp.mean(jnp.einsum('x,xI->xI', weights, trace_holo), axis=0)

        # Calculate matrix components
        int_dIomega_dJomegaholobit = jnp.einsum('x,xI,xJ->xIJ', weights, trace_holo, jnp.conjugate(trace_holo))

        # Precompute or ensure wedge_the_two is static
        #wedge_the_two = jnp.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=points.dtype.real_dtype) # Example for 3x3, adjust if needed
        #
        wedge_the_two = jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)])
        # Simplified: [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]

        dnu_dI_antiholobit_dJantiholobit = jnp.einsum('x,ij,xIij,xJji->xIJ', weights, wedge_the_two,
                                                      dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))

        # Calculate gWP components using safe division
        safe_vol_omega = jnp.where(vol_omega == 0, 1e-15, vol_omega)
        term1_num = jnp.mean(int_dIomega_dJomegaholobit, axis=0)
        term2_num = jnp.mean(dnu_dI_antiholobit_dJantiholobit, axis=0)
        # Since fullmatrix is static, we can use direct if/else
        if fullmatrix:
            term3 = jnp.einsum('I,J->IJ', int_omega_dIomega, jnp.conjugate(int_omega_dIomega)) / safe_vol_omega**2
            gWP = -(term1_num / safe_vol_omega + term2_num / safe_vol_omega) + term3
        else:
            # If not fullmatrix, assume diagonal or scalar result is expected
            term1_diag = jnp.einsum('II->I', term1_num) / safe_vol_omega
            term2_diag = jnp.einsum('II->I', term2_num) / safe_vol_omega
            term3_diag = jnp.abs(int_omega_dIomega)**2 / safe_vol_omega**2 # einsum('I,I->I', ...) / vol^2
            gWP = -(term1_diag + term2_diag) + term3_diag

        return gWP

    def gwP(self, points, weights, pullbacks, indices = None, fullmatrix = False, simple = False):
        if indices is None:
            indices = PointGenerator._find_max_dQ_coords_jax(jnp.array(points, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128))
        #return gwP_jax(points, self.BASIS['DQDZB0'], self.BASIS['DQDZF0'], self.kmoduli, self.BASIS['DI_DQB0'], self.BASIS['DI_DQF0'], self.BASIS['DI_DQZB0'], self.BASIS['DI_DQZF0'], weights,pullbacks, indices = indices, fullmatrix = fullmatrix)
        if simple:
            return WP.gwP_jax_simple(points, jnp.array(weights, dtype = jnp.float64), jnp.array(pullbacks, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), 
                       jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128), jnp.array(self.kmoduli, dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQB0'], dtype = jnp.complex128),
                         jnp.array(self.BASIS['DI_DQF0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZF0'], dtype = jnp.complex128), 
                         indices = indices, fullmatrix = fullmatrix), None
        return WP.gwP_jax(jnp.array(points, dtype = jnp.complex128), jnp.array(weights, dtype = jnp.float64), jnp.array(pullbacks, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), 
                       jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128), jnp.array(self.kmoduli, dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQB0'], dtype = jnp.complex128),
                         jnp.array(self.BASIS['DI_DQF0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZF0'], dtype = jnp.complex128), 
                         indices = indices, fullmatrix = fullmatrix)

    def gwP_batched(self, points, weights, pullbacks, indices = None, fullmatrix = False, batch_size = 1000, num_batches = None):
        nmoduli = len(self.BASIS['DI_DQF0'])
        if indices is None:
            indices = PointGenerator._find_max_dQ_coords_jax(jnp.array(points, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128))
        out, err = WP.gwP_jax_batched(jnp.array(points, dtype = jnp.complex128), jnp.array(weights, dtype = jnp.float64), jnp.array(pullbacks, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'],
                     dtype = jnp.complex128), jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128), jnp.array(self.kmoduli, dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQB0'], dtype = jnp.complex128),
                       jnp.array(self.BASIS['DI_DQF0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZF0'], dtype = jnp.complex128),
                         indices = indices, fullmatrix = fullmatrix, batch_size = batch_size, num_batches = num_batches, nmoduli = nmoduli)
        if fullmatrix == False and len(np.shape(out)) == 2:
            # Extract the diagonal elements from the matrix
            out = jnp.diag(out)
            err = jnp.diag(err)
        return out, err

    def redefine_basis(self, moduli_space_directions):
        self.pg.moduli_space_directions = np.array(moduli_space_directions)#+ [jax.nn.one_hot(i, 81) for i in range(20,30)])
        self.pg.get_moduli_space_metric = True
        self.pg._generate_dQdz_basis()
        self.pg._generate_padded_dIdQZ_basis()
        self.pg._generate_moduli_space_basis()
        self.BASIS = self.pg.BASIS
        print("success, updated basis")

    def get_moduli_psi_normalisation(self, points, weights, pullbacks, indices = None, fullmatrix = False, psivector = None, batch_size = 10000, num_batches = None):
        if psivector is None:
            psivector = jax.nn.one_hot(40, 81)
        self.redefine_basis(np.array([psivector]))
        gwp, gwpErr = self.gwP_batched(points, weights, pullbacks, indices = indices, fullmatrix = fullmatrix, batch_size = batch_size, num_batches = num_batches)
        print(f"integrated value: {gwp}+-{gwpErr}")
        return gwp, gwpErr, np.abs(gwp)/np.abs(gwpErr)

    def get_moduli_space_determinant(self, points, weights, pullbacks, indices = None, fullmatrix = True, all_vecs = None, batch_size = 1000, num_batches = None):
        if all_vecs is None:
            all_vecs = []
            #a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u = np.random.normal(0,5,21) + 1j*np.random.normal(0,5,21)
            print("Taking a 20-dimensional subset. Should be full rank, so skipping one of the 21.")
            for vec in np.eye(21)[:-1]:#note we skip u!
                #print(vec)
                a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u = vec
                all_vecs.append(np.array([u, 0, t, 0, s, 0, r, 0, q, 0, p, 0, o, 0, n, 0, m, \
                0, l, 0, k, 0, j, 0, i, 0, h, 0, g, 0, f, 0, e, 0, \
                d, 0, c, 0, b, 0, a, 0, b, 0, c, 0, d, 0, e, 0, f, \
                0, g, 0, h, 0, i, 0, j, 0, k, 0, l, 0, m, 0, n, 0, \
                o, 0, p, 0, q, 0, r, 0, s, 0, t, 0, u]))
                #vec.append(all_vecs)
        self.redefine_basis(np.array(all_vecs))
        gwp, gwpErr = self.gwP_batched(points, weights, pullbacks, indices = indices, fullmatrix = fullmatrix, batch_size = batch_size, num_batches = num_batches)
        det_gwp = jnp.linalg.det(gwp)
        gwp_inv = jnp.linalg.inv(gwp)
        det_gwp_error = jnp.sqrt(jnp.sum(jnp.abs(det_gwp * gwp_inv.T * gwpErr)**2))
        print(f"Determinant: {det_gwp}, Error: {det_gwp_error}, zscore: {np.abs(det_gwp)/np.abs(det_gwp_error)}")
        return det_gwp, det_gwp_error, np.abs(det_gwp)/np.abs(det_gwp_error)
