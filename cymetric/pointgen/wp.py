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
        if indices is None:
            indices = PointGenerator._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0)
        vol_omega = jnp.mean(weights)
        dnu_dI_antiholobit, trace_holo = WP.dnu_dI(points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0,pullbacks, indices)
        if not fullmatrix:
            # Calculate per-sample values for int_omega_dIomega
            per_sample_int_omega_dIomega = jnp.einsum('x,xI->xI', weights, trace_holo)
            int_omega_dIomega = jnp.mean(per_sample_int_omega_dIomega, axis=0)
            
            # Track real and imaginary errors separately
            int_omega_dIomega_real_std_error = jnp.std(jnp.real(per_sample_int_omega_dIomega), axis=0) / jnp.sqrt(per_sample_int_omega_dIomega.shape[0])
            int_omega_dIomega_imag_std_error = jnp.std(jnp.imag(per_sample_int_omega_dIomega), axis=0) / jnp.sqrt(per_sample_int_omega_dIomega.shape[0])
            int_omega_dIomega_std_error = int_omega_dIomega_real_std_error + 1j * int_omega_dIomega_imag_std_error
            
            int_dIomega_dJomegaholobit = jnp.einsum('x,xI,xI->xI', weights, trace_holo,jnp.conjugate(trace_holo))#weights has an omega wedge omega
            wedge_the_two =  jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)])
            dnu_dI_antiholobit_dJantiholobit = jnp.einsum('x,ij,xIij,xIji->xI',weights,wedge_the_two,dnu_dI_antiholobit,jnp.conjugate(dnu_dI_antiholobit))
            
            # Calculate the terms for each sample point
            per_sample_term1 = (int_dIomega_dJomegaholobit)/(vol_omega)
            per_sample_term2 = dnu_dI_antiholobit_dJantiholobit/vol_omega
            per_sample_gWP = -(per_sample_term1 + per_sample_term2)
            
            # Calculate mean and standard error for the first part, tracking real and imaginary separately
            gWP_part1 = jnp.mean(per_sample_gWP, axis=0)
            gWP_part1_real_std_error = jnp.std(jnp.real(per_sample_gWP), axis=0) / jnp.sqrt(per_sample_gWP.shape[0])
            gWP_part1_imag_std_error = jnp.std(jnp.imag(per_sample_gWP), axis=0) / jnp.sqrt(per_sample_gWP.shape[0])
            
            # Add the second term (which uses int_omega_dIomega)
            gWP_part2 = jnp.einsum('I,I->I', int_omega_dIomega, jnp.conjugate(int_omega_dIomega))/vol_omega**2
            
            # Propagate error from int_omega_dIomega to gWP_part2, tracking real and imaginary separately
            # For f = a*a*, df = 2*a*da
            gWP_part2_real_std_error = 2 * jnp.abs(int_omega_dIomega) * int_omega_dIomega_real_std_error / vol_omega**2
            gWP_part2_imag_std_error = 2 * jnp.abs(int_omega_dIomega) * int_omega_dIomega_imag_std_error / vol_omega**2
            
            # Combine the parts
            gWP = gWP_part1 + gWP_part2
            
            # Add errors in quadrature for real and imaginary parts separately
            gWP_real_std_error = jnp.sqrt(gWP_part1_real_std_error**2 + gWP_part2_real_std_error**2)
            gWP_imag_std_error = jnp.sqrt(gWP_part1_imag_std_error**2 + gWP_part2_imag_std_error**2)
            
            # Combine into complex std error
            gWP_std_error = gWP_real_std_error + 1j * gWP_imag_std_error
            
        else:
            # Calculate per-sample values for int_omega_dIomega
            per_sample_int_omega_dIomega = jnp.einsum('x,xI->xI', weights, trace_holo)
            int_omega_dIomega = jnp.mean(per_sample_int_omega_dIomega, axis=0)
            
            # Track real and imaginary errors separately
            int_omega_dIomega_real_std_error = jnp.std(jnp.real(per_sample_int_omega_dIomega), axis=0) / jnp.sqrt(per_sample_int_omega_dIomega.shape[0])
            int_omega_dIomega_imag_std_error = jnp.std(jnp.imag(per_sample_int_omega_dIomega), axis=0) / jnp.sqrt(per_sample_int_omega_dIomega.shape[0])
            int_omega_dIomega_std_error = int_omega_dIomega_real_std_error + 1j * int_omega_dIomega_imag_std_error
            
            int_dIomega_dJomegaholobit = jnp.einsum('x,xI,xJ->xIJ', weights, trace_holo,jnp.conjugate(trace_holo))
            wedge_the_two =  jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)])
            dnu_dI_antiholobit_dJantiholobit = jnp.einsum('x,ij,xIij,xJji->xIJ',weights,wedge_the_two,dnu_dI_antiholobit,jnp.conjugate(dnu_dI_antiholobit))
            
            # Calculate the terms for each sample point
            per_sample_term1 = (int_dIomega_dJomegaholobit)/(vol_omega)
            per_sample_term2 = dnu_dI_antiholobit_dJantiholobit/vol_omega
            per_sample_gWP = -(per_sample_term1 + per_sample_term2)
            
            # Calculate mean and standard error for the first part, tracking real and imaginary separately
            gWP_part1 = jnp.mean(per_sample_gWP, axis=0)
            gWP_part1_real_std_error = jnp.std(jnp.real(per_sample_gWP), axis=0) / jnp.sqrt(per_sample_gWP.shape[0])
            gWP_part1_imag_std_error = jnp.std(jnp.imag(per_sample_gWP), axis=0) / jnp.sqrt(per_sample_gWP.shape[0])
            
            # Add the second term (which uses int_omega_dIomega)
            gWP_part2 = jnp.einsum('I,J->IJ', int_omega_dIomega, jnp.conjugate(int_omega_dIomega))/vol_omega**2
            
            # Propagate error from int_omega_dIomega to gWP_part2 (matrix form)
            # For f_IJ = a_I*a_J*, df_IJ depends on both da_I and da_J
            # Handle real and imaginary parts separately
            gWP_part2_real_std_error = jnp.sqrt(
                jnp.einsum('I,J->IJ', (2 * jnp.abs(int_omega_dIomega) * int_omega_dIomega_real_std_error)**2, jnp.ones_like(int_omega_dIomega)) +
                jnp.einsum('I,J->JI', (2 * jnp.abs(int_omega_dIomega) * int_omega_dIomega_real_std_error)**2, jnp.ones_like(int_omega_dIomega))
            ) / vol_omega**2 / 2  # Divide by 2 to avoid double counting
            
            gWP_part2_imag_std_error = jnp.sqrt(
                jnp.einsum('I,J->IJ', (2 * jnp.abs(int_omega_dIomega) * int_omega_dIomega_imag_std_error)**2, jnp.ones_like(int_omega_dIomega)) +
                jnp.einsum('I,J->JI', (2 * jnp.abs(int_omega_dIomega) * int_omega_dIomega_imag_std_error)**2, jnp.ones_like(int_omega_dIomega))
            ) / vol_omega**2 / 2  # Divide by 2 to avoid double counting
            
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

        Args:
            points: Sample points. Shape (total_points, ...).
            weights: Weights for each sample point. Shape (total_points,).
            pullbacks: Pullback metrics for each point. Shape (total_points, ...).
            DQDZB0, DQDZF0: Basis components.
            kmoduli: Kahler moduli.
            DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0: Basis derivatives.
            indices: Indices for coordinate selection. Shape (total_points, ...). Required.
            fullmatrix: Whether to compute the full gWP matrix or just the diagonal.
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

        # Determine the shape for gWP part 1 based on fullmatrix flag
        gWP_part1_shape = (nmoduli, nmoduli) if fullmatrix else (nmoduli,)

        # Initialize accumulators for online mean and variance calculation (Welford's algorithm)
        # State tuple for fori_loop
        initial_state = (
            jnp.zeros((), dtype=jnp.int64), # n: Total samples processed
            jnp.zeros((), dtype=jnp.float64), # vol_omega_mean
            jnp.zeros((), dtype=jnp.float64), # vol_omega_M2
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_mean_real
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_mean_imag
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_M2_real
            jnp.zeros(nmoduli, dtype=jnp.float64), # int_omega_dIomega_M2_imag
            jnp.zeros(gWP_part1_shape, dtype=jnp.float64), # gWP_part1_mean_real
            jnp.zeros(gWP_part1_shape, dtype=jnp.float64), # gWP_part1_mean_imag
            jnp.zeros(gWP_part1_shape, dtype=jnp.float64), # gWP_part1_M2_real
            jnp.zeros(gWP_part1_shape, dtype=jnp.float64)  # gWP_part1_M2_imag
        )

        # Define the loop body function for jax.lax.fori_loop
        def loop_body(i, current_state):
            # Unpack current state
            n, vol_omega_mean, vol_omega_M2, \
            int_omega_dIomega_mean_real, int_omega_dIomega_mean_imag, \
            int_omega_dIomega_M2_real, int_omega_dIomega_M2_imag, \
            gWP_part1_mean_real, gWP_part1_mean_imag, \
            gWP_part1_M2_real, gWP_part1_M2_imag = current_state

            # Calculate batch indices
            start_idx = i * batch_size
            # Use static batch_size for slicing, actual data size might be smaller for the last batch
            # dynamic_slice handles out-of-bounds slicing gracefully if start_idx is too large
            # but we need the actual size for Welford updates.
            current_batch_size = jnp.minimum(batch_size, total_points - start_idx)

            # Slice batch data using dynamic slices
            # Note: dynamic_slice requires slice_sizes to be static or traceable.
            # Using batch_size here, assuming padding or careful handling of the last batch.
            # A potentially safer way is to pad the input arrays to be multiples of batch_size.
            # Let's stick to dynamic_slice but calculate actual size `current_batch_size`.
            batch_points = jax.lax.dynamic_slice_in_dim(points, start_idx, batch_size, axis=0)
            batch_weights = jax.lax.dynamic_slice_in_dim(weights, start_idx, batch_size, axis=0).astype(jnp.float64)
            batch_pullbacks = jax.lax.dynamic_slice_in_dim(pullbacks, start_idx, batch_size, axis=0)
            batch_indices = jax.lax.dynamic_slice_in_dim(indices, start_idx, batch_size, axis=0)

            # Mask invalid entries in the last batch if total_points is not a multiple of batch_size
            # Create a mask for valid entries in the current batch
            valid_mask = jnp.arange(batch_size) < current_batch_size

            # Calculate necessary quantities for the batch
            dnu_dI_antiholobit, trace_holo = WP.dnu_dI(batch_points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, batch_pullbacks, batch_indices)

            # --- Welford Update Logic (applied only to valid data) ---
            batch_n = current_batch_size # Number of valid samples in this batch
            new_n = n + batch_n

            # Helper function for safe division (avoids NaN from 0/0 or x/0)
            def safe_divide(numerator, denominator):
                return jnp.where(denominator == 0, 0.0, numerator / denominator)

            # --- Process Volume (Weights) ---
            valid_weights = jnp.where(valid_mask, batch_weights, 0.0)
            delta_vol = valid_weights - vol_omega_mean
            vol_omega_mean_update = jnp.sum(jnp.where(valid_mask, delta_vol, 0.0)) * safe_divide(1.0, new_n.astype(jnp.float64))
            vol_omega_mean += vol_omega_mean_update
            delta2_vol = valid_weights - vol_omega_mean # Use updated mean
            vol_omega_M2 += jnp.sum(jnp.where(valid_mask, delta_vol * delta2_vol, 0.0))

            # --- Process int_omega_dIomega ---
            per_sample_int_omega_dIomega = jnp.einsum('x,xI->xI', batch_weights, trace_holo) # Shape (batch_size, nmoduli)
            valid_per_sample_int_omega_dIomega = jnp.where(valid_mask[:, None], per_sample_int_omega_dIomega, 0.0)

            # Real part
            delta_real = jnp.real(valid_per_sample_int_omega_dIomega) - int_omega_dIomega_mean_real
            int_omega_dIomega_mean_real_update = jnp.sum(jnp.where(valid_mask[:, None], delta_real, 0.0), axis=0) * safe_divide(1.0, new_n.astype(jnp.float64))
            int_omega_dIomega_mean_real += int_omega_dIomega_mean_real_update
            delta2_real = jnp.real(valid_per_sample_int_omega_dIomega) - int_omega_dIomega_mean_real
            int_omega_dIomega_M2_real += jnp.sum(jnp.where(valid_mask[:, None], delta_real * delta2_real, 0.0), axis=0)

            # Imaginary part
            delta_imag = jnp.imag(valid_per_sample_int_omega_dIomega) - int_omega_dIomega_mean_imag
            int_omega_dIomega_mean_imag_update = jnp.sum(jnp.where(valid_mask[:, None], delta_imag, 0.0), axis=0) * safe_divide(1.0, new_n.astype(jnp.float64))
            int_omega_dIomega_mean_imag += int_omega_dIomega_mean_imag_update
            delta2_imag = jnp.imag(valid_per_sample_int_omega_dIomega) - int_omega_dIomega_mean_imag
            int_omega_dIomega_M2_imag += jnp.sum(jnp.where(valid_mask[:, None], delta_imag * delta2_imag, 0.0), axis=0)

            # --- Process gWP Part 1 ---
            wedge_the_two = jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)], dtype=jnp.float64)

            if not fullmatrix:
                per_sample_term1_num = jnp.einsum('x,xI,xI->xI', batch_weights, trace_holo, jnp.conjugate(trace_holo))
                per_sample_term2_num = jnp.einsum('x,ij,xIij,xIji->xI', batch_weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))
                mask_shape = valid_mask[:, None] # (batch_size, 1)
            else:
                per_sample_term1_num = jnp.einsum('x,xI,xJ->xIJ', batch_weights, trace_holo, jnp.conjugate(trace_holo))
                per_sample_term2_num = jnp.einsum('x,ij,xIij,xJji->xIJ', batch_weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))
                mask_shape = valid_mask[:, None, None] # (batch_size, 1, 1)

            safe_current_vol_omega_mean = jnp.where(vol_omega_mean == 0, 1e-15, vol_omega_mean)
            per_sample_gWP_part1 = -(per_sample_term1_num / safe_current_vol_omega_mean + per_sample_term2_num / safe_current_vol_omega_mean)
            valid_per_sample_gWP_part1 = jnp.where(mask_shape, per_sample_gWP_part1, 0.0)

            # Real part
            delta_gWP_real = jnp.real(valid_per_sample_gWP_part1) - gWP_part1_mean_real
            gWP_part1_mean_real_update = jnp.sum(jnp.where(mask_shape, delta_gWP_real, 0.0), axis=0) * safe_divide(1.0, new_n.astype(jnp.float64))
            gWP_part1_mean_real += gWP_part1_mean_real_update
            delta2_gWP_real = jnp.real(valid_per_sample_gWP_part1) - gWP_part1_mean_real
            gWP_part1_M2_real += jnp.sum(jnp.where(mask_shape, delta_gWP_real * delta2_gWP_real, 0.0), axis=0)

            # Imaginary part
            delta_gWP_imag = jnp.imag(valid_per_sample_gWP_part1) - gWP_part1_mean_imag
            gWP_part1_mean_imag_update = jnp.sum(jnp.where(mask_shape, delta_gWP_imag, 0.0), axis=0) * safe_divide(1.0, new_n.astype(jnp.float64))
            gWP_part1_mean_imag += gWP_part1_mean_imag_update
            delta2_gWP_imag = jnp.imag(valid_per_sample_gWP_part1) - gWP_part1_mean_imag
            gWP_part1_M2_imag += jnp.sum(jnp.where(mask_shape, delta_gWP_imag * delta2_gWP_imag, 0.0), axis=0)

            # Update total count only if the batch was valid
            n = jnp.where(batch_n > 0, new_n, n)

            # Pack and return updated state
            updated_state = (
                n, vol_omega_mean, vol_omega_M2,
                int_omega_dIomega_mean_real, int_omega_dIomega_mean_imag,
                int_omega_dIomega_M2_real, int_omega_dIomega_M2_imag,
                gWP_part1_mean_real, gWP_part1_mean_imag,
                gWP_part1_M2_real, gWP_part1_M2_imag
            )
            # Use jax.lax.cond to return unchanged state if batch_n is 0, preventing NaN propagation
            return jax.lax.cond(batch_n > 0,
                                lambda: updated_state,
                                lambda: current_state)


        # Run the loop
        final_state = jax.lax.fori_loop(0, num_batches_calc, loop_body, initial_state)

        # Unpack final state
        n_final, vol_omega_mean_final, vol_omega_M2_final, \
        int_omega_dIomega_mean_real_final, int_omega_dIomega_mean_imag_final, \
        int_omega_dIomega_M2_real_final, int_omega_dIomega_M2_imag_final, \
        gWP_part1_mean_real_final, gWP_part1_mean_imag_final, \
        gWP_part1_M2_real_final, gWP_part1_M2_imag_final = final_state

        # --- Final Calculations ---
        # Ensure n > 1 for variance calculation
        n_final_float = n_final.astype(jnp.float64)
        # Use safe division for variance (n-1) and std error (n)
        # Avoid division by zero or negative numbers under sqrt
        safe_n_minus_1 = jnp.maximum(1.0, n_final_float - 1.0)
        safe_n = jnp.maximum(1.0, n_final_float)

        # Calculate sample variance (M2 / (n - 1))
        int_omega_dIomega_var_real = int_omega_dIomega_M2_real_final / safe_n_minus_1
        int_omega_dIomega_var_imag = int_omega_dIomega_M2_imag_final / safe_n_minus_1
        gWP_part1_var_real = gWP_part1_M2_real_final / safe_n_minus_1
        gWP_part1_var_imag = gWP_part1_M2_imag_final / safe_n_minus_1

        # Calculate standard error of the mean (sqrt(variance / n))
        # Ensure variance is non-negative before sqrt
        int_omega_dIomega_std_error_real = jnp.sqrt(jnp.maximum(0.0, int_omega_dIomega_var_real / safe_n))
        int_omega_dIomega_std_error_imag = jnp.sqrt(jnp.maximum(0.0, int_omega_dIomega_var_imag / safe_n))
        gWP_part1_std_error_real = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_real / safe_n))
        gWP_part1_std_error_imag = jnp.sqrt(jnp.maximum(0.0, gWP_part1_var_imag / safe_n))

        # Handle case where n <= 1 explicitly for errors: std error should be 0
        zero_error_cond = n_final <= 1
        int_omega_dIomega_std_error_real = jnp.where(zero_error_cond, 0.0, int_omega_dIomega_std_error_real)
        int_omega_dIomega_std_error_imag = jnp.where(zero_error_cond, 0.0, int_omega_dIomega_std_error_imag)
        gWP_part1_std_error_real = jnp.where(zero_error_cond, 0.0, gWP_part1_std_error_real)
        gWP_part1_std_error_imag = jnp.where(zero_error_cond, 0.0, gWP_part1_std_error_imag)


        # Combine final means
        int_omega_dIomega = int_omega_dIomega_mean_real_final + 1j * int_omega_dIomega_mean_imag_final
        gWP_part1 = gWP_part1_mean_real_final + 1j * gWP_part1_mean_imag_final

        # Calculate the second part of gWP and propagate errors
        final_vol_omega_mean = vol_omega_mean_final
        safe_final_vol_omega_mean = jnp.where(final_vol_omega_mean == 0, 1e-15, final_vol_omega_mean)
        vol_denom = safe_final_vol_omega_mean**2

        # Define inputs for error propagation (these will be zero if n <= 1)
        a_Ir = int_omega_dIomega_mean_real_final
        a_Ii = int_omega_dIomega_mean_imag_final
        da_Ir = int_omega_dIomega_std_error_real
        da_Ii = int_omega_dIomega_std_error_imag
        a_Jr = int_omega_dIomega_mean_real_final
        a_Ji = -int_omega_dIomega_mean_imag_final
        da_Jr = int_omega_dIomega_std_error_real
        da_Ji = int_omega_dIomega_std_error_imag

        # Use jax.lax.cond for branching based on `fullmatrix` (static argument)
        def calc_part2_fullmatrix_true():
            gWP_part2 = jnp.einsum('I,J->IJ', int_omega_dIomega, jnp.conjugate(int_omega_dIomega)) / vol_denom
            # Error propagation: var(f(x,y)) approx (df/dx*err_x)^2 + (df/dy*err_y)^2 + ...
            # Here f = (a_Ir + i*a_Ii)*(a_Jr - i*a_Ji) / vol_denom
            # Real part: (a_Ir*a_Jr + a_Ii*a_Ji) / vol_denom
            # Imag part: (a_Ii*a_Jr - a_Ir*a_Ji) / vol_denom
            # Variance calculation assumes independence of errors for different I, J and real/imag parts
            vol_denom_sq = vol_denom**2 # vol_mean^4

            # Variance of Real Part
            # Terms from derivatives w.r.t. a_Ir, a_Ii, a_Jr, a_Ji
            gWP_part2_var_real = (
                jnp.einsum('J,I->IJ', a_Jr**2, da_Ir**2) + # (dReal/da_Ir * da_Ir)^2 = (a_Jr/vol * da_Ir)^2
                jnp.einsum('J,I->IJ', a_Ji**2, da_Ii**2) + # (dReal/da_Ii * da_Ii)^2 = (a_Ji/vol * da_Ii)^2
                jnp.einsum('I,J->IJ', a_Ir**2, da_Jr**2) + # (dReal/da_Jr * da_Jr)^2 = (a_Ir/vol * da_Jr)^2
                jnp.einsum('I,J->IJ', a_Ii**2, da_Ji**2)   # (dReal/da_Ji * da_Ji)^2 = (a_Ii/vol * da_Ji)^2
            ) / vol_denom_sq # Division by vol_denom^2 happens here

            # Variance of Imaginary Part
            # Terms from derivatives w.r.t. a_Ir, a_Ii, a_Jr, a_Ji
            gWP_part2_var_imag = (
                jnp.einsum('J,I->IJ', a_Ji**2, da_Ir**2) + # (dImag/da_Ir * da_Ir)^2 = (-a_Ji/vol * da_Ir)^2
                jnp.einsum('J,I->IJ', a_Jr**2, da_Ii**2) + # (dImag/da_Ii * da_Ii)^2 = (a_Jr/vol * da_Ii)^2
                jnp.einsum('I,J->IJ', a_Ii**2, da_Jr**2) + # (dImag/da_Jr * da_Jr)^2 = (a_Ii/vol * da_Jr)^2
                jnp.einsum('I,J->IJ', a_Ir**2, da_Ji**2)   # (dImag/da_Ji * da_Ji)^2 = (-a_Ir/vol * da_Ji)^2
            ) / vol_denom_sq # Division by vol_denom^2 happens here

            gWP_part2_std_error_real = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var_real))
            gWP_part2_std_error_imag = jnp.sqrt(jnp.maximum(0.0, gWP_part2_var_imag))
            return gWP_part2, gWP_part2_std_error_real, gWP_part2_std_error_imag

        def calc_part2_fullmatrix_false():
            # Calculate diagonal elements
            diag_gWP_part2 = int_omega_dIomega * jnp.conjugate(int_omega_dIomega) / vol_denom # Complex version
            
            # Error propagation: f = (a_Ir^2 + a_Ii^2) / vol_denom
            # var(f) approx (df/da_Ir*da_Ir)^2 + (df/da_Ii*da_Ii)^2
            vol_denom_sq = vol_denom**2 # vol_mean^4

            # Corrected calculation based on formula df/dx = 2*a_Ir/vol_denom:
            diag_gWP_part2_var_corrected = (
                ( (2 * a_Ir / vol_denom) * da_Ir )**2 +
                ( (2 * a_Ii / vol_denom) * da_Ii )**2
            )

            diag_gWP_part2_std_error_real = jnp.sqrt(jnp.maximum(0.0, diag_gWP_part2_var_corrected))
            # gWP_part2 is real, so imag error is zero
            diag_gWP_part2_std_error_imag = jnp.zeros_like(a_Ir)
            
            # Pad to match the shape of the fullmatrix=True branch
            # Create matrix with zeros and put diagonal values in place
            nmoduli = int_omega_dIomega.shape[0]
            gWP_part2 = jnp.zeros((nmoduli, nmoduli), dtype=diag_gWP_part2.dtype)
            gWP_part2_std_error_real = jnp.zeros((nmoduli, nmoduli), dtype=diag_gWP_part2_std_error_real.dtype)
            gWP_part2_std_error_imag = jnp.zeros((nmoduli, nmoduli), dtype=diag_gWP_part2_std_error_imag.dtype)
            
            # Fill diagonals
            gWP_part2 = gWP_part2.at[jnp.diag_indices(nmoduli)].set(diag_gWP_part2)
            gWP_part2_std_error_real = gWP_part2_std_error_real.at[jnp.diag_indices(nmoduli)].set(diag_gWP_part2_std_error_real)
            gWP_part2_std_error_imag = gWP_part2_std_error_imag.at[jnp.diag_indices(nmoduli)].set(diag_gWP_part2_std_error_imag)
            
            return gWP_part2, gWP_part2_std_error_real, gWP_part2_std_error_imag

        # Use lax.cond - note `fullmatrix` must be a static argument for gwP_jax_batched
        gWP_part2, gWP_part2_std_error_real, gWP_part2_std_error_imag = jax.lax.cond(
            fullmatrix,
            calc_part2_fullmatrix_true,
            calc_part2_fullmatrix_false
        )

        # Final gWP
        gWP = gWP_part1 + gWP_part2

        # Combine errors in quadrature (real and imaginary separately)
        # Variances add, so std errors add in quadrature. Max(0,...) ensures non-negative variance.
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
