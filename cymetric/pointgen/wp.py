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
    @partial(jax.jit, static_argnums=(11, 12, 13, 14))
    def gwP_jax_batched(points, weights, pullbacks, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, indices = None, fullmatrix = False, batch_size = 1000, num_batches = None, nmoduli = 1):
        """
        Calculates gWP and its standard error using online Welford algorithm for batches
        to avoid memory issues with large datasets.
        """
        if indices is None:
            # Avoid potential side effects if pg is mutable and used elsewhere
            # Ensure indices calculation happens if needed, but consider passing explicitly
            # indices = pg._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0)
            # Placeholder if pg is not available in this static context or should not be used directly
            # Replace with appropriate index calculation if necessary
             raise ValueError("Indices must be provided or calculated beforehand for batched computation.")


        total_points = points.shape[0]
        if num_batches is None:
            num_batches = (total_points + batch_size - 1) // batch_size # Correct ceiling division

        # Initialize accumulators for online mean and variance calculation (Welford's algorithm)
        vol_omega_mean = 0.0
        vol_omega_M2 = 0.0

        # Accumulators for int_omega_dIomega
        # Shape (nmoduli,) - assuming trace_holo always returns shape (nmoduli,)
        int_omega_dIomega_mean_real = jnp.zeros(nmoduli, dtype=jnp.float64)
        int_omega_dIomega_mean_imag = jnp.zeros(nmoduli, dtype=jnp.float64)
        int_omega_dIomega_M2_real = jnp.zeros(nmoduli, dtype=jnp.float64)
        int_omega_dIomega_M2_imag = jnp.zeros(nmoduli, dtype=jnp.float64)

        # Accumulators for the first part of gWP (before adding the term with squared int_omega_dIomega)
        gWP_part1_shape = (nmoduli, nmoduli) if fullmatrix else (nmoduli,)
        gWP_part1_mean_real = jnp.zeros(gWP_part1_shape, dtype=jnp.float64)
        gWP_part1_mean_imag = jnp.zeros(gWP_part1_shape, dtype=jnp.float64)
        gWP_part1_M2_real = jnp.zeros(gWP_part1_shape, dtype=jnp.float64)
        gWP_part1_M2_imag = jnp.zeros(gWP_part1_shape, dtype=jnp.float64)

        n = 0 # Total samples processed

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_points)
            batch_size_actual = end_idx - start_idx
            if batch_size_actual == 0: continue # Skip empty batches

            n_old = n
            n += batch_size_actual

            # Slice batch data
            batch_points = jax.lax.dynamic_slice_in_dim(points, start_idx, batch_size_actual, axis=0)
            batch_weights = jax.lax.dynamic_slice_in_dim(weights, start_idx, batch_size_actual, axis=0)
            batch_pullbacks = jax.lax.dynamic_slice_in_dim(pullbacks, start_idx, batch_size_actual, axis=0)
            batch_indices = jax.lax.dynamic_slice_in_dim(indices, start_idx, batch_size_actual, axis=0) if indices is not None else None

            # --- Process Volume (Weights) ---
            # Welford update for volume (mean of weights)
            # Process each weight individually for correct variance
            # This can be vectorized for the batch
            batch_delta_vol = batch_weights - vol_omega_mean
            vol_omega_mean += jnp.sum(batch_delta_vol) / n
            batch_delta2_vol = batch_weights - vol_omega_mean # Use updated mean
            vol_omega_M2 += jnp.sum(batch_delta_vol * batch_delta2_vol)
            # Note: We only need the final mean volume, std error not directly used but M2 is calculated

            # --- Process int_omega_dIomega ---
            dnu_dI_antiholobit, trace_holo = WP.dnu_dI(batch_points, DQDZB0, DQDZF0, kmoduli, DI_DQB0, DI_DQF0, DI_DQZB0, DI_DQZF0, batch_pullbacks, batch_indices)

            # Calculate per-sample values, weighted
            per_sample_int_omega_dIomega = jnp.einsum('x,xI->xI', batch_weights, trace_holo)

            # Welford update for int_omega_dIomega (real part)
            batch_delta_real = jnp.real(per_sample_int_omega_dIomega) - int_omega_dIomega_mean_real
            int_omega_dIomega_mean_real += jnp.sum(batch_delta_real, axis=0) / n
            batch_delta2_real = jnp.real(per_sample_int_omega_dIomega) - int_omega_dIomega_mean_real # Use updated mean
            int_omega_dIomega_M2_real += jnp.sum(batch_delta_real * batch_delta2_real, axis=0)

            # Welford update for int_omega_dIomega (imaginary part)
            batch_delta_imag = jnp.imag(per_sample_int_omega_dIomega) - int_omega_dIomega_mean_imag
            int_omega_dIomega_mean_imag += jnp.sum(batch_delta_imag, axis=0) / n
            batch_delta2_imag = jnp.imag(per_sample_int_omega_dIomega) - int_omega_dIomega_mean_imag # Use updated mean
            int_omega_dIomega_M2_imag += jnp.sum(batch_delta_imag * batch_delta2_imag, axis=0)

            # --- Process gWP Part 1 ---
            # Calculate per-sample values for the first part of gWP
            wedge_the_two = jnp.array([[(-1)**((j+1)-(i+1)-1) for i in range(3)] for j in range(3)]) # Assuming fixed size 3x3

            if not fullmatrix:
                # Shape (batch_size_actual, nmoduli)
                per_sample_term1 = jnp.einsum('x,xI,xI->xI', batch_weights, trace_holo, jnp.conjugate(trace_holo))
                per_sample_term2 = jnp.einsum('x,ij,xIij,xIji->xI', batch_weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))
            else:
                # Shape (batch_size_actual, nmoduli, nmoduli)
                per_sample_term1 = jnp.einsum('x,xI,xJ->xIJ', batch_weights, trace_holo, jnp.conjugate(trace_holo))
                per_sample_term2 = jnp.einsum('x,ij,xIij,xJji->xIJ', batch_weights, wedge_the_two, dnu_dI_antiholobit, jnp.conjugate(dnu_dI_antiholobit))

            # Need mean volume *per batch* for this term's definition? Or overall mean?
            # The original code used per-sample weights / vol_omega. Let's use overall mean vol_omega_mean.
            # If vol_omega_mean is 0 initially, this will cause issues. Use a small epsilon or handle division by zero.
            # Using the *current* overall mean vol_omega_mean seems most appropriate for the online calculation.
            current_vol_omega_mean = vol_omega_mean # Mean volume calculated up to *this* batch
            safe_vol_omega_mean = jnp.where(current_vol_omega_mean == 0, 1e-15, current_vol_omega_mean) # Avoid division by zero

            per_sample_gWP_part1 = -(per_sample_term1 / safe_vol_omega_mean + per_sample_term2/safe_vol_omega_mean)

            # Welford update for gWP_part1 (real part)
            batch_delta_gWP_real = jnp.real(per_sample_gWP_part1) - gWP_part1_mean_real
            gWP_part1_mean_real += jnp.sum(batch_delta_gWP_real, axis=0) / n
            batch_delta2_gWP_real = jnp.real(per_sample_gWP_part1) - gWP_part1_mean_real # Use updated mean
            gWP_part1_M2_real += jnp.sum(batch_delta_gWP_real * batch_delta2_gWP_real, axis=0)

            # Welford update for gWP_part1 (imaginary part)
            batch_delta_gWP_imag = jnp.imag(per_sample_gWP_part1) - gWP_part1_mean_imag
            gWP_part1_mean_imag += jnp.sum(batch_delta_gWP_imag, axis=0) / n
            batch_delta2_gWP_imag = jnp.imag(per_sample_gWP_part1) - gWP_part1_mean_imag # Use updated mean
            gWP_part1_M2_imag += jnp.sum(batch_delta_gWP_imag * batch_delta2_gWP_imag, axis=0)

        # --- Final Calculations ---
        if n <= 1:
            # Handle cases with insufficient data points for variance/std error calculation
            zero_shape_I = jnp.zeros(nmoduli, dtype=jnp.float64)
            zero_shape_gWP = jnp.zeros(gWP_part1_shape, dtype=jnp.float64)
            int_omega_dIomega_std_error_real = zero_shape_I
            int_omega_dIomega_std_error_imag = zero_shape_I
            gWP_part1_std_error_real = zero_shape_gWP
            gWP_part1_std_error_imag = zero_shape_gWP
        else:
            # Calculate standard error of the mean for int_omega_dIomega
            int_omega_dIomega_var_real = int_omega_dIomega_M2_real / (n - 1)
            int_omega_dIomega_var_imag = int_omega_dIomega_M2_imag / (n - 1)
            int_omega_dIomega_std_error_real = jnp.sqrt(int_omega_dIomega_var_real / n)
            int_omega_dIomega_std_error_imag = jnp.sqrt(int_omega_dIomega_var_imag / n)

            # Calculate standard error of the mean for gWP_part1
            gWP_part1_var_real = gWP_part1_M2_real / (n - 1)
            gWP_part1_var_imag = gWP_part1_M2_imag / (n - 1)
            gWP_part1_std_error_real = jnp.sqrt(gWP_part1_var_real / n)
            gWP_part1_std_error_imag = jnp.sqrt(gWP_part1_var_imag / n)

        # Combine final means
        int_omega_dIomega = int_omega_dIomega_mean_real + 1j * int_omega_dIomega_mean_imag
        gWP_part1 = gWP_part1_mean_real + 1j * gWP_part1_mean_imag

        # Combine final standard errors (real + 1j * imag)
        int_omega_dIomega_std_error = int_omega_dIomega_std_error_real + 1j * int_omega_dIomega_std_error_imag
        gWP_part1_std_error = gWP_part1_std_error_real + 1j * gWP_part1_std_error_imag # Keep real/imag separate for final combination

        # Calculate the second part of gWP and propagate errors
        final_vol_omega_mean = vol_omega_mean
        safe_final_vol_omega_mean = jnp.where(final_vol_omega_mean == 0, 1e-15, final_vol_omega_mean)
        vol_denom = safe_final_vol_omega_mean**2

        if not fullmatrix:
            # gWP_part2 = |int_omega_dIomega|^2 / vol**2
            gWP_part2 = jnp.einsum('I,I->I', int_omega_dIomega, jnp.conjugate(int_omega_dIomega)) / vol_denom

            # Error propagation: Var(f(x,y)) approx (df/dx*err_x)^2 + (df/dy*err_y)^2
            # f = (x^2 + y^2)/vol**2 where x=real(a), y=imag(a), err_x=da_r, err_y=da_i
            # Var(f) = ( (2x*da_r)^2 + (2y*da_i)^2 ) / vol**4
            gWP_part2_var = (
                (2 * int_omega_dIomega_mean_real * int_omega_dIomega_std_error_real)**2 +
                (2 * int_omega_dIomega_mean_imag * int_omega_dIomega_std_error_imag)**2
            ) / vol_denom**2
            # Split variance into pseudo-real/imag std errors for quadrature sum later
            # This split is somewhat arbitrary; assigning half the variance to each.
            # A more rigorous approach might be needed depending on the application.
            # However, let's follow the original code's structure of combining real/imag errors separately.
            # We need gWP_part2_std_error_real and gWP_part2_std_error_imag.
            # Let's assume the error contributes equally to real/imag variance for simplicity,
            # or use the magnitude of the std error.
            # Using magnitude:
            gWP_part2_std_error_mag = jnp.sqrt(gWP_part2_var)
            # Approximate real/imag std errors by projecting magnitude onto real/imag axes? No, that doesn't make sense.
            # Let's calculate the std error of the real and imag parts of gWP_part2 = 0.
            # gWP_part2 is purely real. So gWP_part2_std_error_imag = 0.
            gWP_part2_std_error_real = jnp.sqrt(gWP_part2_var)
            gWP_part2_std_error_imag = jnp.zeros_like(gWP_part2_std_error_real)

        else:
            # gWP_part2_IJ = a_I * conj(a_J) / vol**2
            gWP_part2 = jnp.einsum('I,J->IJ', int_omega_dIomega, jnp.conjugate(int_omega_dIomega)) / vol_denom

            # Error propagation for real part: Var( (a_Ir*a_Jr + a_Ii*a_Ji) / vol**2 )
            # Assumes errors in I and J are independent (which they are not if I=J)
            # Assumes errors in real/imag parts are independent.
            # Var(gWP_part2_IJ_real) = ( (a_Jr*da_Ir)^2 + (a_Ir*da_Jr)^2 + (a_Ji*da_Ii)^2 + (a_Ii*da_Ji)^2 ) / vol**4
            var_real_term1 = jnp.einsum('J,I->IJ', int_omega_dIomega_mean_real**2, int_omega_dIomega_std_error_real**2) # (aJr*daIr)^2 term needs fixing, indices swapped
            var_real_term1 = jnp.einsum('J,I->IJ', int_omega_dIomega_mean_real**2, int_omega_dIomega_std_error_real**2) # (aJr*daIr)^2 -> einsum('J,I->IJ', a_Jr^2, da_Ir^2) ? No.
            # Let's use the original code's propagation logic, adapted for our std errors
            # Original used: sqrt( einsum( (2*|a|*da_r)^2, 1) + einsum( (2*|a|*da_r)^2, 1).T ) / vol**2 / 2
            # This seems incorrect or at least hard to interpret.

            # Let's recalculate variance propagation for f_IJ = a_I * conj(a_J)
            # f_IJ_real = a_Ir*a_Jr + a_Ii*a_Ji
            # f_IJ_imag = a_Ii*a_Jr - a_Ir*a_Ji
            # Var(f_IJ_real) = (d(f_IJ_real)/da_Ir * da_Ir)^2 + (d(f_IJ_real)/da_Ii * da_Ii)^2 + (d(f_IJ_real)/da_Jr * da_Jr)^2 + (d(f_IJ_real)/da_Ji * da_Ji)^2
            # Var(f_IJ_real) = (a_Jr * da_Ir)^2 + (a_Ji * da_Ii)^2 + (a_Ir * da_Jr)^2 + (a_Ii * da_Ji)^2
            gWP_part2_var_real = (
                jnp.einsum('J,I->IJ', int_omega_dIomega_mean_real**2, int_omega_dIomega_std_error_real**2) + # (aJr*daIr)^2
                jnp.einsum('J,I->IJ', int_omega_dIomega_mean_imag**2, int_omega_dIomega_std_error_imag**2) + # (aJi*daIi)^2
                jnp.einsum('I,J->IJ', int_omega_dIomega_mean_real**2, int_omega_dIomega_std_error_real**2) + # (aIr*daJr)^2
                jnp.einsum('I,J->IJ', int_omega_dIomega_mean_imag**2, int_omega_dIomega_std_error_imag**2)   # (aIi*daJi)^2
            ) / vol_denom**2

            # Var(f_IJ_imag) = (d(f_IJ_imag)/da_Ir * da_Ir)^2 + (d(f_IJ_imag)/da_Ii * da_Ii)^2 + (d(f_IJ_imag)/da_Jr * da_Jr)^2 + (d(f_IJ_imag)/da_Ji * da_Ji)^2
            # Var(f_IJ_imag) = (-a_Ji * da_Ir)^2 + (a_Jr * da_Ii)^2 + (a_Ii * da_Jr)^2 + (-a_Ir * da_Ji)^2
            gWP_part2_var_imag = (
                jnp.einsum('J,I->IJ', int_omega_dIomega_mean_imag**2, int_omega_dIomega_std_error_real**2) + # (aJi*daIr)^2
                jnp.einsum('J,I->IJ', int_omega_dIomega_mean_real**2, int_omega_dIomega_std_error_imag**2) + # (aJr*daIi)^2
                jnp.einsum('I,J->IJ', int_omega_dIomega_mean_imag**2, int_omega_dIomega_std_error_real**2) + # (aIi*daJr)^2 - Error here, should be da_Jr
                jnp.einsum('I,J->IJ', int_omega_dIomega_mean_imag**2, int_omega_dIomega_std_error_real**2) + # (aIi * da_Jr)^2 -> einsum('I,J->IJ', a_Ii^2, da_Jr^2)
                jnp.einsum('I,J->IJ', int_omega_dIomega_mean_real**2, int_omega_dIomega_std_error_imag**2)   # (aIr*daJi)^2
            ) / vol_denom**2


            gWP_part2_std_error_real = jnp.sqrt(gWP_part2_var_real)
            gWP_part2_std_error_imag = jnp.sqrt(gWP_part2_var_imag)


        # Final gWP
        gWP = gWP_part1 + gWP_part2

        # Combine errors in quadrature (real and imaginary separately)
        gWP_std_error_real = jnp.sqrt(gWP_part1_std_error_real**2 + gWP_part2_std_error_real**2)
        gWP_std_error_imag = jnp.sqrt(gWP_part1_std_error_imag**2 + gWP_part2_std_error_imag**2)

        # Final complex standard error
        gWP_std_error = gWP_std_error_real + 1j * gWP_std_error_imag

        return gWP, gWP_std_error

    def gwP(self, points, weights, pullbacks, indices = None, fullmatrix = False):
        if indices is None:
            indices = PointGenerator._find_max_dQ_coords_jax(jnp.array(points, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128))
        #return gwP_jax(points, self.BASIS['DQDZB0'], self.BASIS['DQDZF0'], self.kmoduli, self.BASIS['DI_DQB0'], self.BASIS['DI_DQF0'], self.BASIS['DI_DQZB0'], self.BASIS['DI_DQZF0'], weights,pullbacks, indices = indices, fullmatrix = fullmatrix)
        return WP.gwP_jax(jnp.array(points, dtype = jnp.complex128), jnp.array(weights, dtype = jnp.float64), jnp.array(pullbacks, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), 
                       jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128), jnp.array(self.kmoduli, dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQB0'], dtype = jnp.complex128),
                         jnp.array(self.BASIS['DI_DQF0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZF0'], dtype = jnp.complex128), 
                         indices = indices, fullmatrix = fullmatrix)

    def gwP_batched(self, points, weights, pullbacks, indices = None, fullmatrix = False, batch_size = 1000, num_batches = None):
        nmoduli = len(self.BASIS['DI_DQF0'])
        if indices is None:
            indices = PointGenerator._find_max_dQ_coords_jax(jnp.array(points, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128))
        return WP.gwP_jax_batched(jnp.array(points, dtype = jnp.complex128), jnp.array(weights, dtype = jnp.float64), jnp.array(pullbacks, dtype = jnp.complex128), jnp.array(self.BASIS['DQDZB0'],
                     dtype = jnp.complex128), jnp.array(self.BASIS['DQDZF0'], dtype = jnp.complex128), jnp.array(self.kmoduli, dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQB0'], dtype = jnp.complex128),
                       jnp.array(self.BASIS['DI_DQF0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZB0'], dtype = jnp.complex128), jnp.array(self.BASIS['DI_DQZF0'], dtype = jnp.complex128),
                         indices = indices, fullmatrix = fullmatrix, batch_size = batch_size, num_batches = num_batches, nmoduli = nmoduli)

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
