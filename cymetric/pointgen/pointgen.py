"""
Main PointGenerator module.

:Authors:
    Fabian Ruehle <fabian.ruehle@cern.ch> and 
    Robin Schneider <robin.schneider@physics.uu.se>
"""
import numpy as np
import logging
import sympy as sp
from cymetric.pointgen.nphelper import prepare_basis_pickle, prepare_dataset, get_levicivita_tensor
from sympy.geometry.util import idiff
from joblib import Parallel, delayed
import itertools
from cymetric.config import real_dtype, complex_dtype
import jax.numpy as jnp
from jax import jit as jax_jit
from jax import vmap as jax_vmap
from functools import partial
import gc
from jax import config as jaxconfig
import jax

jaxconfig.update("jax_enable_x64", True)


logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('pointgen')


class PointGenerator:
    r"""The PointGenerator class.

    The numerics are entirely done in numpy; sympy is used for taking 
    (implicit) derivatives.

    Use this one if you want to generate points and data on a CY given by
    one hypersurface.
    
    All other PointGenerators inherit from this class.

    Example:
        We consider the Fermat quintic given by

        .. math::

            Q(z) = z_1^5 + z_2^5 + z_3^5 + z_4^5 + z_5^5

        and set it up with:

        >>> import numpy as np
        >>> from cymetric.pointgen.pointgen import PointGenerator
        >>> monomials = 5*np.eye(5, dtype=int)
        >>> coefficients = np.ones(5)
        >>> kmoduli = np.ones(1)
        >>> ambient = np.array([4])
        >>> pg = PointGenerator(monomials, coefficients, kmoduli, ambient)

        Once the PointGenerator is initialized you can generate a training
        dataset with 

        >>> pg.prepare_dataset(number_of_points, dir_name) 

        and prepare the required tensorflow model data with 
        
        >>> pg.prepare_basis(dir_name)
    """

    def __init__(self, monomials, coefficients, kmoduli, ambient, vol_j_norm=None, verbose=2, backend='multiprocessing', use_jax=True, use_quadratic_method = False, do_multiprocessing = False, max_iter = 10, tol = 1e-20, get_moduli_space_metric=False):
        r"""The PointGenerator uses the *joblib* module to parallelize 
        computations. 

        Args:
            monomials (ndarray[(nMonomials, ncoords), int]): monomials
            coefficients (ndarray[(nMonomials)]): coefficients in front of each
                monomial.
            kmoduli (ndarray[(nProj)]): the kaehler moduli.
            ambient (ndarray[(nProj), int]): the direct product of projective
                spaces making up the ambient space.
            vol_j_norm (float, optional): Normalization of the volume of the
                Calabi-Yau X as computed from

                .. math:: \int_X J^n \; \text{ at } \; t_1=t_2=...=t_n = 1.

                Defaults to None, in which case the normalization will be computed automatically from the intersection numbers.
            verbose (int, optional): Controls logging. 1-Debug, 2-Info,
                else Warning. Defaults to 2.
            backend (str, optional): Backend for Parallel. Defaults to
                'multiprocessing'. 'loky' makes issues with pickle5.
        """
        self.get_moduli_space_metric = get_moduli_space_metric
        if use_jax and use_quadratic_method:
            print("use_jax and use_quadratic_method cannot both be True")
            print("setting use_jax to False")
            use_jax = False
        if verbose == 1:
            level = logging.DEBUG
        elif verbose == 2:
            level = logging.INFO
        else:
            level = logging.WARNING
        logger.setLevel(level=level)
        self.use_jax = use_jax
        self.use_quadratic_method = use_quadratic_method
        self.do_multiprocessing = do_multiprocessing
        self.monomials = monomials.astype(int)
        self.coefficients = coefficients
        self.kmoduli = kmoduli
        self.ambient = ambient.astype(int)
        self.degrees = ambient + 1
        self.nhyper = 1
        self.dimCY = np.sum(self.ambient)-self.nhyper
        self.nmonomials, self.ncoords = monomials.shape
        self.nfold = np.sum(self.ambient) - self.nhyper
        self.backend = backend
        self.p_conf = np.array([[a, d] for a, d in zip(self.ambient, self.degrees)])
        self.lc = get_levicivita_tensor(int(self.nfold))
        # sympy variables
        self.x = sp.var('x0:' + str(self.ncoords))
        self.poly = sum(self.coefficients * np.multiply.reduce(
            np.power(self.x, self.monomials), axis=-1))
        # more general
        # self.c = sp.var('c0:'+str(self.nmonomials))
        # self.gpoly = sum(self.c *
        #    np.multiply.reduce(np.power(self.x, self.monomials), axis=-1))

        old_monoms = self.monomials.copy()  # make dimensions consistent with CICY case
        self.monomials = np.array([self.monomials]) 
        self.intersection_tensor = self._generate_intersection_tensor()
        self.monomials = old_monoms  # undo change
        
        self.vol_j_norm = self.get_volume_from_intersections(np.ones_like(self.kmoduli)) if vol_j_norm is None else vol_j_norm

        # some more internal variables
        self._set_seed(2021)
        self._generate_all_bases()
        if use_jax:
            import cymetric.pointgen.pointgen_jax as pointgen_jax
            self.pointgen_jax = pointgen_jax.JAXPointGenerator(self)

        if use_quadratic_method:
            if len(self.ambient) == 4 and np.all(self.ambient ==np.ones(4)):
                print("ON TETRAQUADRIC")
                from cymetric.pointgen.pointgen_jax_gmpy2 import JAXPointGeneratorQuadratic
                if max_iter==0:
                    print("using quadratic method but not using newton improvement")
                else:
                    print("using quadratic method, with max_iterations: ", max_iter, " and tol: ", tol)
                self.pointgen_jax_quadratic = JAXPointGeneratorQuadratic(self, max_iter = max_iter, tol = tol)
            elif len(self.ambient) == 2 and np.all(self.ambient ==np.ones(2)):
                print("ON TORUS")
                from cymetric.pointgen.pointgen_jax_gmpy2torus import JAXPointGeneratorQuadratic_Torus
                if max_iter==0:
                    print("using quadratic method but not using newton improvement")
                else:
                    print("using quadratic method, with max_iterations: ", max_iter, " and tol: ", tol)
                self.pointgen_jax_quadratic = JAXPointGeneratorQuadratic_Torus(self, max_iter = max_iter, tol = tol)
            else:
                raise ValueError("Quadratic method not implemented for this ambient space")
            

    @staticmethod
    def _set_seed(seed):
        # sets all seeds for point generation
        np.random.seed(seed)
        try:
            import jax.random as jrandom
        except ImportError:
            pass
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

    def _generate_all_bases(self):
        r"""This function calls a bunch of others
        which then generate various monomial bases
        needed for point generation, residue theorem
        and the pullback tensor.
        """
        self.all_ts = np.eye(len(self.ambient), dtype=int)
        #self.selected_t = self.all_ts[np.argmax(self.ambient)] # allow to generate any of them!
        self.selected_t = self.all_ts
        self._generate_root_basis()
        self._generate_dQdz_basis()
        # we disable dzdz derivatives, there is not much difference in 
        # pullback accuracy with the inverse vs implicit derivatives.
        self.dzdz_generated = False
        self._generate_padded_basis()
        if self.get_moduli_space_metric:
            self._generate_moduli_space_basis()
            self._generate_padded_dIdQZ_basis()

    def _generate_root_basis(self):
        r"""Generates monomial basis for the polynomials in 
        the free parameters which are distributed according to
        self.selected_t. This is roughly 
        
        .. math::
        
            Q(\sum_k p_j*t_jk + q_j)

        """
        self.root_vars = {}
        self.root_monomials = []
        self.root_factors = []
        
        # Initialize lists for each selected_t
        for _ in range(len(self.selected_t)):
            self.root_monomials.append([])
            self.root_factors.append([])
            
        for i, sel_t in enumerate(self.selected_t):
            self.tpoly = 0
            self.root_vars[i] = {}
            self.root_vars[i]['p'] = sp.var('p0:{}:{}'.format(
                self.ncoords, np.max(sel_t) + 1))
            self.root_vars[i]['ps'] = sp.Matrix(np.reshape(
                self.root_vars[i]['p'], (self.ncoords, np.max(sel_t) + 1)))
            self.root_vars[i]['t'] = sp.var('t0:{}'.format(self.nhyper))
            self.root_vars[i]['ts'] = sp.ones(
                int(self.ncoords), int(np.max(sel_t) + 1))
            
            for j in range(len(self.ambient)):
                for k in range(np.max(sel_t) + 1):
                    if k > sel_t[j]:
                        s = np.sum(self.ambient[:j]) + j
                        e = np.sum(self.ambient[:j + 1]) + j + 1
                        self.root_vars[i]['ps'][s:e, k] = \
                            sp.zeros(*np.shape(self.root_vars[i]['ps'][s:e, k]))
            
            j = 0
            for j_amb in range(len(self.ambient)):
                for k in range(sel_t[j_amb]):
                    s = np.sum(self.ambient[:j_amb]) + j_amb
                    e = np.sum(self.ambient[:j_amb + 1]) + j_amb + 1
                    self.root_vars[i]['ts'][s:e, 1 + k] = self.root_vars[i]['t'][j] * sp.ones(*np.shape(self.root_vars[i]['ts'][s:e, 1 + k]))
                    j += 1
                    
            self.tpoly = self.poly.subs(
                [(self.x[idx], sum(self.root_vars[i]['ps'].row(idx) * self.root_vars[i]['ts'].row(idx).T))
                 for idx in range(self.ncoords)]).as_poly()
            
            poly_dict = self.tpoly.as_dict()
            all_vars = np.array(list(self.root_vars[i]['p']) + list(self.root_vars[i]['t']))
            root_monomials = np.zeros((len(poly_dict), len(all_vars)), dtype=int)
            root_factors = np.zeros(len(poly_dict), dtype=np.complex128)
            mask = np.logical_or.reduce(all_vars == np.array(list(self.tpoly.free_symbols)).reshape(-1, 1))
            
            for idx, entry in enumerate(poly_dict):
                antry = np.array(entry)
                root_monomials[idx, mask] = antry
                root_factors[idx] = poly_dict[entry]
                
            # sort root_monomials to work with np.root
            t_mask = self.root_vars[i]['t'] == all_vars
            t_index = np.where(t_mask)[0][0]
            # +1 because hypersurface
            max_degree = int(self.ambient[sel_t.astype(bool)][0]) + 1
            # +1 for degree zero
            for j in range(max_degree + 1):
                good = root_monomials[:, t_index] == max_degree - j
                tmp_monomials = root_monomials[good]
                self.root_monomials[i].append(np.delete(tmp_monomials, t_index, axis=1))
                self.root_factors[i].append(root_factors[good])


    def _generate_root_basis_Q(self):
        r"""Generates a monomial basis for the 1d poly in t
        coming from :math:`Q(p \cdot t + q)`. 

        NOTE: this one is legacy code.
        """
        raise ValueError("This function is not used anymore. Also, doesn't have the selected_t_val argument.")
        p = sp.var('p0:{}'.format(self.ncoords))
        q = sp.var('q0:{}'.format(self.ncoords))
        t = sp.var('t')
        # change here to self.gpoly
        poly_p = self.poly.subs([(self.x[i], p[i] * t + q[i])
                                 for i in range(self.ncoords)]).as_poly()
        poly_dict = poly_p.as_dict()
        p_monomials = np.zeros((len(poly_dict), self.ncoords), dtype=int)
        q_monomials = np.zeros((len(poly_dict), self.ncoords), dtype=int)
        factors = np.zeros(len(poly_dict), dtype=np.complex128)
        for i, entry in enumerate(poly_dict):
            p_monomials[i, :] = entry[0:self.ncoords]
            q_monomials[i, :] = entry[self.ncoords:2 * self.ncoords]
            factors[i] = poly_dict[entry]
        self.root_monomials_Q = []
        self.root_factors_Q = []
        for i in range(self.ncoords + 1):
            sums = np.sum(p_monomials, axis=-1)
            indi = np.where(sums == self.ncoords - i)[0]
            self.root_monomials_Q += [(p_monomials[indi], q_monomials[indi])]
            self.root_factors_Q += [factors[indi]]

    # def _generate_dQdz_basis(self):
    #     r"""Generates a monomial basis for dQ/dz_j."""
    #     self.dQdz_basis = []
    #     self.dQdz_factors = []
    #     for i, m in enumerate(np.eye(self.ncoords, dtype=int)):
    #         basis = self.monomials - m
    #         factors = self.monomials[:, i] * self.coefficients
    #         good = np.ones(len(basis), dtype=bool)
    #         good[np.where(basis < 0)[0]] = False
    #         self.dQdz_basis += [basis[good]]
    #         self.dQdz_factors += [factors[good]]

    def _generate_dQdz_basis(self):
        r"""Generates a monomial basis for dQ/dz_j."""
        self.dQdz_basis = []
        self.dQdz_factors = []
        if self.get_moduli_space_metric: 
            self.d2Qdz2_basis = []
            self.d2Qdz2_factors = []
            self.dI_DQZbasis = []
            self.dI_DQZfactors = []
            n_moduli_directions = len(self.moduli_space_directions)
        
        for i, m in enumerate(np.eye(self.ncoords, dtype=int)):
            # First derivatives
            basis = self.monomials - m
            factors = self.monomials[:, i] * self.coefficients
            good = np.ones(len(basis), dtype=bool)
            good[np.where(basis < 0)[0]] = False
            self.dQdz_basis += [basis[good]]
            self.dQdz_factors += [factors[good]]
            if self.get_moduli_space_metric: 
                # Second derivatives (same j twice)
                basis2 = basis - m
                factors2 = factors * (self.monomials[:, i] - 1)
                good2 = np.ones(len(basis2), dtype=bool)
                good2[np.where(basis2 < 0)[0]] = False
                self.d2Qdz2_basis += [basis2[good2]]
                self.d2Qdz2_factors += [factors2[good2]]
        if self.get_moduli_space_metric:
            """then, to evaluate, we need to sum over the 2nd axis of dI_DQAbasis/factors.
                #  The first axis is the direction in moduli space, the second is the z coordinate,
                #  the third axis is the individual monomial contibution (so 4th is the powers of each z).
                #  As usual, sum over third and exp prod over 4th"""
            # Create a mask for moduli space directions to consider
            moduli_mask = np.any(self.moduli_space_directions != 0, axis=0)#i.e. if there is a non-zero entry in any of the directions.
            
            self.DI_DQB0 = []
            self.DI_DQF0 = []
            for i in range(n_moduli_directions):
                self.DI_DQB0.append(self.monomials[moduli_mask])
                self.DI_DQF0.append(self.moduli_space_directions[i][moduli_mask])
                dI_DQZone = []
                dI_DQZfactor = []
                for j, m in enumerate(np.eye(self.ncoords, dtype=int)):
                    # First derivatives
                    basis = self.monomials[moduli_mask] - m
                    factors = self.monomials[moduli_mask, j] * self.moduli_space_directions[i][moduli_mask]  # we actually don't want the coeffs here
                    good = np.ones(len(basis), dtype=bool)
                    good[np.where(basis < 0)[0]] = False
                    dI_DQZone += [basis[good]]
                    dI_DQZfactor += [factors[good]]
                self.dI_DQZbasis += [dI_DQZone]
                self.dI_DQZfactors += [dI_DQZfactor]

    def _generate_dzdz_basis(self, nproc=-1):
        r"""Generates a monomial basis for dz_i/dz_j
        which was needed for the pullback tensor.
        """
        self.dzdz_basis = [[([], []) for _ in range(self.ncoords)]
                           for _ in range(self.ncoords)]
        self.dzdz_factor = [[([0], [0]) for _ in range(self.ncoords)]
                            for _ in range(self.ncoords)]
        # take implicit derivatives
        self.iderivatives = [Parallel(n_jobs=nproc, backend=self.backend)
                             (delayed(self._implicit_diff)(i, j)
                              for i in range(self.ncoords))
                             for j in range(self.ncoords)]
        for j in range(self.ncoords):
            for i in range(self.ncoords):
                if i != j:
                    self.dzdz_basis[j][i], self.dzdz_factor[j][i] = \
                        self._frac_to_monomials(self.iderivatives[j][i])

    def _generate_intersection_tensor(self):
        if self.nfold == 1:
            get_int = self._dr
        elif self.nfold == 2:
            get_int = self._drs
        elif self.nfold == 3:
            get_int = self._drst
        elif self.nfold == 4:
            get_int = self._drstu
        elif self.nfold > 4:
            raise NotImplementedError("Computation of intersection numbers is not supported for {}-folds".format(self.nfold))

        comb = itertools.combinations_with_replacement(range(len(self.kmoduli)), int(self.nfold))
        d = np.zeros([len(self.kmoduli)] * int(self.nfold), dtype=int)
        for x in comb:
            d_int = get_int(*x)
            entries = itertools.permutations(x, int(self.nfold))
            # there will be some redundant elements, but they will only have to be calculated once.
            for b in entries:
                d[b] = d_int
        return d

    def _dr(self, r):
        r"""
        Determines the intersection number d_r.
        We use:
        .. math::
            \begin{align}
             d_{r} = \int_X J_r = \int_A \mu \wedge J_r
            \end{align}
        where \mu is the top form

        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}
        Parameters
        ----------
        r : int
            index r.

        Returns
        -------
        dr: float
            Returns the intersection number dr.

        Example
        -------
        >>> M = CICY([[2,3]])
        >>> M.drst(1)
        3.0
        """
        dr = 0
        i = 0
        combination, count = np.zeros(len(self.monomials), dtype=int), np.zeros(len(self.kmoduli), dtype=int)
        # now we want to fill combination and run over all m Projective spaces, and how often they occur
        for j in range(len(self.kmoduli)):
            if j == r:
                count[j] = self.p_conf[j][0] - 1
                combination[i:i + count[j]] = j
                i += self.p_conf[j][0] - 1
            else:
                count[j] = self.p_conf[j][0]
                combination[i:i + count[j]] = j
                i += self.p_conf[j][0]
        mu = sp.utilities.iterables.multiset_permutations(combination)
        for a in mu:
            v = 1
            for j in range(len(self.monomials)):
                if self.p_conf[a[j]][j + 1] == 0:
                    v = 0
                    break
                else:
                    v *= self.p_conf[a[j]][j + 1]
            dr += v
        return float(dr)

    def _drs(self, r, s):
        r"""
        Determines the intersection number d_rs.
        We use:
        .. math::
            \begin{align}
             d_{rs} = \int_X J_r \wedge J_s = \int_A \mu \wedge J_r \wedge J_s
            \end{align}
        where \mu is the top form

        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}
        Parameters
        ----------
        r : int
            index r.
        s : int
            index s.

        Returns
        -------
        drs: float
            Returns the intersection number drs.

        Example
        -------
        >>> M = CICY([[3,4]])
        >>> M.drst(0)
        4.0
        """
        drs = 0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination, count = np.zeros(len(self.monomials), dtype=int), np.zeros(len(self.kmoduli), dtype=int)
        # now there are 2 distinct cases:
        # 1) r=s or 2) r != s
        # 1)
        if r == s:
            if self.p_conf[r][0] < 2:
                # then drs is zero
                return 0
            else:
                i = 0
                # now we want to fill combination and run over all m Projective spaces,
                # and how often they occur
                for j in range(len(self.kmoduli)):
                    if j == r:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        count[j] = self.p_conf[j][0]
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0]
        # 2)
        else:
            i = 0
            for j in range(len(self.kmoduli)):
                if j == r or j == s:
                    count[j] = self.p_conf[j][0] - 1
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0] - 1
                else:
                    count[j] = self.p_conf[j][0]
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0]

        mu = sp.utilities.iterables.multiset_permutations(combination)
        for a in mu:
            v = 1
            for j in range(len(self.monomials)):
                if self.p_conf[a[j]][j + 1] == 0:
                    v = 0
                    break
                else:
                    v *= self.p_conf[a[j]][j + 1]
            drs += v
        return drs

    def _drst(self, r, s, t):
        r"""
        Determines the triple intersection number d_rst.
        We use:
        .. math::
            \begin{align}
             d_{rst} = \int_X J_r \wedge J_s \wedge J_t = \int_A \mu \wedge J_r \wedge J_s \wedge J_t
            \end{align}
        where \mu is the top form

        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}
        Parameters
        ----------
        r : int
            index r.
        s : int
            index s.
        t : int
            index t.

        Returns
        -------
        drst: float
            Returns the triple intersection number drst.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.drst(0,1,1)
        7.0
        """
        drst = 0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination = np.array([0 for _ in range(len(self.monomials))])
        count = [0 for _ in range(len(self.kmoduli))]
        # now there are 5 distinct cases:
        # 1) r=s=t or 2) all neqal or the 2-5) three cases where two are equal
        # 1)
        if r == s == t:
            if self.p_conf[r][0] < 3:
                # then drst is zero
                return 0
            else:
                i = 0
                # now we want to fill combination and run over all m Projective spaces,
                # and how often they occur
                for j in range(len(self.kmoduli)):
                    if j == r:
                        # we obviously have to subtract 3 in the case of three
                        # times the same index since we already have three kähler forms
                        # in Ambient space coming from the intersection number
                        count[j] = self.p_conf[j][0] - 3
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 3
                    else:
                        count[j] = self.p_conf[j][0]
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0]
        # 2)
        if r != s and r != t and s != t:
            i = 0
            for j in range(len(self.kmoduli)):
                if j == r or j == s or j == t:
                    count[j] = self.p_conf[j][0] - 1
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0] - 1
                else:
                    count[j] = self.p_conf[j][0]
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0]
        # 3)
        if r == s and r != t:
            if self.p_conf[r][0] < 2:
                return 0
            else:
                i = 0
                for j in range(len(self.kmoduli)):
                    if j == r:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        if j == t:
                            count[j] = self.p_conf[j][0] - 1
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0] - 1
                        else:
                            count[j] = self.p_conf[j][0]
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0]
        # 4)
        if r == t and r != s:
            i = 0
            if self.p_conf[r][0] < 2:
                return 0
            else:
                i = 0
                for j in range(len(self.kmoduli)):
                    if j == r:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        if j == s:
                            count[j] = self.p_conf[j][0] - 1
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0] - 1
                        else:
                            count[j] = self.p_conf[j][0]
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0]
        # 5)
        if s == t and s != r:
            i = 0
            if self.p_conf[s][0] < 2:
                return 0
            else:
                i = 0
                for j in range(len(self.kmoduli)):
                    if j == s:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        if j == r:
                            count[j] = self.p_conf[j][0] - 1
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0] - 1
                        else:
                            count[j] = self.p_conf[j][0]
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0]
        # the combinations of mu grow exponentially with len(self.monomials) and the number of ambient spaces
        # Check, when the number of multiset_permutations become to large to handle
        if len(self.monomials) < 8 and len(np.unique(combination)) < 6:
            # Hence, for large K and small len(self.kmoduli), this might take really long.
            mu = sp.utilities.iterables.multiset_permutations(combination)
            # (len(self.monomials))!/(#x_1!*...*#x_n!)
            for a in mu:
                v = 1
                for j in range(len(self.monomials)):
                    if self.p_conf[a[j]][j + 1] == 0:
                        v = 0
                        break
                    else:
                        v *= self.p_conf[a[j]][j + 1]
                drst += v
            return drst
        else:
            # here we calculate the nonzero paths through the CICY
            # much faster since CICYs with large K and large len(self.kmoduli) tend to
            # be pretty sparse
            nonzero = [[] for _ in range(len(self.monomials))]
            combination = np.sort(combination)
            count_2 = [0 for _ in range(len(self.kmoduli))]
            # run over all K to find possible paths
            for i in range(len(self.monomials)):
                for j in range(len(self.kmoduli)):
                    # possible paths are non zero and in combination
                    if self.p_conf[j][i + 1] != 0 and j in combination:
                        nonzero[i] += [j]
                        count_2[j] += 1
            # Next we run over all entries in count to see if any are fixed by number of occurence
            for i in range(len(self.kmoduli)):
                if count[i] == count_2[i]:
                    # if equal we run over all entries in nonzero
                    # count[i] = 0
                    for j in range(len(self.monomials)):
                        # and fix them to i if they contain it
                        if i in nonzero[j]:
                            # and len(nonzero[j]) != 1
                            nonzero[j] = [i]
            # There are some improvements here:
            # 1) take the counts -= 1 if fixed and compare if the left allowed
            # 2) here it would be even more efficient to write a product that respects
            #   the allowed combinations from count.
            mu = itertools.product(*nonzero)
            # len(nonzero[0])*...*len(nonzero[K])
            # since we in principle know the complexity of both calculations
            # one could also do all the stuff before and then decide which way is faster
            for a in mu:
                # if allowed by count
                c = list(a)
                if np.array_equal(np.sort(c), combination):
                    v = 1
                    for j in range(len(self.monomials)):
                        if self.p_conf[c[j]][j + 1] == 0:
                            break
                        else:
                            v *= self.p_conf[c[j]][j + 1]
                    drst += v
            return drst

    def _drstu(self, r, s, t, u):
        r"""
        Determines the quadruple intersection numbers, d_rstu, for Calabi Yau 4-folds.

        Parameters
        ----------
        r : int
            the index r.
        s : int
            the index s.
        t : int
            the index t.
        u : int
            the index u.

        Returns
        -------
        drstu: float
            The quadruple intersection number d_rstu.

        Example
        -------
        >>> M = CICY([[2,3],[2,3],[1,2]])
        >>> M.drstu(0,1,1,2)
        3
        References
        ----------
        .. [1] All CICY four-folds, by J. Gray, A. Haupt and A. Lukas.
            https://arxiv.org/pdf/1303.1832.pdf
        """

        if self.nfold != 4:
            logger.warning('CICY is not a 4-fold.')

        drstu = 0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination = np.array([0 for _ in range(len(self.monomials))])
        count = [0 for _ in range(len(self.kmoduli))]
        # now there are 5 distinct cases:
        # 1) r=s=t=u or 2) all neqal or the 3) two equal, two nonequal
        # 4) two equal and two equal 5) three equal
        un, unc = np.unique([r, s, t, u], return_counts=True)
        for i in range(len(un)):
            if self.p_conf[un[i]][0] < unc[i]:
                return 0
        i = 0
        for j in range(len(self.kmoduli)):
            # if j in rstu subtract
            # else go full
            contained = False
            for a in range(len(un)):
                if j == un[a]:
                    contained = True
                    count[j] = self.p_conf[j][0] - unc[a]
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0] - unc[a]
            if not contained:
                count[j] = self.p_conf[j][0]
                combination[i:i + count[j]] = j
                i += self.p_conf[j][0]
        # just copy from drst
        # the combinations of mu grow exponentially with len(self.monomials) and the number of ambient spaces
        # Check, when the number of multiset_permutations become to large to handle
        if len(self.monomials) < 8 and len(np.unique(combination)) < 6:
            # Hence, for large K and small, this might take really long.
            mu = sp.utilities.iterables.multiset_permutations(combination)
            # (len(self.monomials))!/(#x_1!*...*#x_n!)
            for a in mu:
                v = 1
                for j in range(len(self.monomials)):
                    if self.p_conf[a[j]][j + 1] == 0:
                        v = 0
                        break
                    else:
                        v *= self.p_conf[a[j]][j + 1]
                drstu += v
            return drstu
        else:
            # here we calculate the nonzero paths through the CICY
            nonzero = [[] for _ in range(len(self.monomials))]
            combination = np.sort(combination)
            count_2 = [0 for _ in range(len(self.kmoduli))]
            # run over all len(self.monomials) to find possible paths
            for i in range(len(self.monomials)):
                for j in range(len(self.kmoduli)):
                    # possible paths are non zero and in combination
                    if self.p_conf[j][i + 1] != 0 and j in combination:
                        nonzero[i] += [j]
                        count_2[j] += 1
            # Next we run over all entries in count to see if any are fixed by number of occurence
            for i in range(len(self.kmoduli)):
                if count[i] == count_2[i]:
                    # if equal we run over all entries in nonzero
                    # count[i] = 0
                    for j in range(len(self.monomials)):
                        # and fix them to i if they contain it
                        if i in nonzero[j]:
                            # and len(nonzero[j]) != 1
                            nonzero[j] = [i]
            # There are some improvements here:
            # 1) take the counts -= 1 if fixed and compare if the left allowed
            # 2) here it would be even more efficient to write a product that respects
            #   the allowed combinations from count, but I can't be bothered to do it atm.
            mu = itertools.product(*nonzero)
            # len(nonzero[0])*...*len(nonzero[K])
            # since we in principle know the complexity here and from the other
            # one should also do all the stuff before and then decide which way is faster
            for a in mu:
                # if allowed by count
                c = list(a)
                if np.array_equal(np.sort(c), combination):
                    v = 1
                    for j in range(len(self.monomials)):
                        if self.p_conf[c[j]][j + 1] == 0:
                            break
                        else:
                            v *= self.p_conf[c[j]][j + 1]
                    drstu += v
            return drstu

    def get_volume_from_intersections(self, ts):
        if self.nfold == 1:
            vol = np.einsum("a,a", self.intersection_tensor, ts)
        elif self.nfold == 2:
            vol = np.einsum("ab,a,b", self.intersection_tensor, ts, ts)
        elif self.nfold == 3:
            vol = np.einsum("abc,a,b,c", self.intersection_tensor, ts, ts, ts)
        elif self.nfold == 4:
            vol = np.einsum("abcd,a,b,c,d", self.intersection_tensor, ts, ts, ts, ts)
        else:
            raise NotImplementedError("Computation of intersection numbers is not supported for {}-folds".format(self.nfold))
        return vol
        
    def _implicit_diff(self, i, j):
        r"""Compute the implicit derivative of

        .. math:: dz_i/dz_j

        given the defining CY equation in the sympy polynomial 'self.poly'.

        Args:
            i (int): i index
            j (int): j index

        Returns:
            sympy poly: implicit derivative
        """
        return idiff(self.poly, self.x[i], self.x[j]) if i != j else 0

    def _frac_to_monomials(self, frac):
        r"""Takes a sympy fraction and returns tuples of
        monomials and coefficients for numerator and
        denominator.

        Args:
            frac (sympy.expr): sympy fraction

        Returns:
            ((num_basis, denom_basis), (num_factor, denom_factor))
        """
        num, den = frac.as_numer_denom()
        num_free, den_free = num.free_symbols, den.free_symbols

        # polys as dict
        num = num.as_poly().as_dict()
        den = den.as_poly().as_dict()

        # coordinate mask
        num_mask = [True if self.x[i] in num_free else False for i in range(self.ncoords)]
        den_mask = [True if self.x[i] in den_free else False for i in range(self.ncoords)]

        # initialize output
        num_monomials = np.zeros((len(num), self.ncoords), dtype=int)
        denmonomials = np.zeros((len(den), self.ncoords), dtype=int)
        num_factor = np.zeros(len(num), dtype=np.complex128)
        den_factor = np.zeros(len(den), dtype=np.complex128)

        # fill monomials and factors
        for i, entry in enumerate(num):
            num_monomials[i, num_mask] = entry
            num_factor[i] = num[entry]
        for i, entry in enumerate(den):
            denmonomials[i, den_mask] = entry
            den_factor[i] = den[entry]

        return ((num_monomials, denmonomials), (num_factor, den_factor))

    def _generate_padded_basis(self):
        r"""Generates a padded basis, i.e. padds the monomials in dQdz (and
        dzdz) with zeros at the end if they have uneven length to allow for
        vectorized computations.
        """
        self.BASIS = {}
        shape = np.array([np.shape(mb) for mb in self.dQdz_basis])
        DQDZB = np.zeros((len(shape), np.max(shape[:, 0]), len(shape)), dtype=np.complex128)
        DQDZF = np.zeros((len(shape), np.max(shape[:, 0])), dtype=np.complex128)
        for i, m in enumerate(zip(self.dQdz_basis, self.dQdz_factors)):
            DQDZB[i, 0:shape[i, 0]] += m[0]
            DQDZF[i, 0:shape[i, 0]] += m[1]
        if self.dzdz_generated:
            shapes = np.array([[[np.shape(t[0]), np.shape(t[1])]
                                if i != j else [[-1, -1], [-1, -1]] for i, t in enumerate(zi)]
                               for j, zi in enumerate(self.dzdz_basis)])
            DZDZB_d = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 0, 0]), len(shapes)), dtype=int)
            DZDZB_n = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 1, 0]), len(shapes)), dtype=int)
            DZDZF_d = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 0, 0])), dtype=np.complex128)
            DZDZF_n = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 1, 0])), dtype=np.complex128)
            for i in range(len(shapes)):
                for j in range(len(shapes)):
                    if i != j:
                        DZDZB_d[i, j, 0:shapes[i, j, 0, 0]] += self.dzdz_basis[i][j][0]
                        DZDZB_n[i, j, 0:shapes[i, j, 1, 0]] += self.dzdz_basis[i][j][1]
                        DZDZF_d[i, j, 0:shapes[i, j, 0, 0]] += self.dzdz_factor[i][j][0]
                        DZDZF_n[i, j, 0:shapes[i, j, 1, 0]] += self.dzdz_factor[i][j][1]
            self.BASIS['DZDZB_d0'] = DZDZB_d
            self.BASIS['DZDZB_n0'] = DZDZB_n
            self.BASIS['DZDZF_d0'] = DZDZF_d
            self.BASIS['DZDZF_n0'] = DZDZF_n
        self.BASIS['DQDZB0'] = DQDZB
        self.BASIS['DQDZF0'] = DQDZF
        self.BASIS['QB0'] = self.monomials
        self.BASIS['QF0'] = self.coefficients

    def _generate_moduli_space_basis(self):
        shape = np.array([np.shape(mb) for mb in self.d2Qdz2_basis])
        D2QDZ2B = np.zeros((len(shape), np.max(shape[:, 0]), len(shape)), dtype=np.complex128)
        D2QDZ2F = np.zeros((len(shape), np.max(shape[:, 0])), dtype=np.complex128)
        for i, m in enumerate(zip(self.d2Qdz2_basis, self.d2Qdz2_factors)):
            D2QDZ2B[i, 0:shape[i, 0]] += m[0]
            D2QDZ2F[i, 0:shape[i, 0]] += m[1]
        self.BASIS['D2QDZ2B0'] = D2QDZ2B
        self.BASIS['D2QDZ2F0'] = D2QDZ2F

        #DI_DQZB
        #n_moduli_directions = len(self.n_moduli_directions_d)
        #for
    def _generate_padded_dIdQZ_basis(self):
        """Generates padded basis for moduli space derivatives.
        
        The structure is:
        - First axis: direction in moduli space
        - Second axis: z coordinate
        - Third axis: individual monomial contribution
        - Fourth axis: powers of each z
        """
        n_moduli_directions = len(self.moduli_space_directions)
        
        # Find maximum shape across all directions and coordinates
        max_shape = 0
        for direction_basis in self.dI_DQZbasis:
            for coord_basis in direction_basis:
                if len(coord_basis) > max_shape:
                    max_shape = len(coord_basis)
        
        # Create padded arrays
        DI_DQZB = np.zeros((n_moduli_directions, self.ncoords, max_shape, self.ncoords), dtype=np.complex128)
        DI_DQZF = np.zeros((n_moduli_directions, self.ncoords, max_shape), dtype=np.complex128)
        max_shape_2 = max(len(self.DI_DQB0[i]) for i in range(n_moduli_directions))
        DI_DQB0 = np.zeros((n_moduli_directions, max_shape_2, self.ncoords), dtype=np.complex128)
        DI_DQF0 = np.zeros((n_moduli_directions, max_shape_2), dtype=np.complex128)
        
        # Fill the padded arrays
        for i in range(n_moduli_directions):
            DI_DQB0[i] = self.DI_DQB0[i]
            DI_DQF0[i] = self.DI_DQF0[i]
            for j in range(self.ncoords):
                basis = self.dI_DQZbasis[i][j]
                factors = self.dI_DQZfactors[i][j]
                DI_DQZB[i, j, 0:len(basis)] += basis
                DI_DQZF[i, j, 0:len(factors)] += factors
        
        self.BASIS['DI_DQZB0'] = DI_DQZB
        self.BASIS['DI_DQZF0'] = DI_DQZF
        self.BASIS['DI_DQB0'] = DI_DQB0
        self.BASIS['DI_DQF0'] = DI_DQF0
    

    def generate_points_quadratic(self, n_p):
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
        print(f"using JAX and the quadratic sampling method")
        numpy_seed = np.random.get_state()[1][0]# use the same seed as numpy for the jax seed
        print("numpy seed here: ", numpy_seed)
        #points = self.pointgen_jax.generate_points_jax(n_p, numpy_seed, selected_t_val=selected_t_val)
        points = self.pointgen_jax_quadratic.generate_points_jax(n_p, numpy_seed)
        return points

    def generate_points(self, n_p, nproc=-1, batch_size=5000, use_jax=True, process_parallel=False, selected_t_val=None, use_quadratic_method = False):
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
        if use_jax:
            print(f"using JAX: solving roots with jnp.roots on P1: {selected_t_val}")
            numpy_seed = np.random.get_state()[1][0]# use the same seed as numpy for the jax seed
            points = self.pointgen_jax.generate_points_jax(n_p, numpy_seed, selected_t_val=selected_t_val)
            return points
        if use_quadratic_method:
            points = self.generate_points_quadratic(n_p)
            return points
        else:
            max_ts = np.max(self.selected_t[selected_t_val])
            max_degree = self.ambient[self.selected_t[selected_t_val].astype(bool)] + 1
            n_p_red = int(n_p / max_degree) + 1
            pn_pnts = np.zeros((n_p_red, self.ncoords, max_ts + 1),
                               dtype=np.complex128)
            for i in range(len(self.ambient)):
                for k in range(self.selected_t[selected_t_val][i] + 1):
                    s = np.sum(self.ambient[:i]) + i
                    e = np.sum(self.ambient[:i + 1]) + i + 1
                    pn_pnts[:, s:e, k] += self.generate_pn_points(n_p_red, self.ambient[i])
            if not use_jax and process_parallel:
                points = Parallel(n_jobs=nproc, backend=self.backend, batch_size=batch_size)(
                    delayed(self._take_roots)(pi, selected_t_val=selected_t_val) for pi in pn_pnts)
            elif not use_jax and not process_parallel:
                # Sequential processing without parallelization
                points = [self._take_roots(pi, selected_t_val=selected_t_val) for pi in pn_pnts]
        
            points = np.vstack(points)
            return self._rescale_points(points)

    def _generate_points_Q(self, n_p, nproc=-1, batch_size=10000):
        r"""Generates complex points using a single intersecting line 
        through *all* projective spaces. This correlates the points somewhat
        and the correct measure is currently unknown.

        NOTE: Legacy code.

        Args:
            n_p (int): # of points.
            nproc (int, optional): # of jobs used. Defaults to -1. Then
                uses all available resources.
            batch_size (int, optional): batch_size of Parallel. 
                Defaults to 10000.

        Returns:
            ndarray[(n_p, ncoords), np.complex128]: rescaled points
        """
        self._generate_root_basis_Q()
        p = np.hstack([self.generate_pn_points(int(n_p / self.ncoords) + 1, n) for n in self.ambient])
        q = np.hstack([self.generate_pn_points(int(n_p / self.ncoords) + 1, n) for n in self.ambient])

        # TODO: vectorize this nicely
        points = np.vstack(
            Parallel(n_jobs=nproc, backend=self.backend, batch_size=batch_size)(
                delayed(self._take_roots_Q)(pi, qi) for pi, qi in zip(p, q)))

        return self._rescale_points(points)

    def _rescale_points(self, points):
        r"""Rescales points in place such that for every P_i^n
        max(abs(coords_i)) == 1 + 0j.

        Args:
            points (ndarray[(n_p, ncoords), complex]): Points.

        Returns:
            ndarray[(n_p, ncoords), complex]: rescaled points
        """
        # iterate over all projective spaces and rescale in each
        for i in range(len(self.ambient)):
            s = np.sum(self.degrees[0:i])
            e = np.sum(self.degrees[0:i + 1])
            points[:, s:e] = points[:, s:e] * (points[np.arange(len(points)), s + np.argmax(np.abs(points[:, s:e]), axis=-1)].reshape((-1, 1))) ** -1
        return points

    @staticmethod
    def generate_pn_points(n_p, n):
        r"""Generates points on the sphere :math:`S^{2n+1}`.

        Args:
            n_p (int): number of points.
            n (int): degree of projective space.

        Returns:
            ndarray[(np, n+1), np.complex128]: complex points
        """
        # to not get a higher concentration from the corners of the hypercube,
        #  sample with gaussian
        points = np.random.randn(n_p, 2 * (n + 1))
        # put them on the sphere
        norm = np.expand_dims(np.linalg.norm(points, axis=-1), -1)
        # make them complex
        return (points / norm).view(dtype=np.complex128)

    def _point_from_sol(self, p, sol, selected_t_val=None):
        r"""Generates a point on the CICY.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            sol (ndarray[(nHyper), np.complex]): Complex t-values.

        Returns:
            ndarray[(ncoords), np.complex128]: Point on the (CI-)CY.
        """
        # use this over point from sol sympy >100 factor improvement
        t = np.ones_like(p)
        j = 0
        for i in range(len(self.ambient)):
            for k in range(1, self.selected_t[selected_t_val][i] + 1):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                t[s:e, k] = sol[j] * np.ones_like(t[s:e, k])
                j += 1
        point = np.sum(p * t, axis=-1)
        return point

    def _take_roots(self, p, selected_t_val=None):
        r"""We generate points on Q by defining a line p*t+q 
        in *one* of the projective ambient spaces and taking all
        the intersections with Q.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...

        Returns:
            ndarray[(nsol, ncoords), np.complex128]: all points from the 
                intersection
        """
        all_sums = [
            np.sum(c * np.multiply.reduce(np.power(p.flatten(), m), axis=-1))
            for m, c in zip(self.root_monomials[selected_t_val], self.root_factors[selected_t_val])]
        roots = np.roots(all_sums)
        # we give [t] to work with more general hypersurfaces.
        return np.array([self._point_from_sol(p, [t], selected_t_val=selected_t_val) for t in roots])

    def _take_roots_Q(self, p, q):
        r"""We generate points on Q by taking two points
        p, q \in A defining the line p*t+q and taking all
        the intersections with Q.

        Args:
            p (ndarray[(ncoords), np.complex128]): Points on spheres.
            q (ndarray[(ncoords), np.complex128]): Points on spheres.

        Returns:
            ndarray[(nsol, ncoords), np.complex128]: all points 
                from the intersection
        """
        all_sums = [
            np.sum(c * np.multiply.reduce(np.power(p, m[0]), axis=-1) *
                   np.multiply.reduce(np.power(q, m[1]), axis=-1))
            for m, c in zip(self.root_monomials_Q, self.root_factors_Q)]
        return np.array([p * t + q for t in np.roots(all_sums)])

    def generate_point_weights(self, n_pw, omega=False, normalize_to_vol_j=False, selected_t_val=None, use_quadratic_method=False, get_pullbacks=False):
        r"""Generates a numpy dictionary of point weights.

        Args:
            n_pw (int): # of point weights.
            omega (bool, optional): If True adds Omega to dict. Defaults to False.
            normalize_to_vol_j (bool, optional): Whether the weights should be normalized by the factor self.vol_j_norm.
                                                 Defaults to False

        Returns:
            np.dict: point weights
        """
        if not use_quadratic_method:
            if selected_t_val is None:
                print("NOT GIVEN SELECTED T: USING DEFAULT")
                selected_t_val = np.argmax(self.ambient)
            data_types = [
                ('point', np.complex128, self.ncoords),
                ('weight', np.float64)
            ]
            data_types = data_types + [('omega', np.complex128)] if omega else data_types
            dtype = np.dtype(data_types)
            points = self.generate_points(n_pw, selected_t_val=selected_t_val)

            #Throw away points for which the patch is ambiguous, since too many coordiantes are too close to 1
            inv_one_mask = np.isclose(points, complex(1, 0))
            bad_indices = np.where(np.sum(inv_one_mask, -1) != len(self.kmoduli))
            point_mask = np.ones(len(points), dtype=bool)
            point_mask[bad_indices] = False
            
            # Also throw away NaN points
            nan_indices = np.where(np.isnan(points).any(axis=1))
            if len(nan_indices[0]) > 0:
                print(f"Warning: Found {len(nan_indices[0])} points/({len(points)}) with NaN values. These will be discarded.")
                point_mask[nan_indices] = False
                
            points = points[point_mask]

            n_p = len(points)
            n_p = n_p if n_p < n_pw else n_pw
            weights = self.point_weight(points, normalize_to_vol_j=normalize_to_vol_j, selected_t_val=selected_t_val)
            point_weights = np.zeros((n_p), dtype=dtype)
            point_weights['point'], point_weights['weight'] = points[0:n_p], weights[0:n_p]
            if omega:
                point_weights['omega'] = self.holomorphic_volume_form(points[0:n_p])
            return point_weights
        else:
            data_types = [
                ('point', np.complex128, self.ncoords),
                ('weight', np.float64),
                ('pullbacks', np.complex128, (self.nfold,self.ncoords))
            ]
            data_types = data_types + [('omega', np.complex128)] if omega else data_types
            dtype = np.dtype(data_types)
            points = self.generate_points_quadratic(n_pw)
            #print("points: ", points)
            
            pullbacks = self.pullbacks(points)

            #Throw away points for which the patch is ambiguous, since too many coordiantes are too close to 1
            inv_one_mask = np.isclose(points, complex(1, 0))
            bad_indices = np.where(np.sum(inv_one_mask, -1) != len(self.kmoduli))
            point_mask = np.ones(len(points), dtype=bool)
            point_mask[bad_indices] = False
            points = points[point_mask]
            n_p = len(points)

            point_weights = np.zeros((n_p), dtype=dtype)
            point_weights['omega'] = self.holomorphic_volume_form(points[0:n_p])
            omegasquared = np.abs(point_weights['omega'])**2

            n_p = n_p if n_p < n_pw else n_pw
            weights = self.point_weight_quadratic(points, omegasquared, pullbacks, normalize_to_vol_j=normalize_to_vol_j)
            point_weights['point'], point_weights['weight'] = points[0:n_p], weights[0:n_p]
            point_weights['pullbacks'] = pullbacks[0:n_p]
            gc.collect()
            return point_weights

    def holomorphic_volume_form(self, points, j_elim=None, use_jax=True):
        r"""We compute the holomorphic volume form
        at all points by solving the residue theorem:

        .. math::

            \Omega &= \int_\rho \frac{1}{Q} \wedge^n dz_i \\
                   &= \frac{1}{\frac{\partial Q}{\partial z_j}}\wedge^{n-1} dz_a

        where the index a runs over the local n-fold good coordinates.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            j_elim (ndarray([n_p], int)): index to be eliminated. 
                Defaults not None. If None eliminates max(dQdz).
            use_jax (bool, optional): Whether to use JAX implementation.
                Defaults to True.

        Returns:
            ndarray[(n_p), np.complex128]: Omega evaluated at each point.
        """
        if isinstance(points, jnp.ndarray) or (isinstance(points, np.ndarray) and use_jax):
            ambient_ones = np.all(self.ambient ==np.ones_like(self.ambient))
            if not ambient_ones:
                print("warning: not adding sign for holomorphic volume form")
            return PointGenerator._holomorphic_volume_form_jax(points, jnp.squeeze(j_elim) if j_elim is not None else None, jnp.array(self.BASIS['DQDZB0']), jnp.array(self.BASIS['DQDZF0']), ambient_ones=ambient_ones)
        else:
            print(f"using legacy holomorphic volume form, {type(points)}, use_jax = {use_jax}")
            return self._holomorphic_volume_form_legacy(points, j_elim)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def _holomorphic_volume_form_jax(points, j_elim, DQDZB0, DQDZF0, ambient_ones=True):
        """JAX implementation of holomorphic volume form computation."""
        indices = PointGenerator._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0) if j_elim is None else j_elim
        omega = jnp.power(jnp.expand_dims(points, 1), DQDZB0[indices])
        omega = jnp.multiply.reduce(omega, axis=-1)
        omega = jnp.add.reduce(DQDZF0[indices] * omega, axis=-1)
        
        # Handle sign calculation
        if ambient_ones:
            inv_one_mask = jnp.logical_not(jnp.isclose(points, complex(1, 0)))
            ints_for_power = jnp.tile(jnp.expand_dims(jnp.arange(points.shape[1]), 0), (points.shape[0], 1))
            sign_for_omega2 = jnp.prod((-1)**(inv_one_mask*ints_for_power), axis=-1).astype(jnp.complex128)
            which_p1 = (indices//2)
            sign_of_omega = (-1)**which_p1 * (-1)**(indices%2+1) * sign_for_omega2 * (-1)**(indices%2+1)
        else:
            sign_of_omega = 1
            
        # compute (dQ/dzj)**-1
        return 1 / omega * sign_of_omega
    @staticmethod
    @jax_jit
    def _compute_d2q_dz2_jax(points, indices, D2QDZ2B0, D2QDZ2F0):
        """Compute second derivatives of Q with respect to z.  D2QDZ2B0 has shape(8, 27, 8)"""
        d2q_dz2 = jnp.power(jnp.expand_dims(points, 1), D2QDZ2B0[indices])
        d2q_dz2 = jnp.multiply.reduce(d2q_dz2, axis=-1)
        d2q_dz2 = jnp.add.reduce(D2QDZ2F0[indices] * d2q_dz2, axis=-1)
        return d2q_dz2
    
    @staticmethod
    @jax_jit
    def _compute_dq_dz_jax(points, indices, DQDZB0, DQDZF0):
        """Compute first derivatives of Q with respect to z.  DQDZB0 has shape (8, 54, 8)"""
        dq_dz = jnp.power(jnp.expand_dims(points, 1), DQDZB0[indices])
        dq_dz = jnp.multiply.reduce(dq_dz, axis=-1)
        dq_dz = jnp.add.reduce(DQDZF0[indices] * dq_dz, axis=-1)
        return dq_dz
    
    @staticmethod
    @jax_jit
    def _compute_dI_dQZ_jax(points, indices, DI_DQZB0, DI_DQZF0):
        """Compute derivatives of I,z with respect to Q"""
        # Expand points to match DI_DQZB0 shape which is (81, 8, 54, 8), where 81 can just be the number of directions under consideration
        #dotDI_DQZB0 = jnp.einsum('Ii,i...->I...', moduli_space_directions, DI_DQZB0)#shape I
        expanded_points = jnp.expand_dims(jnp.expand_dims(points, 0), 2)# shape 1,N,1, 8
        dI_dQZ = jnp.power(expanded_points, DI_DQZB0[:,indices])#1, N,1,8 and 81, N,54,8
        dI_dQZ = jnp.multiply.reduce(dI_dQZ, axis=-1)
        dI_dQZ = jnp.add.reduce(DI_DQZF0[:,indices] * dI_dQZ, axis=-1)
        return dI_dQZ
    
    @staticmethod
    @jax_jit
    def _compute_dIp_jax(points, DI_DQB0, DI_DQF0):
        """Compute dIp using the moduli space directions."""
        dIp = jnp.power(jnp.expand_dims(points, (0,2)), jnp.expand_dims(DI_DQB0, 1))# shape is 81, N, monoms, 8
        dIp = jnp.multiply.reduce(dIp, axis=-1) # shape is 81, N, monoms
        dIp = jnp.add.reduce(jnp.expand_dims(DI_DQF0, 1) * dIp, axis=-1) # shape is 81, N# 81 is a stand-in for the number of moduli space directions
        return dIp


    # #def get_measure_integral

    # def _measure_integral(self,points,aux_weights, j_elim=None):
    #     aux_weights = jnp.astype(aux_weights, points.dtype)
    #     d_IOmega, dq_dz = self._dI_holomorphic_volume_form_jax(points, j_elim, self.BASIS['DQDZB0'], self.BASIS['DQDZF0'], self.BASIS['DI_DQZB0'],
    #                                                             self.BASIS['DI_DQZF0'], self.BASIS['D2QDZ2B0'], self.BASIS['D2QDZ2F0'], self.BASIS['DI_DQB0'], 
    #                                                             self.BASIS['DI_DQF0'])
    #     d_JbarOmegabar = jnp.conj(d_IOmega)

    #     omega = 1/dq_dz
    #     omegabar = jnp.conj(omega)

    #     v  = jnp.mean(aux_weights*jnp.abs(d_IOmega)**2)
    #     int_dIomega_wedge_omegabar = jnp.mean(jnp.einsum('ix,x->ix',d_IOmega,aux_weights*omegabar), axis=-1)
    #     int_omega_wedge_dJbar_omegabar = jnp.mean(jnp.einsum('x,jx->jx', aux_weights*omega,d_JbarOmegabar), axis=-1)
    #     # Compute mean of d_IOmega * conj(d_JbarOmegabar) across points, for each moduli direction
    #     # Reshape aux_weights to broadcast correctly across moduli directions
    #     int_dIO_dJbarObar = jnp.mean(jnp.einsum('ip,jp,p->ijp', d_IOmega, jnp.conj(d_JbarOmegabar), aux_weights), axis=-1)
    #     result = (1/v**2)*jnp.einsum('i,j->ij', int_omega_wedge_dJbar_omegabar, int_dIomega_wedge_omegabar) - 1/v *int_dIO_dJbarObar
    #     return result




    
    # def _holomorphic_volume_form_legacy(self, points, j_elim=None):
    #     """Legacy numpy implementation of holomorphic volume form computation."""
    #     indices = self._find_max_dQ_coords(points) if j_elim is None else j_elim
    #     omega = np.power(np.expand_dims(points, 1),
    #                      self.BASIS['DQDZB0'][indices])
    #     omega = np.multiply.reduce(omega, axis=-1)
    #     omega = np.add.reduce(self.BASIS['DQDZF0'][indices] * omega, axis=-1)
    #     if np.all(self.ambient ==np.ones_like(self.ambient)):
    #         which_p1 = (indices//2)
    #         sign_of_omega = (-1)**which_p1*(-1)**(indices%2)
    #         # compute (dQ/dzj)**-1
    #         return 1 / omega*sign_of_omega
    #     else:
    #         print("warning: not adding sign for holomorphic volume form")
    #         return 1 / omega

    def _holomorphic_volume_form_legacy(self, points, j_elim=None):
        """Legacy numpy implementation of holomorphic volume form computation."""
        indices = self._find_max_dQ_coords(points) if j_elim is None else j_elim
        
        omega = np.power(np.expand_dims(points, 1),
                         self.BASIS['DQDZB0'][indices])
        omega = np.multiply.reduce(omega, axis=-1)
        omega = np.add.reduce(self.BASIS['DQDZF0'][indices] * omega, axis=-1)
        if np.all(self.ambient ==np.ones_like(self.ambient)):
            inv_one_mask = ~np.isclose(points, complex(1, 0))   
            ints_for_power = np.tile(np.expand_dims(np.arange(len(inv_one_mask[-1])),0),(len(inv_one_mask),1))
            sign_for_omega2 = np.prod((-1)**(inv_one_mask*ints_for_power), axis=-1).astype(np.complex128)
            which_p1 = (indices//2)
            sign_of_omega = (-1)**which_p1 * (-1)**(indices%2+1) * sign_for_omega2 * (-1)**(indices%2+1)
            
        else:
            sign_of_omega = 1
        return 1 / omega*sign_of_omega
    


    def _find_max_dQ_coords_legacy(self, points):
        r"""Finds the coordinates for which |dQ/dz| is largest.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p), int]: max(dQdz) indices
        """
        dQdz = np.abs(self._compute_dQdz(points))
        dQdz = dQdz * (~np.isclose(points, complex(1, 0)))
        return np.argmax(dQdz, axis=-1)
    
    @staticmethod
    @jax_jit
    def _find_max_dQ_coords_jax(points, DQDZB0, DQDZF0):
        r"""JAX implementation to find coordinates for which |dQ/dz| is largest.

        Args:
            points (ndarray[(n_p, ncoords), jnp.complex128]): Points.
            DQDZB0: Basis for dQdz computation
            DQDZF0: Factors for dQdz computation

        Returns:
            ndarray[(n_p), int]: max(dQdz) indices
        """
        # Compute dQdz using JAX
        p_exp = jnp.expand_dims(jnp.expand_dims(points, 1), 1)
        dQdz = jnp.power(p_exp, DQDZB0)
        dQdz = jnp.multiply.reduce(dQdz, axis=-1)
        dQdz = jnp.multiply(DQDZF0, dQdz)
        dQdz = jnp.add.reduce(dQdz, axis=-1)
        
        # Take absolute value and mask out points close to 1
        dQdz_abs = jnp.abs(dQdz)
        mask = ~jnp.isclose(points, complex(1, 0))
        masked_dQdz = dQdz_abs * mask
        
        return jnp.argmax(masked_dQdz, axis=-1)
    
    def _find_max_dQ_coords(self, points, use_jax=True):
        r"""Finds the coordinates for which |dQ/dz| is largest.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p), int]: max(dQdz) indices
        """
        if use_jax and (isinstance(points, np.ndarray) or isinstance(points, jnp.ndarray)):
            return PointGenerator._find_max_dQ_coords_jax(points, jnp.array(self.BASIS['DQDZB0']), jnp.array(self.BASIS['DQDZF0']))
        else:
            print(f"using legacy find_max_dQ_coords, {type(points)}, use_jax = {use_jax}")
            return self._find_max_dQ_coords_legacy(points)

    def _find_good_coordinate_mask(self, points):
        r"""Computes a mask for points with True
        in the position of the local three 'good' coordinates.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p, ncoords), bool]: good coordinate mask
        """
        one_mask = ~np.isclose(points, complex(1, 0))
        dQdz = self._compute_dQdz(points)
        dQdz = dQdz * one_mask
        indices = np.argmax(np.abs(dQdz), axis=-1)
        dQdz_mask = -1 * np.eye(self.ncoords)[indices]
        full_mask = one_mask + dQdz_mask
        return full_mask.astype(bool)

    @staticmethod
    @jax_jit
    def _compute_dQdz_jax(points, DQDZB0, DQDZF0):
        r"""Computes dQdz at each point using JAX.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p, ncoords), np.complex128]: dQdz at each point.
        """
        p_exp = jnp.expand_dims(jnp.expand_dims(points, 1), 1)
        dQdz = jnp.power(p_exp, DQDZB0)
        dQdz = jnp.multiply.reduce(dQdz, axis=-1)
        dQdz = jnp.multiply(DQDZF0, dQdz)
        dQdz = jnp.add.reduce(dQdz, axis=-1)
        return dQdz
    def _compute_dQdz(self, points, use_jax=True):
        r"""Computes dQdz at each point.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.

        Returns:
            ndarray([n_p, ncoords], np.complex): dQdz at each point.
        """
        # Check if points is a numpy array or a JAX array
        if use_jax and (isinstance(points, np.ndarray) or isinstance(points, jnp.ndarray)):
            # Use JAX for numpy arrays
            return PointGenerator._compute_dQdz_jax(points, jnp.array(self.BASIS['DQDZB0']), jnp.array(self.BASIS['DQDZF0']))
        else:
            print(f"using legacy compute_dQdz, {type(points)}, use_jax = {use_jax}")
            # Use TensorFlow for other types (assuming TF tensors)
            import tensorflow as tf
            p_exp = tf.expand_dims(tf.expand_dims(points, 1), 1)
            dQdz = tf.pow(p_exp, tf.constant(self.BASIS['DQDZB0'], dtype=points.dtype))
            dQdz = tf.reduce_prod(dQdz, axis=-1)
            dQdz = tf.multiply(tf.constant(self.BASIS['DQDZF0'], dtype=points.dtype), dQdz)
            dQdz = tf.reduce_sum(dQdz, axis=-1)
            return dQdz.numpy() if hasattr(dQdz, 'numpy') else dQdz

    def _compute_dQdz_legacy(self, points):
        r"""Computes dQdz at each point.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.

        Returns:
            ndarray([n_p, ncoords], np.complex): dQdz at each point.
        """
        p_exp = np.expand_dims(np.expand_dims(points, 1), 1)
        dQdz = np.power(p_exp, self.BASIS['DQDZB0'])
        dQdz = np.multiply.reduce(dQdz, axis=-1)
        dQdz = np.multiply(self.BASIS['DQDZF0'], dQdz)
        dQdz = np.add.reduce(dQdz, axis=-1)
        return dQdz

    def point_weight_quadratic(self, points, omegasquared, pullbacks, normalize_to_vol_j=False):
        r"""We compute the weight/mass of each point:

        .. math::

            w &= \frac{d\text{Vol}_\text{cy}}{dA}|_p \\
              &\sim \frac{|\Omega|^2}{\det(g^\text{FS}_{ab})}|_p

        the weight depends on the distribution of free parameters during 
        point sampling. We employ a theorem due to Shiffman and Zelditch. 
        See also: [9803052]."""

        FS_shiffman = self.pointgen_jax_quadratic.pullback_all_3_spaces_vmap(points, pullbacks)
        weights = np.real(omegasquared / FS_shiffman)
        if normalize_to_vol_j:
            fs_ref = self.fubini_study_metrics(points, vol_js=np.ones_like(self.kmoduli))
            fs_ref_pb = np.einsum('xai,xij,xbj->xab', pullbacks, fs_ref, np.conj(pullbacks))
            norm_fac = self.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb))/FS_shiffman)# normalise to vol_j_norm?
            weights = norm_fac * weights
        return weights




    def point_weight(self, points, normalize_to_vol_j=False, j_elim=None, selected_t_val=None):
        r"""We compute the weight/mass of each point:

        .. math::

            w &= \frac{d\text{Vol}_\text{cy}}{dA}|_p \\
              &\sim \frac{|\Omega|^2}{\det(g^\text{FS}_{ab})}|_p

        the weight depends on the distribution of free parameters during 
        point sampling. We employ a theorem due to Shiffman and Zelditch. 
        See also: [9803052].

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            normalize_to_vol_j (bool, optional): Normalize such that

                .. math::

                    \int_X \det(g) &= \sum_i \det(g) \cdot w|_{x_i}\\
                                &= d^{ijk} t_i t_j t_k.

                Defaults to False.
            j_elim (ndarray([n_p, nhyper], int)): Index to be eliminated. 
                Defaults to None. If None eliminates max(dQdz).

        Returns:
            ndarray([n_p], np.float64): weight at each point.
        """
        omegas = self.holomorphic_volume_form(points, j_elim=j_elim)
        pbs = self.pullbacks(points, j_elim=j_elim)
        # find the nfold wedge product of omegas
        all_omegas = self.ambient - self.selected_t[selected_t_val]
        ts = np.zeros((self.nfold, len(all_omegas)))
        j = 0
        for i in range(len(all_omegas)):
            for _ in range(all_omegas[i]):
                ts[j, i] += 1
                j += 1
        fs_pbs = []
        for t in ts:
            fs = self.fubini_study_metrics(points, vol_js=t)
            fs_pbs += [np.einsum('xai,xij,xbj->xab', pbs, fs, np.conj(pbs))]
        # do antisymmetric tensor contraction. is there a nice way to do this
        # in arbitrary dimensions? Not that anyone would study 6-folds ..
        detg_norm = 1.
        if self.nfold == 1:
            detg_norm = np.einsum('xab->x', fs_pbs[0])
        elif self.nfold == 2:
            detg_norm = np.einsum('xab,xcd,ac,bd->x',
                                  fs_pbs[0], fs_pbs[1],
                                  self.lc, self.lc)
        elif self.nfold == 3:
            detg_norm = np.einsum('xab,xcd,xef,ace,bdf->x',
                                  fs_pbs[0], fs_pbs[1], fs_pbs[2],
                                  self.lc, self.lc)
        elif self.nfold == 4:
            detg_norm = np.einsum('xab,xcd,xef,xgh,aceg,bdfh->x',
                                  fs_pbs[0], fs_pbs[1], fs_pbs[2], fs_pbs[3],
                                  self.lc, self.lc)
        elif self.nfold == 5:
            detg_norm = np.einsum('xab,xcd,xef,xgh,xij,acegi,bdfhj->x',
                                  fs_pbs[0], fs_pbs[1], fs_pbs[2], fs_pbs[3],
                                  fs_pbs[4], self.lc, self.lc)
        else:
            logger.error('Weights are only implemented for nfold <= 5.'
                         'Run the tensorcontraction yourself :).')
        omega_squared = np.real(omegas * np.conj(omegas))
        weight = np.real(omega_squared / detg_norm)
        if normalize_to_vol_j:
            fs_ref = self.fubini_study_metrics(points, vol_js=np.ones_like(self.kmoduli))
            fs_ref_pb = np.einsum('xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
            norm_fac = self.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / detg_norm)
            weight = norm_fac * weight
        return weight
    def pullbacks(self, points, j_elim=None, use_jax=True):
        r"""Computes the pullback from ambient space to local CY coordinates
        at each point. 
        
        Denote the ambient space coordinates with z_i and the CY
        coordinates with x_a then

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            j_elim (ndarray([n_p, nhyper], int)): Index to be eliminated. 
                Defaults to None. If None eliminates max(dQdz).
            use_jax (bool, optional): Whether to use JAX implementation.
                Defaults to True.

        Returns:
            ndarray([n_p, nfold, ncoords], np.complex128): Pullback tensor 
                at each point.
        """
        if use_jax and (isinstance(points, np.ndarray) or isinstance(points, jnp.ndarray)):
            return PointGenerator._pullbacks_jax(points, j_elim, jnp.array(self.BASIS['DQDZB0']), jnp.array(self.BASIS['DQDZF0']), self.nfold, self.nhyper, self.ncoords)
        else:
            print("using legacy pullbacks, input type:", type(points), "use_jax = ", use_jax)
            return self._pullbacks_legacy(points, j_elim)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(4,5,6))
    def _pullbacks_jax(points, j_elim, DQDZB0, DQDZF0, nfold, nhyper, ncoords):
        """JAX version of pullbacks computation.
        
        Args:
            points (jnp.ndarray, (batch,complex64): Array of points.
            j_elim (jnp.ndarray): Indices to eliminate.
            DQDZB0, DQDZF0: Sequences of basis arrays.
            nfold (int): Number of folds.
            nhyper (int): Number of hypersurfaces.
            ncoords (int): Number of coordinates.
        """
        if nhyper>1:
            raise ValueError("nhyper > 1 not implemented for JAX pullbacks")
            print("nhyper > 1 not implemented for JAX pullbacks")
            return self._pullbacks_legacy(points, j_elim)
        inv_one_mask = jnp.logical_not(jnp.isclose(points, 1 + 0j))
        if j_elim is None:
            j_elim = PointGenerator._find_max_dQ_coords_jax(points, DQDZB0, DQDZF0)
        if j_elim.ndim == 1:
            j_elim = jnp.reshape(j_elim, (-1, 1))
        full_mask = jnp.array(inv_one_mask)
        for i in range(nhyper):
            full_mask = full_mask.at[jnp.arange(points.shape[0]), j_elim[:, i]].set(False)
        x_indices, z_indices = jnp.where(full_mask, size=nfold*points.shape[0])
        pullbacks = jnp.zeros((points.shape[0], nfold, ncoords), dtype=jnp.complex128)
        y_indices = jnp.repeat(jnp.expand_dims(jnp.arange(nfold), 0), points.shape[0], axis=0)
        y_indices = jnp.reshape(y_indices, (-1,))
        pullbacks = pullbacks.at[x_indices, y_indices, z_indices].set(1 + 0j)
        B_matrix = jnp.zeros((points.shape[0], nhyper, nhyper), dtype=jnp.complex128)
        dz_hyper = jnp.zeros((points.shape[0], nhyper, nfold), dtype=jnp.complex128)
        fixed_indices = jnp.reshape(j_elim, (-1,))
        for i in range(nhyper):
            # Compute p_iα (eq. 5.24)
            pia_polys = DQDZB0[z_indices]
            pia_factors = DQDZF0[z_indices]
            rep_pts = jnp.expand_dims(jnp.repeat(points, nfold, axis=0), 1)
            pia = jnp.power(rep_pts, pia_polys)
            pia = jnp.prod(pia, axis=-1)
            pia = jnp.sum(pia_factors * pia, axis=-1)
            pia = jnp.reshape(pia, (-1, nfold))
            dz_hyper = dz_hyper.at[:, i, :].add(pia)
            # Compute p_i(fixed)
            pif_polys = DQDZB0[fixed_indices]
            pif_factors = DQDZF0[fixed_indices]
            rep_pts_fixed = jnp.expand_dims(jnp.repeat(points, nhyper, axis=0), 1)
            pif = jnp.power(rep_pts_fixed, pif_polys)
            pif = jnp.prod(pif, axis=-1)
            pif = jnp.sum(pif_factors * pif, axis=-1)
            pif = jnp.reshape(pif, (-1, nhyper))
            B_matrix = B_matrix.at[:, i, :].add(pif)
        all_dzdz = jnp.einsum('xij,xjk->xki', jnp.linalg.inv(B_matrix), complex(-1., 0.) * dz_hyper)
        #all_dzdz = jnp.einsum('xij,xjk->xki', jnp.linalg.inv(B_matrix), dz_hyper)
        for i in range(nhyper):
            pullbacks = pullbacks.at[jnp.arange(points.shape[0]), :, j_elim[:, i]].add(all_dzdz[:, :, i])
        return pullbacks

    def _pullbacks_legacy(self, points, j_elim=None):
        r"""Legacy numpy implementation of pullbacks computation.
        
        Denote the ambient space coordinates with z_i and the CY
        coordinates with x_a then

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            j_elim (ndarray([n_p, nhyper], int)): Index to be eliminated. 
                Defaults to None. If None eliminates max(dQdz).

        Returns:
            ndarray([n_p, nfold, ncoords], np.complex128): Pullback tensor 
                at each point.
        """
        inv_one_mask = ~np.isclose(points, complex(1, 0))
        if j_elim is None:
            j_elim = self._find_max_dQ_coords(points)
        if len(j_elim.shape) == 1:
            j_elim = np.reshape(j_elim, (-1, 1))
        full_mask = np.copy(inv_one_mask)
        for i in range(self.nhyper):
            full_mask[np.arange(len(points)), j_elim[:, i]] = np.zeros(len(points), dtype=bool)

        # fill the diagonal ones in pullback
        x_indices, z_indices = np.where(full_mask)# row and column indices
        pullbacks = np.zeros((len(points), self.nfold, self.ncoords), dtype=np.complex128)
        y_indices = np.repeat(np.expand_dims(np.arange(self.nfold), 0), len(points), axis=0)
        y_indices = np.reshape(y_indices, (-1))
        pullbacks[x_indices, y_indices, z_indices] = np.ones(self.nfold * len(points), dtype=np.complex128)
        # next fill the dzdz from every hypersurface
        B_matrix = np.zeros((len(points), self.nhyper, self.nhyper), dtype=np.complex128)
        dz_hyper = np.zeros((len(points), self.nhyper, self.nfold), dtype=np.complex128)
        fixed_indices = np.reshape(j_elim, (-1))
        for i in range(self.nhyper):
            # compute p_i\alpha eq (5.24)
            pia_polys = self.BASIS['DQDZB' + str(i)][z_indices]
            pia_factors = self.BASIS['DQDZF' + str(i)][z_indices]
            pia = np.power(np.expand_dims(
                np.repeat(points, self.nfold, axis=0), 1), pia_polys)
            pia = np.multiply.reduce(pia, axis=-1)
            pia = np.add.reduce(np.multiply(pia_factors, pia), axis=-1)
            pia = np.reshape(pia, (-1, self.nfold))
            dz_hyper[:, i, :] += pia
            # compute p_ifixed
            pif_polys = self.BASIS['DQDZB' + str(i)][fixed_indices]
            pif_factors = self.BASIS['DQDZF' + str(i)][fixed_indices]
            pif = np.power(np.expand_dims(
                np.repeat(points, self.nhyper, axis=0), 1), pif_polys)
            pif = np.multiply.reduce(pif, axis=-1)
            pif = np.add.reduce(np.multiply(pif_factors, pif), axis=-1)
            pif = np.reshape(pif, (-1, self.nhyper))
            B_matrix[:, i, :] += pif
        all_dzdz = np.einsum('xij,xjk->xki',
                             np.linalg.inv(B_matrix),
                             #dz_hyper)
                             complex(-1., 0.) * dz_hyper)
        for i in range(self.nhyper):
            pullbacks[np.arange(len(points)), :, j_elim[:, i]] += all_dzdz[:, :, i]
        return pullbacks

    def _pullbacks_dzdz(self, points, j_elim=None):
        r"""Computes the pullback from ambient space to local CY coordinates
        at each point. 
        
        Denote the ambient space coordinates with z_i and the CY
        coordinates with x_a then

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        NOTE: Uses the implicit derivatives dz/dz rather than inverse matrix.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            j_elim (ndarray([n_p], int)): index to be eliminated. 
                Defaults not None. If None eliminates max(dQdz).

        Returns:
            ndarray([n_p, nfold, ncoords], np.complex128): Pullback tensor
                at each point.
        """
        if not self.dzdz_generated:
            # This will take some time when called for the first time.
            self.dzdz_generated = True
            self._generate_dzdz_basis()
            self._generate_padded_basis()
        one_mask = ~np.isclose(points, complex(1, 0))
        if j_elim is None:
            dQdz = self._compute_dQdz(points)
            dQdz = dQdz * one_mask
            dQdz_indices = np.argmax(np.abs(dQdz), axis=-1)
        else:
            dQdz_indices = j_elim
        dQdz_mask = np.eye(self.ncoords)[dQdz_indices]
        full_mask = one_mask - dQdz_mask
        full_mask = full_mask.astype(bool)
        x_indices, z_indices = np.where(full_mask)
        nrepeat = self.nfold
        dQdz_indices = np.repeat(dQdz_indices, nrepeat)

        # compute everything
        numerators = self.BASIS['DZDZB_n0'][dQdz_indices, z_indices]
        num_factors = self.BASIS['DZDZF_n0'][dQdz_indices, z_indices]
        denominators = self.BASIS['DZDZB_d0'][dQdz_indices, z_indices]
        den_factors = self.BASIS['DZDZF_d0'][dQdz_indices, z_indices]
        num_res = np.power(np.expand_dims(
            np.repeat(points, nrepeat, axis=0), 1), numerators)
        num_res = np.multiply.reduce(num_res, axis=-1)
        num_res = np.add.reduce(np.multiply(num_factors, num_res), axis=-1)
        den_res = np.power(np.expand_dims(
            np.repeat(points, nrepeat, axis=0), 1), denominators)
        den_res = np.multiply.reduce(den_res, axis=-1)
        den_res = np.add.reduce(np.multiply(den_factors, den_res), axis=-1)
        all_dzdz = num_res / den_res

        # fill everything
        x_indices = np.concatenate((x_indices, x_indices))
        z_indices = np.concatenate((z_indices, dQdz_indices))
        y_indices = np.repeat(np.expand_dims(
            np.arange(nrepeat), 0), len(points), axis=0)
        y_indices = np.reshape(y_indices, (-1))
        y_indices = np.concatenate((y_indices, y_indices))
        all_values = np.concatenate(
            (np.ones(nrepeat * len(points), dtype=np.complex128), all_dzdz),
            axis=0)
        pullbacks = np.zeros(
            (len(points), nrepeat, self.ncoords), dtype=np.complex128)
        pullbacks[x_indices, y_indices, z_indices] = all_values
        return pullbacks

    def compute_kappa(self, points, weights, omegas):
        r"""We compute kappa from the Monge-Ampère equation

        .. math:: J^3 = \kappa |\Omega|^2
        
        such that after integrating we find

        .. math::

            \kappa = \frac{J^3}{|\Omega|^2} =
                \frac{\text{Vol}_K}{\text{Vol}_{\text{CY}}}

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            weights (ndarray[n_p, np.float64]): weights of the points.
            omegas (ndarray[n_p, np.complex128]): Omega \wedge Omega* of the points.

        Returns:
            np.float: kappa
        """
        weights, omegas = weights.flatten(), omegas.flatten()
        pbs = self.batch_function(self.pullbacks, points)
        gFS = self.batch_function(self.fubini_study_metrics, points)
        gFS_pbs = np.einsum('xai,xij,xbj->xab', pbs, gFS, np.conj(pbs))
        dets = np.real(np.linalg.det(gFS_pbs))

        vol_k = np.mean(weights * dets / omegas)
        vol_cy = np.mean(weights)
        logger.info('Vol_k: {}, Vol_cy: {}.'.format(vol_k, vol_cy))
        kappa = vol_k / vol_cy
        return kappa

    def fubini_study_metrics(self, points, vol_js=None):
        r"""Computes the FS metric at each point.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factor.
                Defaults to None.

        Returns:
            ndarray[(n_p, ncoords, ncoords), np.complex128]: g^FS
        """
        gFS = np.zeros((len(points), self.ncoords, self.ncoords),
                       dtype=np.complex128)
        kmoduli = self.kmoduli if vol_js is None else vol_js
        for i in range(len(self.ambient)):
            s = np.sum(self.degrees[0:i])
            e = np.sum(self.degrees[0:i + 1])
            gFS[:, s:e, s:e] += self._fubini_study_n_metrics(points[:, s:e], vol_j=kmoduli[i])
        return gFS

    @staticmethod
    def inverse_fubini_study_metrics_jax(points, vol_js):
        r"""Computes the FS metric at each point.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factor.
                Defaults to None.

        Returns:
            ndarray[(n_p, ncoords, ncoords), np.complex128]: g^FS
        """
        gFS = np.zeros((len(points),8,8),
                       dtype=np.complex128)
        for i in range(4):
            s = 2*i
            e =2*i + 2
            gFS[:, s:e, s:e] += jnp.matrix_transpose(PointGenerator._fubini_study_n_metrics_jax(points[:, s:e], vol_js[i]**(-1))*np.sum(np.abs(points[:, s:e])**2,axis=-1)**2)
            #so it has indices bbar a
        return gFS

    @staticmethod
    def _fubini_study_n_metrics(points, vol_j=1. + 0.j):
        r"""Computes the FS metric for a single projective space of points.

        Args:
            point (ndarray[(n_p, n), np.complex128]): Points.
            vol_j (complex): Volume factor. Defaults to 1+0.j.

        Returns:
            ndarray[(n_p, n, n), np.complex128]: g^FS
        """
        # we want to compute d_i d_j K^FS
        point_square = np.add.reduce(np.abs(points) ** 2, axis=-1)
        outer = np.einsum('xi,xj->xij', np.conj(points), points)
        gFS = np.einsum('x,ij->xij', point_square, np.eye(points.shape[1]))
        gFS = gFS.astype(np.complex128) - outer
        return np.einsum('xij,x->xij', gFS, 1 / (point_square ** 2)) * vol_j / np.pi

    @staticmethod
    def _fubini_study_n_metrics_jax(points, vol_j=1. + 0.j):
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
        gFS = jnp.einsum('x,ij->xij', point_square, jnp.eye(points.shape[1]))
        gFS = gFS.astype(jnp.complex128) - outer
        return jnp.einsum('xij,x->xij', gFS, 1 / (point_square ** 2)) * vol_j / np.pi

    def prepare_basis(self, dirname, kappa=1.):
        r"""Prepares pickled monomial basis for the tensorflow models.

        Args:
            dirname (str): dir name to save

        Returns:
            int: 0
        """
        return prepare_basis_pickle(self, dirname, kappa)

    def prepare_dataset(self, n_p, dirname, val_split=0.1, ltails=0, rtails=0,average_selected_t = False, batch_size = None):
        r"""Prepares training and validation data.

        Args:
            n_p (int): Number of points to generate.
            dirname (str): Directory name to save dataset in.
            val_split (float, optional): train-val split. Defaults to 0.1.
            ltails (float, optional): Percentage discarded on the left tail
                of weight distribution. Defaults to 0.
            rtails (float, optional): Percentage discarded on the right tail
                of weight distribution. Defaults to 0.

        Returns:
            np.float: kappa = vol_k / vol_cy
        """
        return prepare_dataset(self, n_p, dirname, val_split=val_split, ltails=ltails, rtails=rtails,average_selected_t = average_selected_t, batch_size = batch_size)

    def cy_condition(self, points):
        r"""Computes the CY condition at each point.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points (on the CY).

        Returns:
            ndarray(n_p, np.complex128): CY condition
        """
        cy_condition = np.power(np.expand_dims(points, 1), self.monomials)
        cy_condition = np.multiply.reduce(cy_condition, axis=-1)
        cy_condition = np.add.reduce(self.coefficients * cy_condition, axis=-1)
        return cy_condition

    def __call__(self, points, vol_js=None):
        r"""Computes the FS metric at each point.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factors. 
                Defaults to None.

        Returns:
            ndarray[(n_p, ncoords, ncoords), np.complex128]: g^FS
        """
        return self.fubini_study_metrics(points, vol_js=vol_js)

    def batch_function(self, func, *args, batch_size=30000, **kwargs):
        """Batches any function to process data in chunks of specified size.
        
        Args:
            func (callable): Function to batch process
            *args: Arguments to pass to the fonction
            t_
            batch_size (int, optional): Size of each batch. Defaults to 1000000.
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Combined results from all batches
        """
        # Get the first argument's length to determine number of batches
        total_size = len(args[0])
        if total_size<batch_size:
            return func(*args, **kwargs)
        results = []
        
        for i in range(0, total_size, batch_size):
            # Create batch slices for all array arguments
            batch_args = [arg[i:i+batch_size] if hasattr(arg, '__len__') else arg for arg in args]
            batch_result = func(*batch_args, **kwargs)
            results.append(batch_result)
            
        # Handle different return types
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], jnp.ndarray):
            return jnp.concatenate(results)
        else:
            import tensorflow as tf
            if isinstance(results[0], tf.Tensor):
                return tf.concat(results, axis=0)
            else:
                raise ValueError(f"Unsupported return type: {type(results[0])}")
