from operator import add
from ufl import TrialFunction, TestFunction, dx, dot, grad, Constant, Measure
from dolfinx import fem

# from dolfinx import io
from dolfinx import mesh as _mesh

# from mpi4py import MPI
import basix
from petsc4py import PETSc
import logging
import numpy as np

from networks_fenicsx.mesh import mesh
from networks_fenicsx.utils import petsc_utils
from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
Modified by Joseph P. Dean - 2023
"""


class Assembler:
    # TODO
    # G: mesh.NetworkGraph
    # __slots__ = tuple(__annotations__)

    def __init__(self, config: config.Config, graph: mesh.NetworkGraph):
        self.G = graph
        self.function_spaces = None
        self.lm_function_spaces = None
        self.a = None
        self.L = None
        self.A = None
        self.L = None
        self.cfg = config

    def dds(self, f):
        """
        function for derivative df/ds along graph
        """
        return dot(grad(f), self.G.global_tangent)

    # Compute jump vector when Lagrange multipliers are manually inserted in the matrix
    def jump_vector(self, q, ix, j):
        """
        Returns the signed jump vector for a flux function q on edge ix
        over bifurcation j
        """

        edge_list = list(self.G.edges.keys())

        # Iitialize form to zero
        zero = fem.Function(q.ufl_function_space())
        L = zero * q * dx

        # Add point integrals (jump)
        for i, e in enumerate(self.G.in_edges(j)):
            ds_edge = Measure(
                "ds",
                domain=self.G.edges[e]["submesh"],
                subdomain_data=self.G.edges[e]["vf"],
            )
            edge_ix = edge_list.index(e)
            if ix == edge_ix:
                L += q * ds_edge(self.G.BIF_IN)

        for i, e in enumerate(self.G.out_edges(j)):
            ds_edge = Measure(
                "ds",
                domain=self.G.edges[e]["submesh"],
                subdomain_data=self.G.edges[e]["vf"],
            )
            edge_ix = edge_list.index(e)
            if ix == edge_ix:
                L -= q * ds_edge(self.G.BIF_OUT)

        L = fem.form(L)

        return L

    @timeit
    def compute_forms(self, f=None, p_bc_ex=None):
        """
        Compute forms for hydraulic network model
            R q + d/ds p = 0
            d/ds q = f
        on graph G, with bifurcation condition q_in = q_out
        and jump vectors the bifurcation conditions

        Args:
           f (dolfinx.fem.function): source term
           p_bc (class): neumann bc for pressure
        """

        if f is None:
            f = Constant(self.G.mesh, 0)

        submeshes = self.G.submeshes()

        # Flux spaces on each segment, ordered by the edge list
        # Using equispaced elements to match with legacy FEniCS
        flux_degree = self.cfg.flux_degree
        flux_element = basix.ufl.element(
            family="Lagrange",
            cell="interval",
            degree=flux_degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
        Pqs = [fem.functionspace(submsh, flux_element) for submsh in submeshes]

        pressure_degree = self.cfg.pressure_degree
        pressure_element = basix.ufl.element(
            family="Lagrange",
            cell="interval",
            degree=pressure_degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
        Pp = fem.functionspace(self.G.mesh, pressure_element)

        self.function_spaces = Pqs + [Pp]

        if len(self.G.bifurcation_ixs) > 0:
            raise RuntimeError("Lets fix this")
        self.lm_function_spaces = [
            fem.functionspace(self.G.lm_smsh, ("Discontinuous Lagrange", 0))
            for i in self.G.bifurcation_ixs
        ]

        # Fluxes on each branch
        qs = []
        vs = []
        for Pq in Pqs:
            qs.append(TrialFunction(Pq))
            vs.append(TestFunction(Pq))

        # Lagrange multipliers
        lmbdas = []
        mus = []
        if self.cfg.lm_spaces:
            for fs in self.lm_function_spaces:
                lmbdas.append(TrialFunction(fs))
                mus.append(TestFunction(fs))
        else:
            self.L_jumps = [
                [self.jump_vector(q, ix, j) for j in self.G.bifurcation_ixs]
                for ix, q in enumerate(qs)
            ]

        # Pressure
        p = TrialFunction(Pp)
        phi = TestFunction(Pp)

        # Assemble variational formulation
        dx_zero = Measure(
            "dx",
            domain=self.G.mesh,
            subdomain_data=_mesh.meshtags(
                self.G.mesh,
                self.G.mesh.topology.dim,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            ),
        )

        # Initialize forms
        num_qs = len(submeshes)
        num_lmbdas = len(lmbdas)
        num_blocks = num_qs + num_lmbdas + 1
        self.a = [[None] * num_blocks for i in range(num_blocks)]
        self.L = [None] * num_blocks

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            submsh = self.G.edges[e]["submesh"]
            entity_maps = {self.G.mesh: self.G.edges[e]["entity_map"]}

            dx_edge = Measure("dx", domain=submsh)
            ds_edge = Measure("ds", domain=submsh, subdomain_data=self.G.edges[e]["vf"])

            self.a[i][i] = fem.form(qs[i] * vs[i] * dx_edge)
            self.a[num_qs][i] = fem.form(
                phi * self.dds(qs[i]) * dx_edge, entity_maps=entity_maps
            )
            self.a[i][num_qs] = fem.form(
                -p * self.dds(vs[i]) * dx_edge, entity_maps=entity_maps
            )

            # Boundary condition on the correct space
            P1_e = fem.functionspace(self.G.edges[e]["submesh"], ("Lagrange", 1))
            p_bc = fem.Function(P1_e)
            p_bc.interpolate(p_bc_ex.eval)

            self.L[i] = fem.form(
                p_bc * vs[i] * ds_edge(self.G.BOUN_IN)
                - p_bc * vs[i] * ds_edge(self.G.BOUN_OUT)
            )

        if self.cfg.lm_spaces:
            edge_list = list(self.G.edges.keys())
            entity_maps = {self.G.lm_smsh: np.zeros(1, dtype=np.int32)}
            for j, bix in enumerate(self.G.bifurcation_ixs):
                # Add point integrals (jump)
                for i, e in enumerate(self.G.in_edges(bix)):
                    ds_edge = Measure(
                        "ds",
                        domain=self.G.edges[e]["submesh"],
                        subdomain_data=self.G.edges[e]["vf"],
                    )
                    edge_ix = edge_list.index(e)
                    assert self.a[num_qs + 1 + j][edge_ix] is None
                    assert self.a[edge_ix][num_qs + 1 + j] is None

                    self.a[num_qs + 1 + j][edge_ix] = fem.form(
                        mus[j] * qs[edge_ix] * ds_edge(self.G.BIF_IN),
                        entity_maps=entity_maps,
                    )
                    self.a[edge_ix][num_qs + 1 + j] = fem.form(
                        lmbdas[j] * vs[edge_ix] * ds_edge(self.G.BIF_IN),
                        entity_maps=entity_maps,
                    )

                for i, e in enumerate(self.G.out_edges(bix)):
                    ds_edge = Measure(
                        "ds",
                        domain=self.G.edges[e]["submesh"],
                        subdomain_data=self.G.edges[e]["vf"],
                    )
                    edge_ix = edge_list.index(e)
                    assert self.a[num_qs + 1 + j][edge_ix] is None
                    assert self.a[edge_ix][num_qs + 1 + j] is None

                    self.a[num_qs + 1 + j][edge_ix] = fem.form(
                        -mus[j] * qs[edge_ix] * ds_edge(self.G.BIF_OUT),
                        entity_maps=entity_maps,
                    )
                    self.a[edge_ix][num_qs + 1 + j] = fem.form(
                        -lmbdas[j] * vs[edge_ix] * ds_edge(self.G.BIF_OUT),
                        entity_maps=entity_maps,
                    )

                self.L[num_qs + 1 + j] = fem.form(
                    1e-16 * mus[j] * dx
                )  # TODO Use constant

        # Add zero to uninitialized diagonal blocks (needed by petsc)
        zero = fem.Function(Pp)
        self.a[num_qs][num_qs] = fem.form(zero * p * phi * dx_zero)
        self.L[num_qs] = fem.form(zero * phi * dx_zero)

    @timeit
    def assemble(self):
        # Get the forms
        a = self.bilinear_forms()
        L = self.linear_forms()

        # Assemble system from the given forms
        A = fem.petsc.assemble_matrix_block(a)
        A.assemble()
        b = fem.petsc.assemble_vector_block(L, a)
        b.assemble()

        if self.cfg.lm_spaces:
            self.A = A
            self.b = b
            return (A, b)
        else:
            _A_size = A.getSize()
            _b_size = b.getSize()

            _A_values = A.getValues(range(_A_size[0]), range(_A_size[1]))
            _b_values = b.getValues(range(_b_size))

            # Build new system to include Lagrange multipliers for the bifurcation conditions
            num_bifs = len(self.G.bifurcation_ixs)
            A_ = PETSc.Mat().create()
            A_.setSizes(list(map(add, _A_size, (num_bifs, num_bifs))))
            A_.setUp()

            b_ = PETSc.Vec().create()
            b_.setSizes(_b_size + num_bifs)
            b_.setUp()

            # Copy _A and _b values into (bigger) system
            A_.setValuesBlocked(range(_A_size[0]), range(_A_size[1]), _A_values)
            b_.setValuesBlocked(range(_b_size), _b_values)

            # Assemble jump vectors and convert to PETSc.Mat() object
            self.jump_vectors = [
                [fem.petsc.assemble_vector(L) for L in qi] for qi in self.L_jumps
            ]
            jump_vecs = [
                [petsc_utils.convert_vec_to_petscmatrix(b_row) for b_row in qi]
                for qi in self.jump_vectors
            ]

            # Insert jump vectors into A_new
            for i in range(0, num_bifs):
                for j in range(0, self.G.num_edges):
                    jump_vec = jump_vecs[j][i]
                    jump_vec_values = jump_vec.getValues(
                        range(jump_vec.getSize()[0]), range(jump_vec.getSize()[1])
                    )[0]
                    A_.setValuesBlocked(
                        _A_size[0] + i,
                        range(
                            jump_vec.getSize()[1] * j, jump_vec.getSize()[1] * (j + 1)
                        ),
                        jump_vec_values,
                    )
                    jump_vec.transpose()
                    jump_vec_T_values = jump_vec.getValues(
                        range(jump_vec.getSize()[0]), range(jump_vec.getSize()[1])
                    )
                    A_.setValuesBlocked(
                        range(
                            jump_vec.getSize()[0] * j, jump_vec.getSize()[0] * (j + 1)
                        ),
                        _A_size[1] + i,
                        jump_vec_T_values,
                    )

            # Assembling A and b
            A_.assemble()
            b_.assemble()

            self.A = A_
            self.b = b_
            return (A_, b_)

    def bilinear_forms(self):
        if self.a is None:
            logging.error(
                "Bilinear forms haven't been computed. Need to call compute_forms()"
            )
        else:
            return self.a

    def bilinear_form(self, i: int, j: int):
        a = self.bilinear_forms()
        if i > len(a) or j > len(a[i]):
            logging.error(
                "Bilinear form a[" + str(i) + "][" + str(j) + "] out of range"
            )
        return a[i][j]

    def linear_forms(self):
        if self.L is None:
            logging.error(
                "Linear forms haven't been computed. Need to call compute_forms()"
            )
        else:
            return self.L

    def linear_form(self, i: int):
        L = self.linear_forms()
        if i > len(L):
            logging.error("Linear form L[" + str(i) + "] out of range")
        return L[i]

    def assembled_matrix(self):
        if self.A is None:
            logging.error("Matrix has not been assemble. Need to call assemble()")
        else:
            return self.A

    def assembled_rhs(self):
        if self.b is None:
            logging.error("RHS has not been assemble. Need to call assemble()")
        else:
            return self.b
