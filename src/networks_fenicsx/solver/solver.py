from petsc4py import PETSc

from networks_fenicsx.mesh import mesh
from networks_fenicsx.solver import assembly
from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

from mpi4py import MPI

"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
"""


class Solver:
    def __init__(
        self,
        config: config.Config,
        graph: mesh.NetworkGraph,
        assembler: assembly.Assembler,
    ):
        self.G = graph
        self.assembler = assembler
        self.cfg = config

        if self.assembler is not None:
            self.A = assembler.assembled_matrix()
            self.b = assembler.assembled_rhs()

    @timeit
    def solve(self):
        # Configure solver
        ksp = PETSc.KSP().create(self.G.mesh.comm)
        ksp.setOperators(self.A)

        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        if MPI.COMM_WORLD.size > 1:
            ksp.getPC().setFactorSolverType("mumps")
        else:
            ksp.getPC().setFactorSolverType("umfpack")

        # Solve
        x = self.A.createVecLeft()
        ksp.solve(self.b, x)

        return x
