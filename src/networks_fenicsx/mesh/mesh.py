import networkx as nx
import numpy as np
from typing import List
import copy

from mpi4py import MPI
from dolfinx import fem, io, mesh

import gmsh

from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
Modified by Joseph P. Dean - 2023
"""


class NetworkGraph(nx.DiGraph):
    """
    Make FEniCSx mesh from networkx directed graph
    """

    _msh: mesh.Mesh | None
    _subdomains: mesh.MeshTags | None
    _facet_markers: mesh.MeshTags | None

    def __init__(self, config: config.Config, graph: nx.DiGraph = None):
        nx.DiGraph.__init__(self, graph)

        self.comm = MPI.COMM_WORLD
        self.cfg = config
        self.cfg.clean_dir()

        self.bifurcation_ixs: List[int] = []  # noqa: F821
        self.boundary_ixs: List[int] = []  # noqa: F821

        self._msh = None
        self.lm_smsh = None
        self._subdomains = None
        self._facet_markers = None
        self.global_tangent = None

        self.BIF_IN = 1
        self.BIF_OUT = 2
        self.BOUN_IN = 3
        self.BOUN_OUT = 4

    @timeit
    def build_mesh(self):
        """
        Makes a fenics mesh from the graph
        Args:
            lcar (float): Characteristic length of the elements
        Returns:
            mesh : the global mesh
        """

        self.geom_dim = len(self.nodes[1]["pos"])
        self.num_edges = len(self.edges)

        vertex_coords = np.asarray([self.nodes[v]["pos"] for v in self.nodes()])
        cells_array = np.asarray([[u, v] for u, v in self.edges()])

        line_weights = np.linspace(
            0, 1, int(np.floor(1 / self.cfg.lcar)), endpoint=False
        )[1:][:, None]

        # Create mesh segments
        mesh_nodes = vertex_coords.copy()
        cells = []
        cell_markers = []
        for i, segment in enumerate(cells_array):
            start_coord_pos = mesh_nodes.shape[0]
            start = vertex_coords[segment[0]]
            end = vertex_coords[segment[1]]

            internal_line_coords = start * (1 - line_weights) + end * line_weights
            mesh_nodes = np.vstack((mesh_nodes, internal_line_coords))
            cells.append([segment[0], start_coord_pos])
            segment_connectivity = (
                np.repeat(np.arange(internal_line_coords.shape[0]), 2)[1:-1].reshape(
                    internal_line_coords.shape[0] - 1, 2
                )
                + start_coord_pos
            )
            cells.append(segment_connectivity)
            cells.append(
                [start_coord_pos + internal_line_coords.shape[0] - 1, segment[1]]
            )
            cell_markers.extend(
                np.full(internal_line_coords.shape[0] + 2, i, dtype=np.int32)
            )

        cells = np.vstack(cells).astype(np.int64)
        breakpoint()
        # for i, v in enumerate(vertex_coords):
        #     if len(v) == 2:
        #         pts.append(gmsh.model.geo.addPoint(v[0], v[1], self.cfg.lcar))
        #     elif len(v) == 3:
        #         pts.append(gmsh.model.geo.addPoint(v[0], v[1], v[2], self.cfg.lcar))

        # for i, c in enumerate(cells_array):
        #     lines.append(gmsh.model.geo.addLine(pts[c[0]], pts[c[1]]))

        # gmsh.model.geo.synchronize()
        # for i, line in enumerate(lines):
        #     gmsh.model.addPhysicalGroup(1, [line], i)

        # gmsh.model.mesh.generate(1)

        # mesh_data = io.gmsh.model_to_mesh(
        #     gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=self.geom_dim
        # )

        self._msh = mesh_data.mesh
        assert mesh_data.cell_tags is not None
        self._subdomains = mesh_data.cell_tags
        self._facet_markers = mesh_data.facet_tags

        if self.cfg.export:
            with io.XDMFFile(
                self.comm, self.cfg.outdir + "/mesh/mesh.xdmf", "w"
            ) as file:
                file.write_mesh(self.mesh)
                file.write_meshtags(self.subdomains, self.mesh.geometry)
            gmsh.write(self.cfg.outdir + "/mesh/mesh.msh")
        gmsh.finalize()

        # Submesh for the Lagrange multiplier
        # self.lm_smsh = mesh.create_submesh(self.mesh, self.mesh.topology.dim, [0])[0]

    @timeit
    def build_network_submeshes(self):
        for i, (u, v) in enumerate(self.edges):
            edge_subdomain = self.subdomains.find(i)

            self.edges[u, v]["submesh"], self.edges[u, v]["entity_map"] = (
                mesh.create_submesh(self.mesh, self.mesh.topology.dim, edge_subdomain)[
                    0:2
                ]
            )
            self.edges[u, v]["tag"] = i

            self.edges[u, v]["entities"] = []
            self.edges[u, v]["b_values"] = []

    @timeit
    def build_markers(self):
        # Marking the bifurcations (in/out) and boundaries (in/out) for extermities of each edges
        for n, v in enumerate(self.nodes()):
            num_conn_edges = len(self.in_edges(v)) + len(self.out_edges(v))
            if num_conn_edges == 0:
                print(f"Node {v} in G is lonely (i.e. unconnected)")

            bifurcation = bool(num_conn_edges > 1)
            boundary = bool(num_conn_edges == 1)
            if bifurcation:
                self.bifurcation_ixs.append(v)
            if boundary:
                self.boundary_ixs.append(v)

            for i, e in enumerate(self.in_edges(v)):
                e_msh = self.edges[e]["submesh"]
                if len(self.nodes[v]["pos"]) == 2:
                    entities = mesh.locate_entities(
                        e_msh,
                        0,
                        lambda x: np.logical_and(
                            np.isclose(x[0], self.nodes[v]["pos"][0]),
                            np.isclose(x[1], self.nodes[v]["pos"][1]),
                        ),
                    )
                else:
                    entities = mesh.locate_entities(
                        e_msh,
                        0,
                        lambda x: np.logical_and(
                            np.isclose(x[0], self.nodes[v]["pos"][0]),
                            np.isclose(x[1], self.nodes[v]["pos"][1]),
                            np.isclose(x[2], self.nodes[v]["pos"][2]),
                        ),
                    )
                self.edges[e]["entities"].append(entities)
                if bifurcation:
                    b_values_in = np.full(entities.shape, self.BIF_IN, np.intc)
                elif boundary:
                    b_values_in = np.full(entities.shape, self.BOUN_OUT, np.intc)
                self.edges[e]["b_values"].append(b_values_in)

            for i, e in enumerate(self.out_edges(v)):
                e_msh = self.edges[e]["submesh"]
                if len(self.nodes[v]["pos"]) == 2:
                    entities = mesh.locate_entities(
                        e_msh,
                        0,
                        lambda x: np.logical_and(
                            np.isclose(x[0], self.nodes[v]["pos"][0]),
                            np.isclose(x[1], self.nodes[v]["pos"][1]),
                        ),
                    )
                else:
                    entities = mesh.locate_entities(
                        e_msh,
                        0,
                        lambda x: np.logical_and(
                            np.isclose(x[0], self.nodes[v]["pos"][0]),
                            np.isclose(x[1], self.nodes[v]["pos"][1]),
                            np.isclose(x[2], self.nodes[v]["pos"][2]),
                        ),
                    )

                self.edges[e]["entities"].append(entities)
                if bifurcation:
                    b_values_out = np.full(entities.shape, self.BIF_OUT, np.intc)
                elif boundary:
                    b_values_out = np.full(entities.shape, self.BOUN_IN, np.intc)
                self.edges[e]["b_values"].append(b_values_out)

        # Creating the edges meshtags
        for i, e in enumerate(self.edges):
            e_msh = self.edges[e]["submesh"]
            indices, pos = np.unique(
                np.hstack(self.edges[e]["entities"]), return_index=True
            )
            self.edges[e]["vf"] = mesh.meshtags(
                e_msh, 0, indices, np.hstack(self.edges[e]["b_values"])[pos]
            )
            e_msh.topology.create_connectivity(0, 1)
            # print("vf (", i , ")= ", self.edges[e]['vf'].values)

            if self.cfg.export:
                with io.XDMFFile(
                    self.comm, self.cfg.outdir + "/mesh/edge_" + str(i) + ".xdmf", "w"
                ) as file:
                    file.write_mesh(e_msh)
                    file.write_meshtags(self.edges[e]["vf"], e_msh.geometry)

    @timeit
    def compute_tangent(self):
        gdim = self.mesh.geometry.dim

        boun_in = []
        boun_out = []

        DG0 = fem.functionspace(self.mesh, ("DG", 0, (gdim,)))
        self.global_tangent = fem.Function(DG0)

        for i, (u, v) in enumerate(self.edges):
            edge_subdomain = self.subdomains.find(i)

            tangent = np.asarray(self.nodes[v]["pos"]) - np.asarray(
                self.nodes[u]["pos"]
            )
            tangent = tangent * 1 / np.linalg.norm(tangent)
            self.edges[u, v]["tangent"] = tangent

            # Finding BOUN_IN and BOUN_OUT dofs coordinates
            P1_e = fem.functionspace(self.edges[(u, v)]["submesh"], ("Lagrange", 1))
            dof_coords = P1_e.tabulate_dof_coordinates()
            # Translate from numpy.int32 types to native Python int
            vf_edge = [val.item() for val in self.edges[(u, v)]["vf"].values]

            if self.BOUN_IN in vf_edge:
                boun_in.append(dof_coords[vf_edge.index(self.BOUN_IN)][0:gdim])
            if self.BOUN_OUT in vf_edge:
                boun_out.append(dof_coords[vf_edge.index(self.BOUN_OUT)][0:gdim])

        # Gather boun_in and boun_out lists to proc 0
        boun_in_global = self.comm.gather(boun_in, root=0)
        boun_out_global = self.comm.gather(boun_out, root=0)
        if self.comm.rank == 0:
            assert len(boun_in_global) > 0 and len(boun_out_global) > 0, (
                "Error in submeshes markers : Need at least one inlet and one outlet"
            )

            boun_in_0 = next(boun_in for boun_in in boun_in_global if len(boun_in) > 0)
            boun_out_0 = next(
                boun_out for boun_out in boun_out_global if len(boun_out) > 0
            )

            global_dir = boun_out_0[0] - boun_in_0[0]
            global_dir[0] = 0  # Global tangent oriented in the y direction
        else:
            global_dir = None

        # Broadcast global direction from root (0) to all processors
        global_dir = self.comm.bcast(global_dir, root=0)
        global_dir_copy = copy.deepcopy(global_dir)

        for i, (u, v) in enumerate(self.edges):
            tangent = self.edges[u, v]["tangent"]
            t_dot_glob_dir = np.dot(tangent, global_dir)
            while t_dot_glob_dir == 0:  # if global_dir is perpendicular to tangent
                global_dir_copy[0] += 1
                t_dot_glob_dir = np.dot(tangent, global_dir_copy)
            global_dir_correction = t_dot_glob_dir * 1 / np.linalg.norm(t_dot_glob_dir)

            # Update tangent with corrected direction
            self.edges[u, v]["tangent"] *= global_dir_correction

            edge_subdomain = self.subdomains.find(i)
            for cell in edge_subdomain:
                self.global_tangent.x.array[gdim * cell : gdim * (cell + 1)] = (
                    self.edges[u, v]["tangent"]
                )
        self.global_tangent.x.scatter_forward()

    @property
    def mesh(self):
        if self._msh is None:
            raise RuntimeError("Mesh has not been built yet. Call build_mesh() first.")
        return self._msh

    @property
    def subdomains(self):
        if self._subdomains is None:
            raise RuntimeError("Mesh has no subdomains")
        return self._subdomains

    @property
    def boundaries(self):
        if self._facet_markers is None:
            raise RuntimeError("Mesh has no boundaries/facet markers")
        return self._facet_markers

    def submeshes(self):
        return list(nx.get_edge_attributes(self, "submesh").values())

    def tangent(self):
        if self.cfg.export:
            self.global_tangent.x.scatter_forward()
            with io.XDMFFile(
                self.comm, self.cfg.outdir + "/mesh/tangent.xdmf", "w"
            ) as file:
                file.write_mesh(self.mesh)
                file.write_function(self.global_tangent)

        return self.global_tangent
