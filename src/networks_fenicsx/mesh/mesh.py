"""
Interface for converting a networkx graph into a {py:class}`dolfinx.mesh.Mesh`.

This idea stems from the Graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks
by Ingeborg Gjerde.

Modified by Cécile Daversin-Catty - 2023
Modified by Joseph P. Dean - 2023
Modified by Jørgen S. Dokken - 2025
"""

import networkx as nx
import numpy as np
import basix.ufl
from mpi4py import MPI
from dolfinx import fem, io as _io, mesh, graph as _graph
import ufl

from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

__all__ = ["NetworkMesh", "compute_tangent"]


class NetworkMesh:
    """A representation of a Networkx graph in DOLFINx.

    Stores the resulting mesh, subdomains, and facet markers for bifurcations and boundary nodes.
    Has a globally oriented tangent vector field.
    Has a submesh for each edge in the Networkx graph.
    """

    # Configuration
    _cfg: config.Config

    # Graph properties
    _geom_dim: int
    _num_segments: int

    # Mesh properties
    _msh: mesh.Mesh | None
    _subdomains: mesh.MeshTags | None
    _facet_markers: mesh.MeshTags | None
    _edge_meshes = list[mesh.Mesh]
    _edge_entity_maps = list[mesh.EntityMap]
    _tangent: fem.Function

    def __init__(self, graph: nx.DiGraph, config: config.Config):
        self._cfg = config
        self._cfg.clean_dir()
        self._build_mesh(graph)
        self._build_network_submeshes()
        self._tangent = compute_tangent(self.mesh)

    @property
    def cfg(self) -> config.Config:
        return self._cfg

    @timeit
    def _build_mesh(self, graph: nx.DiGraph):
        """Convert the networkx graph into a {py:class}`dolfinx.mesh.Mesh`.

        The element size is controlled by `self.cfg.lcar`.
        Each segment in the networkx graph gets a unique subdomain marker.
        Each bifurcation and boundary node is marked on the facets with a unique integer.

        Note:
            This function attaches data to `self.mesh`, `self.subdomains` and
            `self.boundaries`.
        """

        self._geom_dim = len(graph.nodes[1]["pos"])

        vertex_coords = np.asarray([graph.nodes[v]["pos"] for v in graph.nodes()])
        cells_array = np.asarray([[u, v] for u, v in graph.edges()])

        line_weights = np.linspace(
            0, 1, int(np.ceil(1 / self.cfg.lcar)), endpoint=False
        )[1:][:, None]

        self._num_segments = cells_array.shape[0]
        # Create mesh segments
        # TODO: Extract graph coloring coloring information to reduce the number of unique cell markers,
        # which results in a reduction in the number of submeshes.
        if MPI.COMM_WORLD.rank == 0:
            mesh_nodes = vertex_coords.copy()
            cells = []
            cell_markers = []
            for i, segment in enumerate(cells_array):
                if len(line_weights) == 0:
                    cells.append([segment[0], segment[1]])
                    cell_markers.append(i)
                else:
                    start_coord_pos = mesh_nodes.shape[0]
                    start = vertex_coords[segment[0]]
                    end = vertex_coords[segment[1]]

                    internal_line_coords = (
                        start * (1 - line_weights) + end * line_weights
                    )
                    mesh_nodes = np.vstack((mesh_nodes, internal_line_coords))
                    cells.append([segment[0], start_coord_pos])
                    segment_connectivity = (
                        np.repeat(np.arange(internal_line_coords.shape[0]), 2)[
                            1:-1
                        ].reshape(internal_line_coords.shape[0] - 1, 2)
                        + start_coord_pos
                    )
                    cells.append(segment_connectivity)
                    cells.append(
                        [
                            start_coord_pos + internal_line_coords.shape[0] - 1,
                            segment[1],
                        ]
                    )
                    cell_markers.extend(
                        np.full(internal_line_coords.shape[0] + 1, i, dtype=np.int32)
                    )
            cells = np.vstack(cells).astype(np.int64)
            cell_markers = np.array(cell_markers, dtype=np.int32)
        else:
            cells = np.empty((0, 2), dtype=np.int64)
            mesh_nodes = np.empty((0, self.geom_dim), dtype=np.float64)
            cell_markers = np.empty((0,), dtype=np.int32)

        bifurcations, num_connections = np.unique(
            cells_array.flatten(), return_counts=True
        )
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
        graph_mesh = mesh.create_mesh(
            MPI.COMM_WORLD,
            x=mesh_nodes,
            cells=cells,
            e=ufl.Mesh(
                basix.ufl.element("Lagrange", "interval", 1, shape=(self.geom_dim,))
            ),
            partitioner=partitioner,
            max_facet_to_cell_links=np.max(num_connections),
        )
        self._msh = graph_mesh

        local_entities, local_values = _io.distribute_entity_data(
            self.mesh,
            self.mesh.topology.dim,
            cells,
            cell_markers,
        )
        self._subdomains = mesh.meshtags_from_entities(
            self.mesh,
            self.mesh.topology.dim,
            _graph.adjacencylist(local_entities),
            local_values,
        )

        if MPI.COMM_WORLD.rank == 0:
            lb = bifurcations.astype(np.int64).reshape((-1, 1))
            lbv = np.arange(len(bifurcations), dtype=np.int32)
        else:
            lb = np.empty((0, 1), dtype=np.int64)
            lbv = np.empty((0,), dtype=np.int32)
        self.mesh.topology.create_connectivity(0, 1)
        local_bifurcations, local_bifurcation_values = _io.distribute_entity_data(
            self.mesh, 0, lb, lbv
        )
        self._facet_markers = mesh.meshtags_from_entities(
            self.mesh,
            0,
            _graph.adjacencylist(local_bifurcations),
            local_bifurcation_values,
        )
        self.subdomains.name = "subdomains"
        self.boundaries.name = "bifurcations"
        if self.cfg.export:
            with _io.XDMFFile(
                self.comm, self.cfg.outdir + "/mesh/mesh.xdmf", "w"
            ) as file:
                file.write_mesh(self.mesh)
                file.write_meshtags(self.subdomains, self.mesh.geometry)
                file.write_meshtags(self.boundaries, self.mesh.geometry)

    @timeit
    def _build_network_submeshes(self):
        """Create submeshes for each edge in the network."""
        assert self._msh is not None
        assert self._subdomains is not None
        assert self._facet_markers is not None
        assert len(self._edge_meshes) == 0
        assert len(self._edge_entity_maps) == 0
        for i in range(self._num_segments):
            edge_subdomain = self.subdomains.array[self.subdomains.indices == i]

            edge_mesh, edge_map = mesh.create_submesh(
                self.mesh, self.mesh.topology.dim, edge_subdomain
            )[0:2]
            self._edge_meshes.append(edge_mesh)
            self._edge_entity_maps.append(edge_map)

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

    @property
    def submeshes(self):
        if len(self._edge_meshes) == 0:
            raise RuntimeError(
                "Submeshes have not been built yet. Call build_network_submeshes() first."
            )
        return self._edge_meshes

    @property
    def tangent(self):
        return self._tangent

    def export_tangent(self):
        if self.cfg.export:
            with _io.XDMFFile(
                self.comm, self.cfg.outdir + "/mesh/tangent.xdmf", "w"
            ) as file:
                file.write_mesh(self.mesh)
                file.write_function(self.tangent)
        else:
            print("Export of tangent skipped as cfg.export is set to False.")


def compute_tangent(domain: mesh.Mesh) -> fem.Function:
    """Compute tangent vector for all cells.

    Tangent is oriented according to positive y-axis.
    If perpendicular to y-axis, align with x-axis.

    Note:
        Assuming that the mesh is affine.
    """
    cell_map = domain.topology.index_map(domain.topology.dim)
    geom_indices = mesh.entities_to_geometry(
        domain,
        domain.topology.dim,
        np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32),
    )
    geom_coordinates = domain.geometry.x[geom_indices]
    tangent = geom_coordinates[:, 0, :] - geom_coordinates[:, 1, :]
    global_orientation = np.sign(np.dot(tangent, [0, 1, 0]))
    is_x_aligned = np.isclose(global_orientation, 0)
    global_orientation[is_x_aligned] = np.sign(np.dot(tangent[is_x_aligned], [1, 0, 0]))
    tangent *= global_orientation[:, None]
    assert np.all(np.linalg.norm(tangent, axis=1) > 0), (
        "Zero-length tangent vector detected"
    )
    gdim = domain.geometry.dim
    DG0 = fem.functionspace(domain, ("DG", 0, (gdim,)))
    global_tangent = fem.Function(DG0)
    global_tangent.x.array[:] = tangent.flatten()
    return global_tangent
