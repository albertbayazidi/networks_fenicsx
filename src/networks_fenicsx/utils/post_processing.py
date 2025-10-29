from dolfinx import fem, io
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np

from networks_fenicsx.mesh import mesh
from networks_fenicsx import config
from networks_fenicsx.utils.mesh_utils import transfer_submesh_data


def export(
    config: config.Config,
    graph: mesh.NetworkGraph,
    function_spaces: list,
    sol,
    export_dir="results",
):
    q_space = function_spaces[0]
    p_space = function_spaces[-1]
    q_degree = q_space.element.basix_element.degree
    global_q_space = fem.functionspace(graph.msh, ("DG", q_degree))
    global_q = fem.Function(global_q_space)

    # Recover solution
    fluxes = []
    start = 0
    for i, e in enumerate(graph.edges):
        q_space = function_spaces[i]
        q = fem.Function(q_space)
        offset = q_space.dofmap.index_map.size_local * q_space.dofmap.index_map_bs
        q.x.array[:offset] = sol.array_r[start : start + offset]
        q.x.scatter_forward()
        start += offset
        fluxes.append(q)

        # Interpolated to DG space
        DG_q_space = fem.functionspace(q_space.mesh, ("DG", q_degree))
        DG_q = fem.Function(DG_q_space)
        DG_q.interpolate(q)
        # Transferring from DG submesh to DG parent
        transfer_submesh_data(
            global_q, DG_q, graph.edges[e]["entity_map"], inverse=True
        )

    offset = p_space.dofmap.index_map.size_local * p_space.dofmap.index_map_bs
    pressure = fem.Function(p_space)
    pressure.x.array[: (len(sol.array_r) - start)] = sol.array_r[start : start + offset]
    pressure.x.scatter_forward()

    # Write to file
    for i, q in enumerate(fluxes):
        with io.VTXWriter(
            MPI.COMM_WORLD,
            config.outdir + "/" + export_dir + "/flux_" + str(i) + ".bp",
            q,
        ) as f:
            f.write(0.0)
    with io.VTXWriter(
        MPI.COMM_WORLD, config.outdir + "/" + export_dir + "/flux.bp", global_q
    ) as f:
        f.write(0.0)
    with io.VTXWriter(
        MPI.COMM_WORLD, config.outdir + "/" + export_dir + "/pressure.bp", pressure
    ) as f:
        f.write(0.0)

    return (fluxes, global_q, pressure)


def perf_plot(timing_dict):
    # set width of bar
    barWidth = 0.1
    # fig = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(timing_dict["n"]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(
        br1,
        timing_dict["compute_forms"],
        color="r",
        width=barWidth,
        edgecolor="grey",
        label="forms",
    )
    plt.bar(
        br2,
        timing_dict["assemble"],
        color="g",
        width=barWidth,
        edgecolor="grey",
        label="assembly",
    )
    plt.bar(
        br3,
        timing_dict["solve"],
        color="b",
        width=barWidth,
        edgecolor="grey",
        label="solve",
    )

    # Adding Xticks
    plt.xlabel("Number of generations", fontweight="bold", fontsize=15)
    plt.ylabel("Time [s]", fontweight="bold", fontsize=15)
    plt.xticks([r + barWidth for r in range(len(timing_dict["n"]))], timing_dict["n"])

    plt.legend()
    plt.show()
