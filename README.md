# MPI compatible implementation of graph networks in FEniCSx

![formatting](https://github.com/cdaversin/networks_fenicsx/actions/workflows/check_formatting.yml/badge.svg)
![pytest](https://github.com/cdaversin/networks_fenicsx/actions/workflows/test_package.yml/badge.svg)


This repository contains a re-implementation of [GraphNics](https://github.com/IngeborgGjerde/graphnics/)
by I. Gjerde (DOI: [10.48550/arXiv.2212.02916](https://doi.org10.48550/arXiv.2212.02916)).

An initial implementation compatible with [DOLFINx](https://github.com/FEniCS/dolfinx/)
(I. Baratta et al, DOI: [10.5281/zenodo.10447665](https://doi.org/10.5281/zenodo.10447665)) with performance benchmarks presented by C. Daversin-Catty et al. in
[Finite Element Software and Performance for Network Models with Multipliers](https://doi.org/10.1007/978-3-031-58519-7_4).

However, this implementation was not **MPI compatible**.
This repository contains an **MPI compatible** implementation of graph support in DOLFINx (v0.10.0).

Please cite usage of this repository by using the [CITATION-file](./CITATION.cff).
