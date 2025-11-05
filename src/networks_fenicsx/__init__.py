"""Top-level package for 1D networks simulations with FEniCSx."""

__author__ = """Cécile Daversin-Catty"""
__email__ = "cecile@simula.no"
__version__ = "0.1.0"

from .mesh.mesh import NetworkMesh
from .solver.assembly import HydraulicNetworkAssembler
__all__ = ["HydraulicNetworkAssembler", "NetworkMesh"]
