"""
Space Suit Thermal & Radiation Analysis for LEO Conditions
===========================================================

OBJECTIVES:
  1. 1D transient thermal conduction in multi-layer space suit materials
  2. Radiation attenuation modeling using Beer-Lambert law
  3. Performance summary under orbital heating/eclipse cycles

PHYSICS ASSUMPTIONS & SIMPLIFICATIONS:
  - No contact resistance between layers (perfect thermal contact)
  - Constant material properties (temperature-independent)
  - Neglect inter-layer radiation (relative to conduction dominates)
  - No convection (vacuum environment)
  - 1D heat flow perpendicular to suit surface
  - Crank-Nicolson scheme: unconditionally stable, suitable for <10s runtime
  - Uniform grid spacing (simplified from adaptive meshing)
  - Radiation model: high-energy particles (e.g., X-rays, cosmic rays ~1 MeV)

DISCLAIMER: Simplified model for demonstration. Real analysis requires:
  - Temperature-dependent properties
  - Multi-directional heat flow
  - Contact resistance
  - Detailed solar spectrum integration
  - Cosmic ray spectrum