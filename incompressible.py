import logging
import dolfin
import ufl
import models
import solvers


def main():
    use_snes = False

    # Create a Unit Cube Mesh
    mesh = dolfin.UnitCubeMesh(10, 10, 10)

    # Function space for the displacement
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = dolfin.FunctionSpace(mesh, P2 * P1)
    # The displacement
    state = dolfin.Function(V)
    # Test function for the displacement
    state_test = dolfin.TestFunction(V)

    u, p = dolfin.split(state)
    u_test, p_tests = dolfin.split(state_test)

    # Compute the deformation gradient
    F = dolfin.grad(u) + dolfin.Identity(3)

    # Active tension
    Ta = dolfin.Constant(1.0)
    # Set fiber direction to be constant in the x-direction
    f0 = dolfin.Constant([1.0, 0.0, 0.0])

    # Collect the contributions to the total energy (here using the Holzapfel Ogden model)
    elastic_energy = (
        models.transverse_holzapfel_ogden(F, f0=f0)
        + models.active_stress_energy(F, f0, Ta)
        + models.incompressibility(F, p)
    )

    # Define some subdomain. Here we mark the x = 0 plane with the marker 1
    left = dolfin.CompiledSubDomain("near(x[0], 0)")
    left_marker = 1
    # And we define the Dirichlet boundary condition on this side
    # We specify that the displacement should be zero in all directions
    bcs = dolfin.DirichletBC(V.sub(0), dolfin.Constant((0.0, 0.0, 0.0)), left)

    # We also define a subdomain on the opposite wall
    right = dolfin.CompiledSubDomain("near(x[0], 1)")
    # and we give a marker of two
    right_marker = 2

    # We create a facet function for marking the facets
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    # We set all values to zero
    ffun.set_all(0)
    # Then then mark the left and right subdomains
    left.mark(ffun, left_marker)
    right.mark(ffun, right_marker)

    # Now we can form to total internal virtual work which is the
    # derivative of the energy in the system
    quad_degree = 4
    internal_virtual_work = dolfin.derivative(
        elastic_energy * dolfin.dx(metadata={"quadrature_degree": quad_degree}),
        state,
        state_test,
    )

    # We can also apply a force on the right boundary using a Neumann boundary condition
    traction = dolfin.Constant(-0.5)
    N = dolfin.FacetNormal(mesh)
    n = traction * ufl.cofac(F) * N
    ds = dolfin.ds(domain=mesh, subdomain_data=ffun)
    external_virtual_work = dolfin.inner(u_test, n) * ds(right_marker)

    # The total virtual work is the sum of the internal and external virtual work
    total_virtual_work = internal_virtual_work + external_virtual_work

    jacobian = dolfin.derivative(
        total_virtual_work,
        state,
        dolfin.TrialFunction(V),
    )

    problem = solvers.NonlinearProblem(
        J=jacobian, F=total_virtual_work, bcs=bcs, V=V.sub(0)
    )
    if use_snes:
        solver_parameters = solvers.NonlinearSolver.default_solver_parameters()
    else:
        solver_parameters = solvers.NewtonSolver.default_solver_parameters()

    solver_parameters["petsc"]["ksp_rtol"] = 1e-8
    solver_parameters["petsc"]["pc_gamg_type"] = "agg"
    solver_parameters["petsc"]["pc_gamg_sym_graph"] = True
    solver_parameters["petsc"]["matptap_via"] = "scalable"
    solver_parameters["petsc"]["pc_gamg_threshold"] = 0.02

    solver_parameters["petsc"]["pc_type"] = "gamg"
    solver_parameters["petsc"]["pc_gamg_agg_nsmooths"] = 1
    solver_parameters["petsc"]["pc_gamg_square_graph"] = 2
    solver_parameters["petsc"]["pc_gamg_coarse_eq_limit"] = 2000
    solver_parameters["petsc"]["pc_gamg_esteig_ksp_type"] = "cg"
    solver_parameters["petsc"]["pc_gamg_esteig_ksp_max_it"] = 20
    solver_parameters["petsc"]["pc_gamg_threshold"] = 0.01
    solver_parameters["petsc"]["mg_levels_ksp_type"] = "chebyshev"
    solver_parameters["petsc"]["mg_levels_esteig_ksp_type"] = "cg"
    solver_parameters["petsc"]["mg_levels_esteig_ksp_max_it"] = 20
    solver_parameters["petsc"]["mg_levels_ksp_max_it"] = 5
    solver_parameters["petsc"]["mg_levels_pc_type"] = "jacobi"

    if use_snes:
        solver = solvers.NonlinearSolver(
            problem=problem,
            state=state,
            parameters=solver_parameters,
        )
        nliter, nlconv = solver.solve()
    else:
        solver = solvers.NewtonSolver(
            problem=problem,
            state=state,
            parameters=solver_parameters,
        )
        ret = solver.solve()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
