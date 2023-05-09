import logging
import time
import dolfin

logger = logging.getLogger(__name__)


def build_elastic_nullspace(V, x):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh().geometry().dim()
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(dim)]

    # Build translational null space basis
    for i in range(gdim):
        V.sub(i).dofmap().set(nullspace_basis[i], 1.0)

    # Build rotational null space basis
    if gdim == 2:
        V.sub(0).set_x(nullspace_basis[2], -1.0, 1)
        V.sub(1).set_x(nullspace_basis[2], 1.0, 0)
    elif gdim == 3:
        V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
        V.sub(1).set_x(nullspace_basis[3], 1.0, 0)

        V.sub(0).set_x(nullspace_basis[4], 1.0, 2)
        V.sub(2).set_x(nullspace_basis[4], -1.0, 0)

        V.sub(2).set_x(nullspace_basis[5], 1.0, 1)
        V.sub(1).set_x(nullspace_basis[5], -1.0, 2)

    for x in nullspace_basis:
        x.apply("insert")

    return dolfin.VectorSpaceBasis(nullspace_basis)


class NonlinearProblem(dolfin.NonlinearProblem):
    def __init__(self, J, F, bcs, V, **kwargs):
        super().__init__(**kwargs)
        self._J = J
        self._F = F
        if not isinstance(bcs, (list, tuple)):
            bcs = [bcs]
        self.bcs = bcs
        self.verbose = True
        self.V = V

    def F(self, b: dolfin.PETScVector, x: dolfin.PETScVector):
        logger.debug("Calling F")
        dolfin.assemble(self._F, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A: dolfin.PETScMatrix, x: dolfin.PETScVector):
        logger.debug("Calling J")
        dolfin.assemble(self._J, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

        x = dolfin.Vector()
        A.init_vector(x, 1)
        nullspace = build_elastic_nullspace(V=self.V, x=x)

        A.set_near_nullspace(nullspace)


class NonlinearSolver:
    def __init__(
        self,
        problem: NonlinearProblem,
        state,
        parameters=None,
    ):
        dolfin.PETScOptions.clear()
        self.update_parameters(parameters)
        self._problem = problem
        self._state = state

        self._solver = dolfin.PETScSNESSolver(dolfin.MPI.comm_world)
        self._solver.set_from_options()

        self._solver.parameters.update(self.parameters)
        self._snes = self._solver.snes()
        self._snes.setConvergenceHistory()

        logger.debug(f"Linear Solver : {self._solver.parameters['linear_solver']}")
        logger.debug(f"Preconditioner:  {self._solver.parameters['preconditioner']}")
        logger.debug(f"atol: {self._solver.parameters['absolute_tolerance']}")
        logger.debug(f"rtol: {self._solver.parameters['relative_tolerance']}")
        logger.debug(f" Size          : {self._state.function_space().dim()}")
        dolfin.PETScOptions.clear()

    def update_parameters(self, parameters):
        ps = NonlinearSolver.default_solver_parameters()
        if hasattr(self, "parameters"):
            ps.update(self.parameters)
        if parameters is not None:
            ps.update(parameters)
        petsc = ps.pop("petsc")

        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)

        self.verbose = ps.pop("verbose", False)

        if self.verbose:
            dolfin.PETScOptions.set("ksp_monitor")
            dolfin.PETScOptions.set("log_view")
            dolfin.PETScOptions.set("ksp_view")
            dolfin.PETScOptions.set("pc_view")
            dolfin.PETScOptions.set("mat_superlu_dist_statprint", True)

        self.parameters = ps

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
            },
            "verbose": True,
        }

    def solve(self):
        logger.debug(" Solving NonLinearProblem ...")

        start = time.perf_counter()
        self._solver.solve(self._problem, self._state.vector())
        end = time.perf_counter()

        logger.debug(f" ... Done in {end - start:.3f} s")

        residuals = self._snes.getConvergenceHistory()[0]
        num_iterations = self._snes.getLinearSolveIterations()
        logger.debug(f"Iterations    : {num_iterations}")
        if num_iterations > 0:
            logger.debug(f"Resiudal      : {residuals[-1]}")

        return num_iterations, self._snes.converged


class NewtonSolver(dolfin.NewtonSolver):
    def __init__(
        self,
        problem: NonlinearProblem,
        state: dolfin.Function,
        update_cb=None,
        parameters=None,
        use_lu: bool = True,
    ):
        logger.debug(f"Initialize NewtonSolver with parameters: {parameters!r}")
        dolfin.PETScOptions.clear()
        self._problem = problem
        self._state = state
        self._update_cb = update_cb

        if use_lu:
            petsc_solver = dolfin.PETScLUSolver()
        else:
            petsc_solver = dolfin.PETScKrylovSolver()

        super().__init__(
            self._state.function_space().mesh().mpi_comm(),
            petsc_solver,
            dolfin.PETScFactory.instance(),
        )
        self._handle_parameters(parameters)

    def _handle_parameters(self, parameters):
        # Setting default parameters
        params = NewtonSolver.default_solver_parameters()
        params.update(parameters)

        for k, v in params.items():
            if self.parameters.has_parameter(k):
                self.parameters[k] = v
            if self.parameters.has_parameter_set(k):
                for subk, subv in params[k].items():
                    self.parameters[k][subk] = subv

        petsc = params.pop("petsc")
        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)

        self.newton_verbose = params.pop("newton_verbose", False)
        self.ksp_verbose = params.pop("ksp_verbose", False)
        self.debug = params.pop("debug", False)
        if self.newton_verbose:
            dolfin.set_log_level(dolfin.LogLevel.INFO)
            self.parameters["report"] = True
        if self.ksp_verbose:
            self.parameters["lu_solver"]["report"] = True
            self.parameters["lu_solver"]["verbose"] = True
            self.parameters["krylov_solver"]["monitor_convergence"] = True
            dolfin.PETScOptions.set("ksp_monitor_true_residual")
        self.linear_solver().set_from_options()

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "preonly",
                # "ksp_type": "cg",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
            },
            "newton_verbose": True,
            "ksp_verbose": True,
            "debug": True,
            "linear_solver": "mumps",
            "error_on_nonconvergence": True,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 20,
            "report": True,
            "krylov_solver": {
                "nonzero_initial_guess": True,
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": True, "symmetric": False, "verbose": True},
        }

    def converged(self, r, p, i):
        self._converged_called = True

        res = r.norm("l2")
        logger.debug(f"Mechanics solver residual: {res}")

        return super().converged(r, p, i)

    def solve(self):
        logger.debug("Solving mechanics")
        self._solve_called = True

        ret = super().solve(self._problem, self._state.vector())

        self._state.vector().apply("insert")
        logger.debug("Done solving mechanics")

        return ret
