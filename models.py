import dolfin
import ufl


def subplus(x):
    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(x):
    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


def transverse_holzapfel_ogden(
    F: ufl.Coefficient,
    f0: dolfin.Function,
    a: float = 2.280,
    b: float = 9.726,
    a_f: float = 1.685,
    b_f: float = 15.779,
) -> ufl.Coefficient:
    C = F.T * F
    I1 = dolfin.tr(C)
    I4f = dolfin.inner(C * f0, f0)

    return (a / (2.0 * b)) * (dolfin.exp(b * (I1 - 3)) - 1.0) + (
        a_f / (2.0 * b_f)
    ) * heaviside(I4f - 1) * (dolfin.exp(b_f * subplus(I4f - 1) ** 2) - 1.0)


def active_stress_energy(
    F: ufl.Coefficient, f0: dolfin.Function, Ta: dolfin.Constant
) -> ufl.Coefficient:
    I4f = dolfin.inner(F * f0, F * f0)
    return 0.5 * Ta * (I4f - 1)


def compressibility(F: ufl.Coefficient, kappa: float = 1e3) -> ufl.Coefficient:
    J = dolfin.det(F)
    return kappa * (J * dolfin.ln(J) - J + 1)


def incompressibility(F: ufl.Coefficient, p: dolfin.Function) -> ufl.Coefficient:
    J = dolfin.det(F)
    return p * (J - 1)
