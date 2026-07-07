#  1D Gross-Pitaevskii equation solver
#  for a harmonic oscillator using JAX
#
#  Equation:
#      [ -1/2 d^2/dx^2 + V(x) + g|psi|^2 ] psi = i d(psi)/dt
#      (in natutal units)
#
#  Split-step Fourier Method (Strang splitting)


import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from jax import Array

from functools import partial


# Grid
def make_grid(N: int, L: float) -> tuple[Array, Array, float]:
    """
    Uniform grid on [-L/2, L/2) and
    the matching FFT angular wavenumbers
    """
    dx = L / N
    x = -L / 2 + dx * jnp.arange(N)
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
    return x, k, dx


# Harmonic Potential
def harmonic_potential(x: Array) -> Array:
    return 0.5 * x**2


# Wavefunction norm
def norm_wf(psi: Array, dx: float) -> Array:
    return jnp.sum(jnp.abs(psi) ** 2) * dx


# Energy functional
def energy_func(psi: Array, k: Array, V: Array, g: float, dx: float) -> Array:
    """
    E = integral[ 1/2|d(psi)/dx|^2 + V|psi|^2 + g/2|psi|^4 ] dx
    """
    dpsi_dx = jnp.fft.ifft(1j * k * jnp.fft.fft(psi))
    e_kin = 0.5 * jnp.sum(jnp.abs(dpsi_dx) ** 2) * dx
    e_pot = jnp.sum(V * jnp.abs(psi) ** 2) * dx
    e_int = 0.5 * g * jnp.sum(jnp.abs(psi) ** 4) * dx
    return e_kin + e_pot + e_int


# Imaginary time evolution
@partial(jax.jit, static_argnames=("N_steps"))
def imaginary_time_evolution(
    psi0: Array,
    k: Array,
    V: Array,
    g: float,
    dx: float,
    N_particles: int,
    dtau: float,
    N_steps: int,
) -> tuple[Array, Array]:
    k2 = k**2
    psi0 = psi0 * jnp.sqrt(N_particles / norm_wf(psi0, dx))

    def step(psi: Array, _) -> tuple[Array, Array]:
        v_nl = V + g * jnp.abs(psi) ** 2
        psi = psi * jnp.exp(-0.5 * dtau * v_nl)
        psi = jnp.fft.ifft(jnp.fft.fft(psi) * jnp.exp(-0.5 * dtau * k2))
        v_nl = V + g * jnp.abs(psi) ** 2
        psi = psi * jnp.exp(-0.5 * dtau * v_nl)
        psi = psi * jnp.sqrt(
            N_particles / norm_wf(psi, dx)
        )  # imaginary time is not norm-preserving
        E = energy_func(psi, k, V, g, dx)
        return psi, E

    psi_final, energy_history = lax.scan(step, psi0, None, length=N_steps)
    return psi_final, energy_history


# Main function
def main() -> None:
    N = 1024  # grid points
    L = 20.0  # box length
    g = 5.0  # interaction strength (>0 repulsive, <0 attractive)
    N_particles = 1.0

    x, k, dx = make_grid(N, L)
    V = harmonic_potential(x)

    dtau = 1e-3
    N_steps = 10_000

    psi_guess = jnp.exp(-(x**2) / 2).astype(jnp.complex128)

    psi_ground, E_history = imaginary_time_evolution(
        psi_guess, k, V, g, dx, N_particles, dtau, N_steps
    )
    print(f"[ground state] energy = {float(E_history[-1]):.6f}")
    print(f"               |dE| over last 1000 steps = {float(abs(E_history[-1]-E_history[-1000])):.2e}")
    print(f"               norm = {float(norm_wf(psi_ground, dx)):.10f}")
    

    # validation: g=0 case has an exact analytic ground state
    psi0_lin, E_history = imaginary_time_evolution(
        psi_guess, k, V, 0.0, dx, N_particles, dtau, N_steps
    )
    analytic_gaussian = (1 / jnp.pi**0.25) * jnp.exp(-(x**2) / 2)
    lin_error = float(jnp.max(jnp.abs(jnp.abs(psi0_lin) - analytic_gaussian)))
    print()
    print(f"[validation]  max|psi_numeric - psi_analytic| for g=0: {lin_error:.2e}")
    print(f"              energy = {float(E_history[-1]):.6f}")


if __name__ == "__main__":
    main()
