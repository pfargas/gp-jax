#  1D Gross-Pitaevskii equation solver
#  for a harmonic oscillator using JAX
#
#  Equation:
#      i d(psi)/dt = [ -1/2 d^2/dx^2 + V(x) + g|psi|^2 ]
#      (in natutal units)
#
#  Split-step Fourier method (with Strang splitting)


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
    return jnp.sum(psi.real**2 + psi.imag**2) * dx


# Energy functional
def energy_func(
    psi: Array, k: Array, V: Array, g: float, dx: float, grid_scale: float
) -> Array:
    """
    E = integral[ 1/2|d(psi)/dx|^2 + V|psi|^2 + g/2|psi|^4 ] dx
    """
    psi_sq = psi.real**2 + psi.imag**2

    psi_k = jnp.fft.fft(psi)
    psi_k_sq = psi_k.real**2 + psi_k.imag**2
    e_kin = 0.5 * jnp.sum(k**2 * psi_k_sq) * grid_scale

    e_pot = jnp.sum(V * psi_sq) * dx
    e_int = 0.5 * g * jnp.sum(psi_sq**2) * dx
    return e_kin + e_pot + e_int


# Imaginary time evolution
@partial(jax.jit, static_argnames=("N_steps",))
def imaginary_time_evolution(
    psi0: Array,
    k: Array,
    V: Array,
    g: float,
    dx: float,
    N_particles: float,
    dtau: float,
    N_steps: int,
) -> tuple[Array, Array]:

    minus_half_dtau = -0.5 * dtau
    kinetic_factor = jnp.exp(minus_half_dtau * k**2)

    grid_scale = dx / k.shape[0]

    psi0 = psi0 * jnp.sqrt(N_particles / norm_wf(psi0, dx))

    def step(psi: Array, _) -> tuple[Array, Array]:
        v_nl = V + g * (psi.real**2 + psi.imag**2)
        psi = psi * jnp.exp(minus_half_dtau * v_nl)

        psi = jnp.fft.ifft(jnp.fft.fft(psi) * kinetic_factor)

        v_nl = V + g * (psi.real**2 + psi.imag**2)
        psi = psi * jnp.exp(minus_half_dtau * v_nl)

        psi = psi * jnp.sqrt(N_particles / norm_wf(psi, dx))

        E = energy_func(psi, k, V, g, dx, grid_scale)
        return psi, E

    psi_final, energy_history = lax.scan(step, psi0, None, length=N_steps)
    return psi_final, energy_history


# Main function
def main() -> None:
    N = 2048  # grid points
    L = 20.0  # box length
    g = 5.0  # interaction strength (>0 repulsive, <0 attractive)
    N_particles = 10

    x, k, dx = make_grid(N, L)
    V = harmonic_potential(x)

    dtau = 5e-4
    N_steps = 20_000

    psi_guess = jnp.exp(-(x**2) / 2).astype(jnp.complex128)

    psi_ground, E_history = imaginary_time_evolution(
        psi_guess, k, V, g, dx, N_particles, dtau, N_steps
    )
    print(f"[ground state]")
    print(f"energy = {float(E_history[-1]):.6f}")
    print(
        f"|dE| over last 1000 steps = {float(abs(E_history[-1]-E_history[-1000])):.2e}"
    )
    print(f"norm = {float(norm_wf(psi_ground, dx)):.10f}")

    # validation: g=0 case has an exact analytic ground state
    psi0, E_history = imaginary_time_evolution(
        psi_guess, k, V, 0.0, dx, N_particles, dtau, N_steps
    )
    analytic_gaussian = (
        jnp.sqrt(N_particles) * (1 / jnp.pi**0.25) * jnp.exp(-(x**2) / 2)
    )
    error = float(jnp.max(jnp.abs(jnp.abs(psi0) - analytic_gaussian)))
    print(f"[validation]")
    print(f"energy = {float(E_history[-1]):.6f}")
    print(f"max|psi_numeric - psi_analytic| for g=0: {error:.2e}")


if __name__ == "__main__":
    main()
