#  1D Gross-Pitaevskii equation solver
#  for a harmonic oscillator using JAX
#
#  Equation:
#      i d(psi)/dt = [ -1/2 d^2/dx^2 + V(x) + g|psi|^2 ] psi
#      (in natutal units)
#
#  Split-step Fourier method (with Strang splitting)


import jax

# 64-bit floats (double precision)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from jax import Array

from functools import partial

import numpy as np
import matplotlib.pyplot as plt


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
    return jnp.sum(psi**2) * dx


# Energy functional
@jax.jit
def energy_func(
    psi: Array, k: Array, V: Array, g: float, dx: float, dx_N: float
) -> Array:
    """
    E = integral[ 1/2|d(psi)/dx|^2 + V|psi|^2 + g/2|psi|^4 ] dx
    """
    psi_sq = psi**2

    # using Parseval's theorem
    psi_k = jnp.fft.fft(psi)
    psi_k_sq = psi_k.real**2 + psi_k.imag**2
    e_kin = 0.5 * jnp.sum(k**2 * psi_k_sq) * dx_N

    e_pot = jnp.sum(V * psi_sq) * dx
    e_int = 0.5 * g * jnp.sum(psi_sq**2) * dx
    return e_kin + e_pot + e_int


# Imaginary time evolution
@partial(jax.jit, static_argnames=("N_steps",))
def imaginary_time_evolution(
    psi0: Array,
    k_rfft: Array,
    V: Array,
    g: float,
    dx: float,
    N_particles: int,
    dtau: float,
    N_steps: int,
) -> tuple[Array, Array]:

    minus_half_dtau = -0.5 * dtau

    # only positive wavenumbers for Real FFT
    kinetic_factor = jnp.exp(minus_half_dtau * k_rfft**2)

    # precompute static exponential factors once
    minus_half_dtau_V = minus_half_dtau * V
    minus_half_dtau_g = minus_half_dtau * g

    # using fast inverse square root
    psi0 = psi0 * jax.lax.rsqrt(norm_wf(psi0, dx) / N_particles)
    n = psi0.shape[0]

    def step(psi: Array, _) -> tuple[Array, Array]:
        # first half-step potential
        psi = psi * jnp.exp(minus_half_dtau_V + minus_half_dtau_g * psi**2)

        # full-step kinetic (using Real FFT)
        psi = jnp.fft.irfft(jnp.fft.rfft(psi) * kinetic_factor, n=n)

        # second half-step potential
        rho = psi**2
        S = jnp.exp(minus_half_dtau_V + minus_half_dtau_g * rho)
        psi = psi * S

        # norm calculation from rho and S
        current_norm = jnp.sum(rho * S**2) * dx

        psi = psi * jax.lax.rsqrt(current_norm / N_particles)

        return psi, current_norm

    psi_final, norm_history = lax.scan(step, psi0, None, length=N_steps)

    return psi_final, norm_history


# Main function
def main() -> None:
    N = 8192  # grid points (2^n for FFT)
    L = 20.0  # box length
    g = 5.0   # interaction strength (>0 repulsive, <0 attractive)
    N_particles = 10

    x, k, dx = make_grid(N, L)

    # positive wavenumbers for Real FFT
    k_rfft = k[: N // 2 + 1]

    dx_N = dx / N

    V = harmonic_potential(x)

    dtau = 5e-4
    N_steps = 20_000

    psi_trial = jnp.exp(-0.5 * x**2)

    psi_ground, norm_history = imaginary_time_evolution(
        psi_trial, k_rfft, V, g, dx, N_particles, dtau, N_steps
    )
    E_final = energy_func(psi_ground, k, V, g, dx, dx_N)

    print(f"[ground state]")
    print(f"energy = {float(E_final):.6f}")
    print(f"norm = {float(norm_wf(psi_ground, dx)):.6f}")

    # validation: g=0 case E=N/2
    psi_g0, _ = imaginary_time_evolution(
        psi_trial, k_rfft, V, 0.0, dx, N_particles, dtau, N_steps
    )
    E_final_g0 = energy_func(psi_g0, k, V, 0.0, dx, dx_N)
    analytic_gaussian = (
        jnp.sqrt(N_particles) * (1 / jnp.pi**0.25) * jnp.exp(-0.5 * x**2)
    )
    error = float(jnp.max(jnp.abs(psi_g0 - analytic_gaussian)))

    print(f"[validation]")
    print(f"energy = {float(E_final_g0):.6f}")
    print(f"max|psi_numeric - psi_analytic| for g=0: {error:.2e}")

    # plot ground state density
    fig, ax = plt.subplots(figsize=(8, 6))
    density = np.array(psi_ground**2)
    ax.plot(np.array(x), density, color="b", lw=1.8, label=r"Numerical $|\Psi_0|^2$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\Psi|^2$")
    ax.set_title(f"Ground state (N={N_particles}, g={g})")
    fig.tight_layout()
    fig.savefig("gpe_1D_gs.png", dpi=300)

    # plot unnormalized norm convergence
    ax.clear()
    ax.plot(np.array(norm_history), color="b", lw=1.8, label=r"Norm $N_u$")
    ax.set_xlabel("imaginary-time step")
    ax.set_ylabel(r"$N_u$")
    ax.set_title(f"Convergence to ground state (N={N_particles}, g={g})")
    fig.tight_layout()
    fig.savefig("gpe_1D_convergence.png", dpi=300)


if __name__ == "__main__":
    main()
