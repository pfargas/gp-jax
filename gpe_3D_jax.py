#  3D Gross-Pitaevskii equation solver
#  for a harmonic oscillator using JAX
#
#  Equation:
#      i d(psi)/dt = [ -1/2 ∇^2 + V(x,y,z) + g|psi|^2 ] psi
#      (in natutal units)
#
#  V(x,y,z) = 1/2 ( x^2 + λxy^2 y^2 + λxz^2 z^2 )
#
#  Split-step Fourier method (with Strang splitting)

"""
IN DEVELOPMENT
"""

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
def make_grid_3d(
    Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float
) -> tuple[
    tuple[Array, Array, Array], tuple[Array, Array, Array], tuple[float, float, float]
]:
    """
    Uniform grid on [-L/2, L/2) in each direction
    and the matching FFT angular wavenumbers
    """
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    x = -Lx / 2 + dx * jnp.arange(Nx)
    y = -Ly / 2 + dy * jnp.arange(Ny)
    z = -Lz / 2 + dz * jnp.arange(Nz)

    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dz)

    return (x, y, z), (kx, ky, kz), (dx, dy, dz)


# k² Grid
def k_squared_grids(kx: Array, ky: Array, kz: Array, Nz: int) -> tuple[Array, Array]:
    """
    Builds k^2 = kx^2 + ky^2 + kz^2 on the two spectra needed
      K2_full: full FFT spectrum, shape (Nx, Ny, Nz)          -> energy (fftn)
      K2_rfft: real-FFT spectrum, shape (Nx, Ny, Nz / /2 + 1) -> evolution (rfftn)

    Only the last axis is compressed by rfftn/irfftn,
    so kx and ky stay full-length and only kz is truncated
    """
    KX = kx[:, None, None]
    KY = ky[None, :, None]

    K2_full = KX**2 + KY**2 + kz[None, None, :] ** 2

    kz_rfft = kz[: Nz // 2 + 1]  # matches jnp.fft.rfftn's compressed last axis
    K2_rfft = KX**2 + KY**2 + kz_rfft[None, None, :] ** 2

    return K2_full, K2_rfft


# Harmonic Potential
def harmonic_potential(x: Array, y: Array, z: Array, w_xy: float, w_xz: float) -> Array:
    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]
    return 0.5 * (X**2 + w_xy**2 * Y**2 + w_xz**2 * Z**2)


# Wavefunction norm
def norm_wf(psi: Array, dV: float) -> Array:
    return jnp.sum(psi**2) * dV


# Energy functional
@jax.jit
def energy_func(
    psi: Array, K2_full: Array, V: Array, g: float, dV: float, dV_N: float
) -> Array:
    """
    E = integral[ 1/2|d(psi)/dx|^2 + V|psi|^2 + g/2|psi|^4 ] dx
    """
    psi_sq = psi**2

    # using Parseval's theorem
    psi_k = jnp.fft.fftn(psi)
    psi_k_sq = psi_k.real**2 + psi_k.imag**2
    e_kin = 0.5 * jnp.sum(K2_full * psi_k_sq) * dV_N

    e_pot = jnp.sum(V * psi_sq) * dV
    e_int = 0.5 * g * jnp.sum(psi_sq**2) * dV
    return e_kin + e_pot + e_int


# Imaginary time evolution
@partial(jax.jit, static_argnames=("N_steps",))
def imaginary_time_evolution(
    psi0: Array,
    K2_rfft: Array,
    V: Array,
    g: float,
    dV: float,
    N_particles: int,
    dtau: float,
    N_steps: int,
) -> tuple[Array, Array]:

    minus_half_dtau = -0.5 * dtau

    # only positive wavenumbers for Real FFT
    kinetic_factor = jnp.exp(minus_half_dtau * K2_rfft)

    # precompute static exponential factors once
    minus_half_dtau_V = minus_half_dtau * V
    minus_half_dtau_g = minus_half_dtau * g

    # using fast inverse square root
    psi0 = psi0 * jax.lax.rsqrt(norm_wf(psi0, dV) / N_particles)

    grid_shape = psi0.shape

    def step(psi: Array, _) -> tuple[Array, Array]:
        # first half-step potential
        psi = psi * jnp.exp(minus_half_dtau_V + minus_half_dtau_g * psi**2)

        # full-step kinetic (using Real FFT)
        psi = jnp.fft.irfftn(jnp.fft.rfftn(psi) * kinetic_factor, s=grid_shape)

        # second half-step potential
        rho = psi**2
        S = jnp.exp(minus_half_dtau_V + minus_half_dtau_g * rho)
        psi = psi * S

        # norm calculation from rho and S
        current_norm = jnp.sum(rho * S**2) * dV

        psi = psi * jax.lax.rsqrt(current_norm / N_particles)

        return psi, current_norm

    psi_final, norm_history = lax.scan(step, psi0, None, length=N_steps)

    return psi_final, norm_history


# Main function
def main() -> None:
    Nx = 64  # grid points (2^n for FFT)
    Ny = 64
    Nz = 64
    w_xy = 3.0
    w_xz = 5.0
    Lx = 20.0  # box length
    Ly = 20.0
    Lz = 20.0
    g = 5.0  # interaction strength (>0 repulsive, <0 attractive)
    N_particles = 10

    (x, y, z), (kx, ky, kz), (dx, dy, dz) = make_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

    dV = dx * dy * dz
    dV_N = dV / (Nx * Ny * Nz)

    K2_full, K2_rfft = k_squared_grids(kx, ky, kz, Nz)

    V = harmonic_potential(x, y, z, w_xy, w_xz)

    dtau = 5e-4
    N_steps = 20_000

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    psi_trial = jnp.exp(-0.5 * (X**2 + w_xy * Y**2 + w_xz * Z**2))

    psi_ground, norm_history = imaginary_time_evolution(
        psi_trial, K2_rfft, V, g, dV, N_particles, dtau, N_steps
    )
    E_final = energy_func(psi_ground, K2_full, V, g, dV, dV_N)

    print(f"[ground state]")
    print(f"energy = {float(E_final):.6f}")
    print(f"norm = {float(norm_wf(psi_ground, dV)):.6f}")

    # validation: g=0 case E=N/2
    psi_g0, _ = imaginary_time_evolution(
        psi_trial, K2_rfft, V, 0.0, dV, N_particles, dtau, N_steps
    )
    E_final_g0 = energy_func(psi_g0, K2_full, V, 0.0, dV, dV_N)
    analytic_gaussian = (
        jnp.sqrt(N_particles)
        * (w_xy * w_xz) ** 0.25
        / jnp.pi**0.75
        * jnp.exp(-0.5 * (X**2 + w_xy * Y**2 + w_xz * Z**2))
    )
    error = float(jnp.max(jnp.abs(psi_g0 - analytic_gaussian)))
    E_analytic_g0 = N_particles * 0.5 * (1 + w_xy + w_xz)

    print(f"[validation]")
    print(f"energy = {float(E_final_g0):.6f}  (analytic = {E_analytic_g0:.6f})")
    print(f"max|psi_numeric - psi_analytic| for g=0: {error:.2e}")

    # # plot ground state density
    # fig, ax = plt.subplots(figsize=(8, 6))
    # density = np.array(psi_ground**2)
    # ax.plot(np.array(x), density, color="b", lw=1.8, label=r"Numerical $|\Psi_0|^2$")
    # ax.set_xlabel("x")
    # ax.set_ylabel(r"$|\Psi|^2$")
    # ax.set_title(f"Ground state (N={N_particles}, g={g})")
    # fig.tight_layout()
    # fig.savefig("gpe_3D_gs.png", dpi=300)

    # plot unnormalized norm convergence
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.array(norm_history), color="b", lw=1.8, label=r"Norm $N_u$")
    ax.set_xlabel("imaginary-time step")
    ax.set_ylabel(r"$N_u$")
    ax.set_title(f"Convergence to ground state (N={N_particles}, g={g})")
    fig.tight_layout()
    fig.savefig("gpe_3D_convergence.png", dpi=300)



    # --- 3D Plotting with Plotly ---
    import plotly.graph_objects as go

    density = np.array(psi_ground**2)
    
    X_dense, Y_dense, Z_dense = np.meshgrid(
        np.array(x), np.array(y), np.array(z), indexing="ij"
    )

    # Plotly requires flattened 1D arrays for volumetric data
    fig = go.Figure(data=go.Volume(
        x=X_dense.flatten(),
        y=Y_dense.flatten(),
        z=Z_dense.flatten(),
        value=density.flatten(),
        isomin=0.05 * np.max(density), # Hide outer empty space
        isomax=np.max(density),
        opacity=0.2,                   # Make layers transparent
        surface_count=20,              # Number of nested layers
        colorscale="magma"
    ))

    fig.update_layout(
        title=f"3D BEC Probability Density (N={N_particles}, g={g})",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.write_html("gpe_3D_density.html")


if __name__ == "__main__":
    main()
