using QuadGK
using SpecialFunctions
using DelimitedFiles
using Trapz
using Plots
using Interpolations

# Define cosmological parameters for DUSTGRAIN-pathfinder simulations
# as they appear in https://doi:10.1093/mnras/sty2465 for LCDM background
Ω_b = 0.0481
h = 0.6731
H0 = h*100 # Hubble constant in km/s/Mpc

# For fr4
Ω_m = 0.31345
Ω_cdm = Ω_m - Ω_b
Ω_de = 1 - Ω_m

plot_label = "fR4"

# Load matter power spectrum from file (assuming it contains only P_kappa(ell) values)
Pnl = readdlm("fr4.txt")
k = readdlm("fr4_k.txt")
z = readdlm("fr4_z.txt")

zs = 0.5

# Function to calculate E(z)
function E(z)
    return sqrt(Ω_m * (1 + z)^3 + Ω_de)
end

# Function to calculate comoving distance r(z)
function r(z)
    c = 299792.458 # Speed of light in km/s
    integrand(z_prime) = 1 / E(z_prime)
    result, _ = quadgk(integrand, 0.0, z)
    return (c / H0) * result
end

zk_values = (z,k)

Pzk = CubicSplineInterpolation(zk_values, Pnl)

# Angular power spectrum without tomography
function C_ℓ(ℓ)
    c = 299792.458 # Speed of light in km/s

    # Function to calculate W(z)
    function W(z)
        return 1.5 * Ω_m * (H0 / c)^2 * (1 + z) * r(z) * (1 - (r(z) / r(zs)))
    end

    # Integrating function
    integrand(z) = W(z)^2 * P_zk[z, (ℓ + 0.5) / r(z)]/ r(z)^2 / E(z)
    
    # Perform the integral using quadgk
    result, _ = quadgk(integrand, 0.0, zs)
    
    # Return the result
    return (c / H0) * result
end

# Define the range of ell values
ℓ = range(10, stop=10000, length=100)

# Define the range of theta values
θ = exp10.(range(start=0.1, stop=100, length=100))

# Define the integrand function for xi_plus(theta)
function integrand(θ, ℓ)
    return (1 / (2π)) * C_ℓ(ℓ)* besselj0(θ * ℓ) * ℓ
end


# Compute xi_plus(theta) for each theta value
ξ = zeros(length(θ))
for (i, val_θ) in enumerate(θ)
    integrand_ℓ(ℓ) = integrand(val_θ, ℓ)
    ξ[i], _ = quadgk(integrand_ℓ, ℓ)
end

# Plot xi_plus(theta)
plot(θ, ξ, xaxis=:log, yaxis=:log, xlabel="θ", ylabel="ξ_+(θ)", title="Two-point Correlation Function")
