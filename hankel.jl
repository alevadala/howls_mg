using SpecialFunctions

function hankel_transform(P, θ, ℓ_max)
    # Define the integration range
    ℓ_values = collect(0:ℓ_max)

    # Perform the Hankel transform
    integral_result = sum(2 * π * P(ℓ) * besselj(0, ℓ * θ) * ℓ for ℓ in ℓ_values)

    return integral_result
end

# Example function P(ℓ)
P(ℓ) = ℓ^2  # Replace this with your actual function

# Parameters
θ = 0.1  # Adjust the value of θ as needed
ℓ_max = 100

# Perform the Hankel transform
result = hankel_transform(P, θ, ℓ_max)

# Display the result
println("Hankel Transform Result: ", result)
