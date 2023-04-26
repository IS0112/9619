#PS05 Q2 Used Sergio's code and internet

using Parameters
using Interpolations
using Distributions
using LinearAlgebra
using Statistics

# Set parameters
@with_kw struct Par
    α::Float64 = 1/3;
    β::Float64 = 0.98;
    δ::Float64=0.05;
    η::Float64=1.0;
    sigma::Float64=2.0;
    xpar::Float64=0.0817;
    max_iter::Int64   = 800  ; 
    dist_tol::Float64 = 1E-9  ; # Tolerance for distance
    N::Int64          = 28  # grid points
    ρ::Float64 = 0.9 ; # Persistence of productivity process
    σ::Float64=0.1; 
end

# Put parameters into object par
par = Par()

# Define the utility function
function u(c,l, par::Par)
    @unpack σ, xpar, η = par 
    if l < 0 || l > 1
        return NaN
    end
    return c^(1 - sigma) / (1 - sigma) - xpar*l^(1+η) / (1 + η)
end

# Define a markov process struct
    # Generate structure for markov processes 
    @with_kw struct MP
        # Model Parameters
        N::Int64 # Number of states
        grid     # Grid of discrete markov process
        Π        # Transition matrix
        PDF      # Stationary distribution
        CDF      # Stationary distribution
    end
#Q2a)
function Rouwenhorst95(ρ,σ,N)
    # Define parameters for Rouwenhorst's approximation
        p = (1+ρ)/2
        q = p                   # Note: I am leaving q here for comparability with source
            ψ = sqrt((N-1)/(1-ρ^2)) * sqrt(log(1+σ^2))
        s = (1-q)/(2-(p+q))     # Note: s=0.5, I leave it for comparability with source
    # Fill in transition matrix
    if N==2
        Π_z = [p 1-p ; 1-q q]
    else
        MP_aux = Rouwenhorst95(ρ,σ,N-1)
        o = zeros(N-1)
        Π_z = p*[MP_aux.Π o ; o' 0] + (1-p)*[o MP_aux.Π ; 0 o'] + (1-q)*[o' 0 ; MP_aux.Π o] + q*[0 o' ; o MP_aux.Π]
        # Adjust scale for double counting
        Π_z = Π_z./repeat(sum(Π_z,dims=2),1,N)
    end
    # Distribution
    PDF_z = pdf.(Binomial(N-1,1-s),(0:N-1))
    CDF_z = cumsum(PDF_z)
    
    # Create z grid
    log_z_grid = range(-ψ, ψ, length = N)
    z_grid = exp.(log_z_grid)
        return MP(N=N,grid=z_grid,Π=Π_z,PDF=PDF_z,CDF=CDF_z)
end

# Define the grid of capital stock
k_grid = range(0.01, stop=300, length=1000)

# Define Bellman operator with Markov Process
function T_MP(v,MP::MP,par::Par)
    # Check if the same dimension as the Markov Process
    if length(v) != MP.N * length(k_grid)
        error("Error")
    end
    v = reshape(v, MP.N, length(k_grid))
    # Unpack parameters
         @unpack β, α, δ, xpar,sigma,η = par 
    # Consumption and labor initialize
    c = zeros(MP.N, MP.N, length(k_grid))
    l = zeros(MP.N, MP.N, length(k_grid))
    for i in 1:MP.N
        for j in 1:MP.N
            for k_idx in 1:length(k_grid)
                k = k_grid[k_idx]
                c[i,j,k_idx] = k^α * (exp(MP.grid[i]) * l[i,j,k_idx]^(1-α)) + (1 - δ) * k - k_grid[j]
                l[i,j,k_idx] = ((1 - α) * k * c[i,j,k_idx]^(-sigma)*exp(MP.grid[i])/xpar)^((1 / (α+η)))
                end
        end
    end
    # Initialize value function output
        Tv = zeros(MP.N, length(k_grid))
    # Loop through each state in the grid
    for i in 1:MP.N
        for k_idx in 1:length(k_grid)
            EV = 0
            # Calculate expected value
            for j in 1:MP.N
                EV += MP.Π[i, j] * interpolate(v[j, :], NoInterp(), k_grid[k_idx])
            end
            # Update value function
            Tv[i, k_idx] = maximum(u.(c[i, :, k_idx], l[i, :, k_idx], Ref(par)) .+ β * EV)
    end
        end
        return Tv
end

# Value Function Iteration
function iteration(T::Function,par::Par)
    
    @unpack max_iter, dist_tol, N = par
    # Initialize variables for loop
    v_old  = zeros(N)     ; # Initialize value function
    V_dist = 1              ; # Initialize distance
        for iter=1:max_iter
        # Update value function
        v_new = T(v_old)
                    # Update distance and iterations
        V_dist = maximum(abs.(v_new./v_old.-1))
        # Update old function
        v_old  = v_new
        # Report progress
        if mod(iter,100)==0
            println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
        end
        # Check convergence and return results
        if V_dist<=dist_tol
            println("VFI - N=$N ")
            println("Iterations = $iter and Distance = ",100*V_dist,"%")
            println("------------------------")
            println(" ")
            # Return results
            return v_old
        end
    end
    # If loop ends there was no convergence -> Error!
    error("Error in VFI")
end

log_z_grid = range(-4*par.σ,4*par.σ,length=par.N)
  MP_R = Rouwenhorst95(par.ρ,par.σ,par.N) 
  @time v_R = iteration(x -> T_MP(x, MP_R, par), par)
    
    # Plot results
    #using Plots
    #gr()
    # Reshape v to match the dimensions of the Markov Process
#v = reshape(v, MP.N, length(k_grid))

# 3D Plot of the value function
#plot(MP.grid, k_grid, v, st=:surface,title="Value Function")

# Level curves of the value function
#contour(MP.grid, k_grid, v, levels = 15, title="Value Function Level Curves")

#Q2c

using Distributions
using Random
# Set seed
Random.seed!(3486);

# Set number of simulations
T = 100

# Create AR(1) process
z_dist = Normal(0, par.σ)
z = zeros(T+1)
for t in 1:T
    z[t+1] = par.ρ*z[t] + rand(z_dist)
end

# Convert log z to z
z = exp.(z)

# Set initial values
k = .5
l = 0.2

# Initialize arrays to store results
k_si = zeros(T+1)
l_si = zeros(T+1)
y_si = zeros(T+1)
c_si = zeros(T+1)

# Simulate variables over time
for t in 1:T+1
    # Calculate output and consumption
    y = k^par.α * (z[t] * l^(1-par.α))
    c = y + (1 - par.δ)*k - k_si[t]
    # Update capital and labor
    k_si[t+1] = c + par.δ*k
    l_si[t+1] = ((1-par.α)*k_si[t]*c^(-par.σ)*z[t]/par.xpar)^((1/(par.α+par.η)))

    y_si[t] = y
    c_si[t] = c
end

using Plots
time= 0:T
plot(time, c_si, title="Levels")
plot!(time, y_si, label="Output")
plot!(time, k_si[1:T+1], label="Capital")
plot!(time, l_si[1:T+1], label="Labor")

# Report second moments and their first differences changes
println("Second moments:")
println("Consumption:", mean(c_si), var(c_si))
println("Output:", mean(y_si), var(y_si))
println("Capital:", mean(k_si), var(k_si))
println("Labor:", mean(l_si), var(l_si))

println("First difference changes:")
println("Consumption:", mean(diff(c_si)), var(diff(c_si)))
println("Output:", mean(diff(y_si)), var(diff(y_si)))
println("Capital:", mean(diff(k_si)), var(diff(k_si)))
println("Labor:", mean(diff(l_si)), var(diff(l_si)))