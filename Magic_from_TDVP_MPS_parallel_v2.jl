using Distributed
############
n_workers = 8           # nprocs()  # Get the number of available workers
addprocs(n_workers)     # Add 4 worker processes (adjust as needed)
ENV["JULIA_NUM_THREADS"] = 2
############
@everywhere begin
    using LinearAlgebra
    using BenchmarkTools
    using StaticArrays
    using TensorOperations
    using StatsBase
    using NPZ
    using DelimitedFiles
    using Printf
    using JLD
    using CSV, DataFrames, Printf
    using Plots


    global n_r   = 2      ## Rényi index >1
    global n_sam = 160#10000  #### larger sampling is making the error increase !!!

    si = [1.0 0.0; 0.0 1.0]   # Identity matrix
    sx = [0.0 1.0; 1.0 0.0]  # Pauli X
    sy = [0.0 -1.0im; 1.0im 0.0]  # Real-valued version of Pauli Y
    sz = [1.0 0.0; 0.0 -1.0]  # Pauli Z
    ss = [si, sx, sy, sz]
    global sst = reshape([cat(ss[i], ss[j]; dims=3) for i in 1:4, j in 1:4],16)  # all possible 16 combination of the Pauli matrices !!! correct
    global sst_n = collect(1:16)
end
### checking the normalization of the MPS #### 
@everywhere function normalization_mps(N,MPS)
    MPS_norm = Array{Float64,2}(undef, 0, 0) 
    for i_n in 1:N
        A_i = MPS[i_n]; Ac_i = conj(A_i)  ## A for the ith site 
        c1,s1,s2,c2 = size(A_i)
        AA1   = @tensor A[i_,ip_,k_, kp_] :=  A_i[i_, i1_, i2_, k_] * Ac_i[ip_, i1_, i2_, kp_]
        if i_n == 1
            I_matrix= I(c1)
            MPS_norm = @tensor A[k_, kp_] :=  AA1[i_, ip_, k_, kp_] * I_matrix[i_, ip_]
        elseif i_n == N  
            MPS_norm = @tensor A[k_, kp_] :=  AA1[i_, ip_, k_, kp_] * MPS_norm[i_, ip_]
            I_matrix= I(c2) 
            MPS_norm = @tensor MPS_norm[k_,kp_]*I_matrix[k_,kp_]
        else   
            MPS_norm = @tensor A[k_, kp_] :=  AA1[i_, ip_, k_, kp_] * MPS_norm[i_, ip_]
        end    
        ### 
    end 
    return MPS_norm
end
#############################################
### estimator of magic via sampling #### 
@everywhere function Pi_SIGMA(N,MPS)
    ## an empty environment L has to be defined !!!!
    L_env   = Array{Float64,2}(undef, 0, 0) 
    L_env_c = Array{Float64,2}(undef, 0, 0) 
    Pi      = 1.0 
    for i in 1:N  # for a certain T loop over the sites
        A_i = MPS[i]; Ac_i = conj(A_i)   
        c1,s1,s2,c2 = size(A_i)
        if i == 1
            L_env = I(c1); L_env_c=conj(L_env)           
        end    
        p_list= zeros(Float64,length(sst_n))        
        for is in 1:length(sst_n)
            #######
            sigma = sst[is];sigma1=sigma[:,:,1];sigma2=sigma[:,:,2]
            AA1   = @tensor A[i_, i1_, i2_, j_] :=  sigma1[i1_, s1_] * sigma2[i2_, s2_] * A_i[i_, s1_, s2_, j_]      
            AAA1  = @tensor AA[j_,jp_] :=  L_env[i_,ip_]*Ac_i[i_,i1_,i2_,j_] *AA1[ip_,i1_,i2_,jp_] # ;AAA2=conj(AAA1)
            #######
            sigma1=conj(sigma1);sigma2=conj(sigma2)
            AA1   = @tensor A[i_, i1_, i2_, j_] :=  sigma1[i1_, s1_] * sigma2[i2_, s2_] * Ac_i[i_, s1_, s2_, j_]      
            AAA2  = @tensor AA[j_,jp_] :=  A_i[i_,i1_,i2_,j_] *AA1[ip_,i1_,i2_,jp_]*L_env_c[i_,ip_]
            #######
            AAA   = @tensor AAA1[j_,jp_]*AAA2[j_,jp_]
            p_list[is] = real(AAA)/4.0   
            #######
            if (p_list[is] < 0.0  && abs(log(abs(p_list[is]))) < 5.0)
                println("p_list[is] = ", p_list[is], " which sigma: ", is, " i = ", i, " n_sam = ", i_sam)
                println(log(abs(p_list[is])))
                return 0.0, 1.0
            end
            #######
        end  
        ### pick a random sst from the 16 choices  given the prob list 'p_list'
        chosen_sst  = sample(sst_n, Weights(p_list))
        pi_con = p_list[chosen_sst]  ### conditional probability at ith site.
        Pi     = Pi*pi_con
        ### update L_env = AAAA1 / sqrt() 
        sigma   = sst[chosen_sst];sigma1=sigma[:,:,1];sigma2=sigma[:,:,2]
        AA1     = @tensor A[i_, i1_, i2_, j_] :=  sigma1[i1_, s1_] * sigma2[i2_, s2_] * A_i[i_, s1_, s2_, j_]      
        AAA1    = @tensor AA[i_,ip_,j_,jp_] :=  Ac_i[i_,i1_,i2_,j_] *AA1[ip_,i1_,i2_,jp_] ;AAA2=conj(AAA1)
        AAAA1   = @tensor AAA[j_,jp_] := L_env[ip_,i_]*AAA1[i_,ip_,j_,jp_]
        L_env   = AAAA1/sqrt(4.0 * pi_con); L_env_c = conj(L_env)
        ######
    end # end of loop over the sites
    return Pi^(n_r-1)
end    
#############################################

# Function to compute the sums for each chunk
@everywhere function Pi_SIGMA_sum(N,MPS,n_vals_frac)

    sum_1 = 0.0
    sum_2 = 0.0
    for i_sam in n_vals_frac
        Pi2 = Pi_SIGMA(N,MPS)
        sum_1 += Pi2
        sum_2 += Pi2^2 
    end
    return [sum_1, sum_2]
end
# Function to divide the data among processes
function split_into_chunks(n_vals, n_chunks)
    chunk_size = cld(length(n_vals), n_chunks)  # Ceiling division
    return [n_vals[i:min(i+chunk_size-1, end)] for i in 1:chunk_size:length(n_vals)]
end

function parallel_computation(N,MPS,n_vals, n_workers)
    chunks = split_into_chunks(n_vals, n_workers)  # Split data into chunks  
    # Parallel execution
    t_start_total = time_ns()
    results = pmap(chunk -> Pi_SIGMA_sum(N,MPS,chunk), chunks) ### assigning each chunk to a worker
    t_end_total = time_ns()
    elapsed_time_total = (t_end_total - t_start_total) / 1e9
    # results = pmap((chunk->begin
    #             Pi_SIGMA_sum(N, MPS, chunk)
    #         end), chunks)

    # Sum up the results from all workers
    total_sum_1 = sum(r[1] for r in results)
    total_sum_2 = sum(r[2] for r in results)

    # Compute averages
    average_Pi2  = total_sum_1 / length(n_vals)
    average_dPi2 = total_sum_2 / length(n_vals)
    return average_Pi2 , average_dPi2, elapsed_time_total
end
# main code
#############################################
@everywhere begin
    dt      = 0.001
    h       = 0.0 ## field
    a       = 1.8
    chi_max = 256
    N_list  = [200]
    is      = 1000
    b_i     = 1
    beta    = (is-10.0+b_i)*dt
##############################################
## ------- picking the MPS ------- ##
    N   = N_list[1] 
    out = zeros(Float64,1000,2)
    A   = load( @sprintf("/Users/meghadeepa/Documents/Finite_T_entanglemnt/julia copy/h=0/TS_ctf=E8_dt=%1.3f_N=%d_chi_max=%d_a=%1.2f_h=%1.2f_%d.jld", dt, N, chi_max, a, h, is ))
    B   = A["data"]#[1:10]
    MPS = B[b_i,:]   ### for a vertain temperature
end 

## ------- checking normalization of the MPS ------- ##
# mps_norm = normalization_mps(N,MPS)
# println("MPS_norm: ", mps_norm)

##------------ parallelizing over the sampling ---------##

n_vals = collect(1:n_sam) 
println(length(n_vals))
sam_res, sam_res2, elapsed_time_total = parallel_computation(N,MPS,n_vals, n_workers)

magic   = ((log(sam_res) /(1.0 - n_r)) - N*log(4.0))/N
dm      = sqrt(abs(sam_res2 - sam_res^2))/sam_res  
println("system size: ", N," sample size: ", n_sam)
println("beta: ", beta ," magic: ", magic)
println("propagated stat err: ",dm," elapsed time = ", elapsed_time_total, " seconds")
#----------------------------------------------------##