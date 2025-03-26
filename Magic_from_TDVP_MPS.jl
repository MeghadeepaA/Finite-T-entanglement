using LinearAlgebra
using BenchmarkTools
using StaticArrays
using TensorOperations
using StatsBase
#using QuantumOptics
using NPZ
using DelimitedFiles
using Printf
using JLD
using CSV, DataFrames, Printf
using Plots

############
global n_r   = 2      ## Rényi index >1
global n_sam = 2000  #### larger sampling is making the error increase !!!

si = [1.0 0.0; 0.0 1.0]   # Identity matrix
sx = [0.0 1.0; 1.0 0.0]  # Pauli X
sy = [0.0 -1.0im; 1.0im 0.0]  # Real-valued version of Pauli Y
sz = [1.0 0.0; 0.0 -1.0]  # Pauli Z

ss = [si, sx, sy, sz]
global sst = reshape([cat(ss[i], ss[j]; dims=3) for i in 1:4, j in 1:4],16)  # all possible 16 combination of the Pauli matrices !!! correct
global sst_n = collect(1:16)


### checking the normalization of the MPS #### 
function normalization_mps(N,MPS)
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
function sampling(N,MPS)
    sam_res = 0.0; sam_res2 = 0.0
    for i_sam in 1:n_sam ### averaging over sampling 
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
                sigma = sst[is];sigma1=sigma[1,:,:];sigma2=sigma[2,:,:]
                AA1   = @tensor A[i_, i1_, i2_, j_] :=  sigma1[i1_, s1_] * sigma2[i2_, s2_] * A_i[i_, s1_, s2_, j_]      
                AAA1  = @tensor AA[i_,ip_,j_,jp_] :=  Ac_i[i_,i1_,i2_,j_] *AA1[ip_,i1_,i2_,jp_] ;AAA2=conj(AAA1)
                AAA   = @tensor L_env[i_,ip_]*AAA1[i_,ip_,j_,jp_]*AAA2[ic_,icp_,j_,jp_]*L_env_c[icp_,ic_]
                p_list[is] = real(AAA)/4.0         
            end  

            ### pick a random sst from the 16 choices  given the prob list 'p_list'
            chosen_sst  = sample(sst_n, Weights(p_list))
            pi_con = p_list[chosen_sst]  ### conditional probability at ith site.
            Pi     = Pi*pi_con

            ### update L_env = AAAA1 / sqrt() 
            sigma   = sst[chosen_sst];sigma1=sigma[1,:,:];sigma2=sigma[2,:,:]
            AA1     = @tensor A[i_, i1_, i2_, j_] :=  sigma1[i1_, s1_] * sigma2[i2_, s2_] * A_i[i_, s1_, s2_, j_]      
            AAA1    = @tensor AA[i_,ip_,j_,jp_] :=  Ac_i[i_,i1_,i2_,j_] *AA1[ip_,i1_,i2_,jp_] ;AAA2=conj(AAA1)
            AAAA1   = @tensor AAA[j_,jp_] := L_env[ip_,i_]*AAA1[i_,ip_,j_,jp_]
            L_env   = AAAA1/sqrt(4.0 * pi_con); L_env_c = conj(L_env)
            ######
        end # end of loop over the sites
        if i_sam % 100 == 0
            println("P_con ", Pi,"  n is =",i_sam)
        end
        sam_res += Pi^(n_r-1);sam_res2 += Pi^2*(n_r-1)
    end  # end of loop over the sampling
    sam_res = sam_res/n_sam; sam_res2 = sam_res2/n_sam #!!!!!
    magic   = (log(sam_res) /(1.0 - n_r)) - N*log(4.0)
    var_  = abs(sam_res2 - sam_res^2)/sam_res
    return magic,sam_res2
end    
#############################################
# main code
#############################################
dt      = 0.001
h       = 0.0 ## field
a       = 1.8
chi_max = 256
N_list  = [200]
is      = 1000
b_i     = 1
beta    = (is-10.0+b_i)*dt
##############################################

for N in N_list
    out = zeros(Float64,1000,2)
    A   = load( @sprintf("TS_ctf=E8_dt=%1.3f_N=%d_chi_max=%d_a=%1.2f_h=%1.2f_%d.jld", dt, N, chi_max, a, h, is ))
    B   = A["data"]#[1:10]
    MPS = B[b_i,:]   ### for a vertain temperature

    ##############################################################
    ## ------- normalization of the MPS ------- ##
    mps_norm = normalization_mps(N,MPS)
    println("MPS_norm: ", mps_norm)
     ###--------- here will start the sampling loop ---------##
    magic1,dm = sampling(N,MPS)
    println("system size: ", N," sample size: ", n_sam)
    println("beta: ", beta ," magic: ", magic1/N)
    println("propagated stat err: ", dm)
end 
#############################################   
   