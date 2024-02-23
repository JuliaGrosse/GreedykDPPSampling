for i in {1..5}
do
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=20 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=25 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=30 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=35 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=40 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i

done

for i in {1..5}
do
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=20 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=25 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=30 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=35 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=40 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
done


for i in {1..3}
do
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=40 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=50 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=60 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=70 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=80 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_greedy --discretization=90 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i

done

for i in {1..3}
do
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=40 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=50 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=60 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=70 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=80 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_mcmc_k_grid --discretization=90 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
done

for i in {1..5}
do
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=20 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=25 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=30 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=35 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=40 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
done

for i in {1..5}
do
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=20 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=25 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=30 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=35 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 500s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=40 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
done

for i in {1..3}
do
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=40 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=50 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=60 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=70 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=80 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_alpha_k_grid --discretization=90 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
done

for i in {1..3}
do
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=40 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=50 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=60 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=70 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=80 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 1000s python3 -m runtime_comparison_higher_dim.runtime_comparison_exact_vfx_k_grid --discretization=90 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
done
