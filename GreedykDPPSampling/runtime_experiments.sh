for i in {1..5}
do

gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=100 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=1000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=10000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=50000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=60000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=70000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=100000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i

done

for i in {1..5}
do
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=100 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=1000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=10000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=50000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=60000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=70000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=100000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
done

for i in {1..5}
do
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=100 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=1000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=10000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=50000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=60000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=70000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=100000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
done

for i in {1..5}
do
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=100 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=1000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=10000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=50000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=60000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=70000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
gtimeout 100s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=100000 --nb_samples=100 --k=10 --ell=0.01 --repetition=$i
done


for i in {1..3}
do
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=1000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=10000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=50000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=60000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=70000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_greedy --discretization=100000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i

done

for i in {1..3}
do
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=1000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=10000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=50000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=60000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=70000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_mcmc_k_grid --discretization=100000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
done

for i in {1..3}
do
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=1000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=10000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=50000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=60000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=70000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_vfx_k_grid --discretization=100000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i

done

for i in {1..3}
do
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=1000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=10000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=50000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=60000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=70000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i
gtimeout 360s python3 -m runtime_comparison.runtime_comparison_exact_alpha_k_grid --discretization=100000 --nb_samples=100 --k=100 --ell=0.001 --repetition=$i

done
