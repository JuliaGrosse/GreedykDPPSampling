# GreedykDPPSampling

This folder contains the code needed to reproduce the results and Figures from the paper: "A Greedy Approximation for k-Determinantal Point Processes".

### Instructions for how to reproduce the results and plots in the paper:

1. Install the requirements in the "requirements.txt" file

```
cd GreedykDPPSampling
pip install -r ./requirements.txt
```

2. Run the bash scripts to generate all necessary results for the experiments

```
cd GreedykDPPSampling
bash integration_experiments.sh
bash runtime_experiments.sh
bash runtime_experiments_higher_dim.sh
```

Alternatively, have a look at "runtime_experiments.sh" and "intergation_experiments.sh" to see how to call the python scripts for the several baselines and run them in whatever way you like.

4. To re-create a certain Figure 1-8, just run the corresponding python script named after that Figure from the root folder.

```
cd GreedykDPPSampling
python -m FigureX
```
Figures generated in this way end up as pdfs in the "Figures" folder.
