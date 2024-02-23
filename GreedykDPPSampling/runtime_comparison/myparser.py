import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument("--discretization", type=int, required=True)
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--nb_samples", type=int, required=True)
parser.add_argument("--ell", type=float, required=True)
parser.add_argument("--repetition", type=int, required=True)
