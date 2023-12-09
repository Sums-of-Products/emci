# EMCi: Structure MCMC Discovery Tool

## Description
EMCi is a Python tool for performing Structural MCMC (Markov Chain Monte Carlo) discovery on score files. It leverages graph manipulation and sampling techniques to analyze and visualize score data, providing insights into the underlying structure.

## Installation
To install EMCi, you will need Python installed on your system along with a few dependencies. You can install all required dependencies by running:

`pip install -r requirements.txt`

## Dependencies
- numpy
- python-igraph
- scipy
- matplotlib

## Usage
To use EMCi, run the `main.py` script with a score file name and the number of steps. For example:
`python3 main.py insurance-1000 5000`
This command will process the score file `insurance-1000` with a parameter value of `5000`.

## ScoreManager Class

### Overview
The `ScoreManager` class is a central component of the project, designed for handling and processing score files associated with graph structures. It leverages `igraph` for graph operations and `numpy` for numerical computations.

### Key Features
- **Initialization**: Instantiates with a score file name, reading and storing scores from `data/scores/{score_name}.jkl`.
- **Graph Scoring**: 
  - `get_score(G)`: Computes the total score of a graph `G`, summing up local scores of each vertex based on its predecessors.
  - `get_local_score(v, pa_i, n)`: Calculates the local score for a vertex `v` given a set of parent vertices `pa_i`, incorporating the Koivisto prior.
  - `P(M)`: Calculates a probability measure for a graph `M`.

### Score File Structure
Score files (.jkl) should follow this format:
- The first line indicates the number of vertices.
- Subsequent lines contain scores for each vertex, formatted as `score k parent1 parent2 ... parentK`, where `k` is the number of parents.

Example Line: 
-45.16814748008983 2 1 18
This represents a vertex with a score of `-45.16814748008983`, having two parents (`k=2`), specifically vertices `1` and `18`.

### Additional Utility
- `R(current_score, proposed_score)`: This function computes the acceptance probability for a proposed score in the MCMC algorithm, handling potential overflow issues.


## Contributing
Contributions to EMCi are welcome. Please ensure that your code adheres to the project's style and standards.

## License
This project is licensed under GPL-3.0 license.