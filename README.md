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

## Contributing
Contributions to EMCi are welcome. Please ensure that your code adheres to the project's style and standards.

## License
This project is licensed under GPL-3.0 license.