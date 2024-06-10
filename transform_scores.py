import numpy as np
from scipy.special import comb


def adjust_scores(file_path, output_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    num_vars = int(lines[0].strip())
    output_lines = [f"{num_vars}\n"]
    index = 1

    while index < len(lines):
        variable_info = lines[index].strip()
        output_lines.append(f"{variable_info}\n")
        num_parent_sets = int(variable_info.split()[1])
        index += 1

        for _ in range(num_parent_sets):
            parent_set_info = lines[index].strip().split()
            log_weight = float(parent_set_info[0])
            num_parents = int(parent_set_info[1])

            # Compute the binomial coefficient and subtract its log from the weight
            binom_coeff = comb(num_vars - 1, num_parents)
            adjusted_log_weight = log_weight - np.log(binom_coeff)

            # Construct the new line with adjusted weight
            new_line = f"{adjusted_log_weight} " + " ".join(parent_set_info[1:]) + "\n"
            output_lines.append(new_line)
            index += 1

    # Write the output to a new file
    with open(output_path, "w") as output_file:
        output_file.writelines(output_lines)


score_name = "zoo"

# Example usage
input_file_path = f"data/scores/{score_name}.jkl"
output_file_path = f"data/scores/{score_name}-sparse.jkl"
adjust_scores(input_file_path, output_file_path)
