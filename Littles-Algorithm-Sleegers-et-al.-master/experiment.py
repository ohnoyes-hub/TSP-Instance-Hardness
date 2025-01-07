import glob
import json
import numpy as np

from LittlesAlgorithm.algorithm import get_minimal_route

def replace_dash_with_infity(unparsed_matrix):
  """
  Replaces all elements that are - to numpy.infinity.

  Parameters
  ----------
  unparsed_matrix: List[List[String]]
    representation of a cost matrix

  Returns
  -------
  List[List[String | np.inf]]:
    representation of cost matrix
  """
  return [
    [np.inf if x == '-' else x for x in row]
    for row in unparsed_matrix
  ]

def load_file(filename):
  """
  Load file and parse the files to the float representation of the cost matrix
  and the integer representation of the cost matrix.

  Parameters
  ----------
  filename: String
    filename of the file that contains tsp instance

  Returns
  -------
  np.array:
    float representation of cost matrix
  np.array:
    integer representation of cost matrix
  """
  with open(filename) as f:
    file_as_json = json.load(f)
    unparsed_float_matrix, unparsed_rounded_matrix = file_as_json['float'], file_as_json['int']
    parsed_float_matrix = np.array(replace_dash_with_infity(unparsed_float_matrix))
    parsed_rounded_matrix = np.array(replace_dash_with_infity(unparsed_rounded_matrix))
    return parsed_float_matrix, parsed_rounded_matrix

amount_of_cities = 48
result_file = 'results/results{0}.csv'.format(amount_of_cities)
files_in_csv = {}

try:
  with open(result_file, 'r') as f:
    files_in_csv = {
      line.split(',')[0] + line.split(',')[1] for line in f
    }
except FileNotFoundError:
  with open(result_file, 'w') as f:
    f.write('file,floatingpoint/integer,iterations,optimal tour,optimal tour length\n')
  pass

def handle_matrix(float_or_integer, filename, matrix):
  """
  Solves tsp instance (matrix) and adds the result as a line to the file
  in filename.

  Parameters
  ----------
  float_or_integer: String
    string that respresents if the matrix contains float or integer numbers
  filename: String
    filename of the file that contains the result data
  matrix: np.array
    cost matrix of a tsp instance
  """

  print('solving', float_or_integer, filename)

  iterations, optimale_route, optimal_cost = get_minimal_route(matrix)
  toWrite = '{0},{1},{2},{3},{4}\n'.format(
    filename.replace('.json', ''),
    float_or_integer,
    iterations,
    str(optimale_route).replace(',', ''),
    optimal_cost
  )

  with open(result_file, 'a') as f:
    f.write(toWrite)


for filename in glob.iglob('data/{}cities/**/*.json'.format(amount_of_cities), recursive=True):
  stripped_filename = filename.split('/')[-1]
  parsed_float_matrix, parsed_rounded_matrix = load_file(filename)
  if not stripped_filename.replace('.json', 'integer') in files_in_csv:
    handle_matrix('integer', stripped_filename, parsed_rounded_matrix)

  if not stripped_filename.replace('.json', 'floatingpoint') in files_in_csv:
    handle_matrix('floatingpoint', stripped_filename, parsed_float_matrix)
