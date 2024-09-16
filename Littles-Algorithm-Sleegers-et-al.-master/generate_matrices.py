import numpy as np
import json
import os
import uuid

# Adding time to test how long it takes to generate the matrices
import time

MEAN = 10
AMOUNT_OF_MATRICES_PER_STD = 50
AMOUNT_OF_CITIES = 4

from experiment import replace_dash_with_infity

def generate_matrix_log_norm(amount_of_cities, mean, std):
  """
  Generate matrix where the elements are drawn from a log normal distribution.

  Parameters
  ----------
  amount_of_cities: Int
    size of the resulting matrix
  mean: Float
    mean that the resulting matrix should be (not input to log normal)
  std:
    standard deviation to use for the log normal distribution

  Returns
  -------
  np.array:
    matrix generated according to log normal distribution
  """
  mu = np.log(mean) - 1/2 * std**2
  matrix = np.random.lognormal(mu, std, ((amount_of_cities, amount_of_cities)))
  for i in range(amount_of_cities):
    matrix[i, i] = np.inf

  return matrix

def save_matrix(matrix, amount_of_cities, mean, std, id):
  """
  Saves a matrix to the apropriate location in json format with the 
  corrseponding filename.

  Parameters
  ----------
  matrix: np.array
    cost matrix of the tsp instance
  amount_of_cities: Int
    amount of cities in the tsp instance
  mean: Float
    mean used to generate the tsp cost matrix
  std: Float
    standard deviation used to generate cost matrix
  id: Int
    id of the matrix
  """
  readable_std = str(std).replace('.', '')
  filename = '{0}cities_avg{1}_std{2}_{3}'.format(
    amount_of_cities, mean, readable_std, id
  )
  path = 'data/{0}cities/avg{1}/std{2}/{3}.json'.format(
    amount_of_cities, mean, readable_std, filename, 
  )
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump({ 
        'float': replace_dash_with_infity(matrix.tolist()),
        'int': replace_dash_with_infity(np.around(matrix).tolist())
    }, f)

if __name__ == "__main__":
  start = time.time()

  for i in np.arange(0, 5.1, 0.2):
    std = np.round(i, 1)
    for i in range(AMOUNT_OF_MATRICES_PER_STD):
      matrix = generate_matrix_log_norm(AMOUNT_OF_CITIES, MEAN, std)
      save_matrix(matrix, AMOUNT_OF_CITIES, MEAN, std, i+1)
  
  end = time.time()
  print("Time taken to generate matrices: ", end - start)
  # std08 took 3-4 hours to generate 100 matrices (50 integers and 50 floats)
  # std30 took 2-3 hours to generate 100 matrices (50 integers and 50 floats)

  # stopped at "solving integer 48cities_avg10_std04_25.json"
