import numpy as np
import numpy.ma as ma
import heapq

class Node:
  """Represents a node in the Tree of Little's Algorithm"""
  
  def set_children(self, children):
    """
    Set's the children property to the children argument

    Parameters
    ----------
    self: Node
      reference to the instance
    children: List
      List of the two children of this node
    """
    self.children = children

  def set_parent(self, node):
    """
    Set's the parent property to paramer node and inherits the
    excludes and includes properties of node as copies.

    Parameters
    ----------
    self: Node
      reference to the instance
    node: Node
      Node that is the parent of self
    """
    self.includes = node.includes.copy()
    self.excludes = node.excludes.copy()
    self.parent = node

  def __lt__(self, other):
    """
    Less then operator used by the HeapQ algorithm to retrieve the 
    Node with the lowest lower bound.

    Parameters
    ----------
    self: Node
      reference to the instance
    """
    return self.lower_bound < other.lower_bound

  def __init__(
    self,
    lower_bound,
    left,
    edge=None,
    x=None,
    y=None,
    root=False,
    includes=[],
    excludes=[]
  ):
    """

    """
    self.y = y
    self.x = x
    self.left = left
    self.lower_bound = lower_bound
    self.root = root
    self.includes = []
    self.excludes = []
    self.edge = edge

# Code starts by calculating the column reduction then the row reduction?
def reduce_rows(matrix):
  """
  Substracts the lowest value of each row from the corresponding
  column. Returns the resulting matrix and the sum of all the values
  subtracted.

  Parameters
  ----------
  matrix: np.array
    numpy array corresponding to a tsp instance

  Returns
  -------
  np.array
    matrix after reduction
  np.float
    sum of all the values subtracted
  """
  matrix, reduction = reduce_colums(matrix.T)
  return matrix.T, reduction


def reduce_colums(matrix):
  """
  Substracts the lowest value of each column from the corresponding
  column. Returns the resulting matrix and the sum of all the values
  subtracted.

  Parameters
  ----------
  matrix: np.array
    numpy array corresponding to a tsp instance

  Returns
  -------
  np.array
    matrix after reduction
  np.float
    sum of all the values subtracted
  """
  minimum = np.min(matrix, axis=0)
  matrix = matrix - minimum
  reduction = np.sum(minimum)
  return matrix, reduction


def reduce_matrix(input_matrix):
  """
  Substracts the lowest value of each column and each row from the corresponding
  column or row respectively. Returns the resulting matrix and the sum of all 
  the values subtracted.

  Parameters
  ----------
  matrix: np.array
    numpy array corresponding to a tsp instance

  Returns
  -------
  np.array
    matrix after reduction
  np.float
    sum of all the values subtracted
  """
  matrix = input_matrix.copy()
  matrix, row_reduction = reduce_rows(matrix)

  if row_reduction == np.inf:
    return matrix, row_reduction

  matrix, column_reduction = reduce_colums(matrix)
  reduction = row_reduction + column_reduction
  return matrix, reduction

def calc_jump_in_lower_bound(matrix):
  """
  Calculates the theta parameter of the algorithm for each element of
  matrix. It does so by calculating what the sum of the lowest column
  and row value is while ommiting that particular element.

  Parameters
  ----------
  matrix: np.array
    numpy array corresponding to a tsp instance

  Returns
  -------
  np.array
    matrix that has for each element the theta value
  """
  result_matrix = np.zeros(matrix.shape)
  size = matrix.shape[0]
  coords = ma.where(matrix == 0)

  for i in range(len(coords[0])):
    x = (coords[0][i])
    y = (coords[1][i])

    jump_lower_bound = 0
    row = matrix[x, :]
    row_mask = np.zeros(size)
    row_mask[y] = 1
    masked_row = ma.array(row, mask=row_mask)
    if not masked_row.mask.all():
      jump_lower_bound += np.min(masked_row)

    column = matrix[:, y]
    column_mask = np.zeros(size).T
    column_mask[x] = 1
    masked_column = ma.array(column, mask=column_mask)
    if not masked_column.mask.all():
      jump_lower_bound += np.min(masked_column)

    result_matrix[x, y] += jump_lower_bound
  return result_matrix

def get_inf_mask(matrix):
  """
  Returns a mask to use in a numpy.ma array for all values
  of infinity in matrix.

  Parameters
  ----------
  matrix: np.array
    numpy array corresponding to a tsp instance

  Returns
  -------
  np.array
    mask for all values in matrix set to infinity
  """
  return (matrix == np.inf)

def choose_city_pair_to_branch(matrix):
  """
  Chooses the edge that has the highest theta value (the sum of the lowest 
  column and row value is while ommiting that particular element) to branch on.
  In addition it returns the tetha value.

  Parameters
  ----------
  matrix: np.array
    numpy array corresponding to a tsp instance

  Returns
  -------
  Int:
    index of one of the cities the chosen edge is connected to
  Int:
    index of one of the cities the chosen edge is connected to
  Float:
    theta value (jump in lower bound if edge is ommited)
  """
  inf_mask = get_inf_mask(matrix)
  masked_matrix = ma.array(matrix, mask=inf_mask)

  masked_jumps = calc_jump_in_lower_bound(masked_matrix)

  x, y = np.unravel_index(masked_jumps.argmax(), masked_jumps.shape)


  if masked_jumps.max() == 0:
    indices = ma.where(masked_matrix == 0)
    x = indices[0][0]
    y = indices[1][0]

  jump = masked_jumps[x, y]
  return x, y, jump

def branch_left(x, y, jump, parent):
  """
  Create a left branch on the tree.

  Parameters
  ----------
  x: Int
    index of one of the cities the chosen edge is connected to
  y: Int
    index of one of the cities the chosen edge is connected to
  jump: Float
    jump in lower bound when (x, y) is ommited
  parent: Node
    parent of the new child being made in the tree
  """
  child = Node(parent.lower_bound + jump, True, edge=(x, y))
  child.set_parent(parent)
  return child

def traverse_to_root(node, func):
  """
  Traverse all the way to the root node from :node and applies func
  to every node being traversed.

  Parameters
  ----------
  node: Node
    starting point to traverse up from
  func: Function
    function to apply to every node encountered in the traversal
  """
  if node.root:
    return
  func(node)
  traverse_to_root(node.parent, func)


def find_all_subtours(included):
  """
  Finds the edges in the tsp instances which cannot be
  includes anymore because they would create a subtour with
  edges currently in the tour.

  Parameters
  ----------
  includes: List
    edges that are currently in the tour
  fromCity: Int
    index of one of the cities the chosen edge is connected to
  toCity: Int
    index of one of the cities the chosen edge is connected to

  Returns
  -------
  Tuple:
    edge that would make the current DE

  """
  included_set = set(included)
  paths = []

  if len(included) == 0:
    return paths

  included_nodes = set()
  edge_to = {}
  edge_from = {}

  for from_node, to_node in included:
    included_nodes.add(from_node)
    included_nodes.add(to_node)

    edge_from[from_node] = to_node
    edge_to[to_node] = from_node

  def expand(current_edge, included_set):
    if (edge_to.get(current_edge[0])):
      new_edge = (edge_to.get(current_edge[0]), current_edge[1])
      try:
        included_set.remove((edge_to.get(current_edge[0]), current_edge[0]))
      except: pass

      return expand(new_edge, included_set)
    if (edge_from.get(current_edge[1])):
      try:
        included_set.remove((current_edge[1], edge_from.get(current_edge[1])))
      except: pass
  
      return expand(
        (current_edge[0], edge_from.get(current_edge[1])), included_set)
    return current_edge, included_set
      

  while len(included_set) > 0:
    path, included_set = expand(included_set.pop(), included_set)
    paths.append(path)

  return paths

  

def set_up_matrix(initial_matrix, node):
  """
  Set up matrix before branching procedure. this is used so the matrices
  for each node do not have to be stored in memory. For each included edge
  the row of the emanating node and the column of the terminating node are
  crossed out, and the inverse edge is set to infinity. All exluded edges
  are also set to infinity. The edge that closes a set of other edges wich are
  already included are also set to infinity.

  Parameters
  ----------
  initial_matrix: np.array
    the initial matrix that represents the TSP instance
  node: Node
    the current node that is being branched from
  """
  matrix = initial_matrix.copy()
  included = []
  excluded = []

  def add_edges(node):
    if (node.left):
      excluded.append(node.edge)
    else:
      included.append(node.edge)

  traverse_to_root(node, add_edges)

  mask = np.zeros(initial_matrix.shape)
  cost_of_included = 0
  for from_node, to_node in included:
    matrix[to_node, from_node] = np.inf
    mask[from_node, :] = 1
    mask[:, to_node] = 1
    cost_of_included = cost_of_included + initial_matrix[from_node, to_node]

  for from_node, to_node in excluded:
    matrix[from_node, to_node] = np.inf

  sub_tours = find_all_subtours(included)
  for from_node, to_node in sub_tours:
    matrix[to_node, from_node] = np.inf

  reduced_matrix, lower_bound = reduce_matrix(ma.array(matrix, mask=mask))
  return reduced_matrix, lower_bound + cost_of_included

def branch_right(x, y, parent, initial_matrix):
  """
  Create a right branch on the tree.

  Parameters
  ----------
  x: Int
    index of one of the cities the chosen edge is connected to
  y: Int
    index of one of the cities the chosen edge is connected to
  parent: Node
    parent of the new child being made in the tree

  Returns
  -------
  Node:
    New node that is created on the right side of parent
  """
  child = Node(parent.lower_bound, False, edge=(x, y))
  child.set_parent(parent)

  reduced_matrix, lower_bound = set_up_matrix(initial_matrix, child)
  child.lower_bound = lower_bound
  return child, reduced_matrix


def get_best_tour(current_best_tour, node, finished_up):
  """
  Checks if the node forms the new current best tour.

  Parameters
  ----------
  current_best_tour: Tuple
    first element: cost of current best tour
    second element: currently best tour
  node: Node
    node to check if it is the new best tour
  finished_up: List
    two final edges to finish up the tour

  Returns
  -------
  Tuple:
    first element: cost of new best tour
    second element: new best tour
  """
  included = []

  def add_edges(node):
    if not node.left:
      included.append(node.edge)

  traverse_to_root(node, add_edges)

  if (node.lower_bound < current_best_tour[0]):
    return (node.lower_bound, included + finished_up)
  else:
    return current_best_tour

def get_minimal_route(initial_matrix):
  """
  Finds the shortest route in the tsp inistance represented by initial_matrix
  with Little's (https://www.jstor.org/stable/167836).

  Parameters
  ----------
  initial_matrix: np.array
    matrix that represents the cost matrix of a TSP instance
  
  Returns
  -------
  Int:
    amount of iterations used by the algorithm to find a solution
  List
    edges that make up the tour with the shortest distance
  Float:
    cost of the best tour 
  """
  matrix, lower_bound = reduce_matrix(initial_matrix)
  current_node = Node(lower_bound, False, root=True)

  priority_queue = []
  iterations = 0
  best_tour = (np.inf, [])

  algorithm_finished = False
  while not algorithm_finished:
    iterations = iterations + 1
    matrix, _ = set_up_matrix(initial_matrix, current_node)
    x, y, jump = choose_city_pair_to_branch(matrix)

    child1 = branch_left(x, y, jump, current_node)
    child2, matrix_child2 = branch_right(x, y, current_node, initial_matrix)
    current_node.set_children([child1, child2])

    if (matrix_child2.count() == 4):
      zero_indices = np.where(matrix_child2 == 0)
      finished_up = [(zero_indices[0][0], zero_indices[1][0]),
        (zero_indices[0][1], zero_indices[1][1])]

      best_tour = get_best_tour(best_tour, child2, finished_up)
    else:
      heapq.heappush(priority_queue, child1)
      heapq.heappush(priority_queue, child2)

    current_node = heapq.heappop(priority_queue)
    if (current_node.lower_bound >= best_tour[0]):
      algorithm_finished = True

  return (iterations, best_tour[1], best_tour[0])
