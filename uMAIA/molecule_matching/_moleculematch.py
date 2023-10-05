import os

os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import gurobipy as gp
import numpy as np
import networkx as nx

# multithreading
from threadpoolctl import threadpool_limits
threadpool_limits(4)



class MoleculeMatcher:
    """
    MoleculeMatcher takes in a list of lists representing mz detections across different acquisitions. Each sublist provided to MoleculeMatcher indicates a different acquisition.
    The output of MoleculeMatcher are a series of edges that are placed between detections in the various sections to indicate which molecules are one and the same.
    
    Args
    ----
    mz: List[List]
        Each sublist in mz contain m/z detections from a single acquisition
    NUM_SKIP: Int
        The number of neighboring acquisitions to consider when constructing edges. In general, the greater this value, the more accurate results will be. However, longer processing times will be required for optimization.
    NUM_PERMS: Int
        Value indicating the number of shuffles of acquisition order to consider. The solution returned is the shuffle that produces the best objective cost. This value should scale with the number of sections provided to the solver.
    STD_NORMAL_NOISE: Float
        Estimated size of bins for molecules, i.e., the extent to which one expects noise around a specific mz value
    K: Int
        The number of nearest neighbors to consider when drawing edges from a given detection in a single acquisition
    MAX_DIST: Float
        The cutoff distance for considering a potential edge
    num_threads: Int
        Number of threads to run optimization on
    """


    def __init__(self, mz, mzPermutations=None, NUM_SKIP=2, NUM_PERMS=10, STD_NORMAL_NOISE=0.01, K=3, MAX_DIST=0.1, num_threads=4):
        self.mz = np.array(mz, dtype=object)
        self.NUM_SKIP = NUM_SKIP
        self.NUM_PERMS = NUM_PERMS
        self.STD_NORMAL_NOISE = STD_NORMAL_NOISE
        self.K = K
        self.K = max(self.K, self.NUM_SKIP)
        self.MAX_DIST = MAX_DIST
        self.num_threads = num_threads
        
        self.numSections = len(self.mz)
        if mzPermutations is None:
            self.mzPermutations = []
        else:
            self.mzPermutations = [mzPermutations]
        
        
    def create_permutations(self):
        """
        Create a set of random permutations of the acquisitions based on NUM_PERMS provided
        """
        for i in range(self.NUM_PERMS):
            idx = np.arange(self.numSections)
            idxPerm = np.random.choice(idx, len(idx), replace=False)
            self.mzPermutations.append(idxPerm)

    def get_edge_costs_for_setup(self, mz_permutated, verbose=False):
        """
        Construct list of lists containing source node, end node and cost affiliated with the edge that would connected the two.

        Args
        ----
        mz_permutated: List
            List indicating how to permutate the acquisition numbers

        Returns
        -------
        edge_cost: Dict
            Dictionary containing nodes as keys and costs as values. Values are made into negative values by subtracting the maximum cost in the set of proposed edges.
        edges_unique: Set
            Set containing unique edges between true detections based on proposed edges
        """
        edge_cost = {}
        np.random.seed(0)

        if verbose:
            print(mz_permutated)
            print(self.NUM_PERMS, self.NUM_SKIP, self.MAX_DIST)

        for i, entry in enumerate(mz_permutated):
            # initate start node
            if i == 0:
                for ii, e_mz in enumerate(entry): # for every detection in the first section, connect to a start node
                    e_node = (i,ii)
                    s_node = 's'
                    cost = 0
                    edge_cost[s_node, e_node] = cost

            # find the subset of edges that must exist with sections that come above
            else:
                edge_cost_subset = {}
                b = -1
                count = 0
                
                # fill the list s_nodes until it has some detection from at least NUM_SKIP sections
                while count < self.NUM_SKIP and count < i and (i + b) >= 0:
                    while len(mz_permutated[i + b]) == 0:
                        b -= 1 # identify the next section with a detection
                    count += 1
                        
                    # append all of these edges to edge_cost_subset
                    s_nodes = mz_permutated[i+b]
                    if verbose:
                        print('s_nodes')
                        print(s_nodes)
                        print('after update b')
                        print(entry, b)

                    for i_s, s_node in enumerate(s_nodes): # iterate through the start section
                        for i_e, e_node in enumerate(entry): # iterate though end node section
                            cost = np.abs(s_node - e_node)
                            if verbose:
                                print(f'cost {cost}, min(snode){np.min(s_node)}, min(enode) {np.min(e_node)} (i+b) {i+b}')

                            #print(cost)
                            if (cost < self.MAX_DIST) and (np.min(s_node) >= 0) and (np.min(e_node) >= 0) and (i + b >= 0):
                                if verbose:
                                    print('entered if statement!!!!!!!!!!')
                                edge_cost_subset[(i + b, i_s), (i, i_e)] = cost
                                

                    b -= 1

                edge_cost_subset_array =np.array(list(edge_cost_subset.items()), dtype=object)
                if verbose:
                    print('edge_cost_subset_array')
                    print(edge_cost_subset_array)
                
                # at this point we have a set of edges that we will consider, but they are not necessarily the nearest neighbors
                # we want to filter out this list, but the number of edges must be at least as great as the number of skips to consider
                # iterate over the edges proposed for a single node (as end node) and select the k shortest paths
                
                for i_e,e_node in enumerate(entry):
                    
                    a = np.array([x for x in edge_cost_subset_array if x[0][1] == (i,i_e)])
                    if len(a) == 0:
                        continue
                    a = a[a[:, 1].argsort()][:self.K] # sort by distance to node 
                    # append edge to dictionary
                    for a_ in a:
                        edge_cost[a_[0]] = a_[1]


        # add end node for last section
        for ii, e_mz in enumerate(entry):
            e_node = 'e'
            cost = 0
            s_node = (i,ii)
            edge_cost[s_node, e_node] = cost

        edge_cost['e', 's'] = 0
        
        # make costs negative
        max_ = np.max(list(edge_cost.values()))  
        for k in edge_cost:
            edge_cost[k] -= max_
            
        # v = np.array(list(edge_cost.values()))
        # v = v[np.nonzero(v)]
        # v = np.percentile(v, 1, interpolation='nearest')


        # need to add that every edge can lead into a end node, and every node can be entered from the start node.
        edges_unique = set(np.array(list(edge_cost), dtype=object)[:,0]).union(set(np.array(list(edge_cost), dtype=object)[:,1])).difference('e','s')
        for edge in edges_unique:
            edge_cost[('s', edge)] = 1. * np.log(self.numSections)#v + self.STD_NORMAL_NOISE * 40
            edge_cost[(edge, 'e')] = 1. * np.log(self.numSections)#v  + self.STD_NORMAL_NOISE * 40 
    
        return edge_cost, edges_unique
        
        
    def get_constraints_for_setup(self, edge_cost, edges_unique):
        # for one node, find all indexes in edges that lead into node
        E_in = []
        E_out = []
        edges = pd.DataFrame(np.array(list(edge_cost), dtype=object))
        for edge in edges_unique:
            ix_in = edges.iloc[:,1] == edge
            ix_out = edges.iloc[:,0] == edge

            E_in.append(ix_in.values)
            E_out.append(ix_out.values)

        E_in = np.vstack(E_in)
        E_out = np.vstack(E_out)
        return E_in, E_out
        
    def optimize_setup(self, edge_cost, E_in, E_out):
        """
        Using set of proposed edges and constraints, optimize the solution to retrieve selected edges

        Args
        ----
        edge_cost: Dict
        E_in: Array
            indicates the set of edges that lead into a single node
        E_out: Array
            indicates the set of edges that lead out of a single node

        Returns
        -------
        output: Array
            Binary array containing 1 if proposed edge was selected and 0 otherwise
        selected_edges: Array
            Array containing tuples in the same way as edge_cost, filtered according to selected edges
        obj_val: Float
            objective value affiliated with optimized solution
        """
        # Build model m here
        m = gp.Model('netflow')
        m.Params.LogToConsole = 0 # suppress output
        m.Params.Threads = self.num_threads
        x = m.addMVar(shape=len(edge_cost),  vtype=gp.GRB.BINARY, name="x")
        # Set objective
        m.setObjective(x @ np.array(list(edge_cost.values())).flatten(), gp.GRB.MINIMIZE)
        # add constraints
        m.addConstr(E_in @ x == E_out @ x) # flow is conserved
        m.addConstr(E_in @ x == 1)
        # Compute optimal solution
        m.optimize()

        output = x.X
        selected_edges = np.array(output, dtype=bool)
        selected_edges = np.array(list(edge_cost), dtype=object)[selected_edges]
        
        return output, selected_edges, m.objVal


    def run_setup(self, mz_permutation):

        edge_cost, edges_unique = self.get_edge_costs_for_setup(mz_permutation)
        if len(edges_unique) == 0:
            return np.nan, np.nan, np.nan
        try:
            E_in, E_out = self.get_constraints_for_setup(edge_cost=edge_cost, edges_unique=edges_unique)
        except:
            edge_cost, edges_unique = self.get_edge_costs_for_setup(mz_permutation, verbose=True)
            print(edges_unique)
            print(edge_cost, edges_unique)
            
        output, selected_edges, obj_val = self.optimize_setup(edge_cost=edge_cost, E_in=E_in, E_out=E_out)
        
        return output, selected_edges, obj_val
    
    def assess_permutations(self):
        """
        Function to assess all permutations and provide optimized solution
        """
        # initialize lists of outputs/selected_edges/obj_vals
        output_list = []
        selected_edges_list = []
        obj_val_list = []
        
        # check that permutations are available
        if len(self.mzPermutations) == 0:
            self.create_permutations()
            
        for mz_permutation in self.mzPermutations:
            output, selected_edges, obj_val = self.run_setup(self.mz[mz_permutation])
            
            # append to lists
            output_list.append(output)
            selected_edges_list.append(selected_edges)
            obj_val_list.append(obj_val)

        # retrieve best sequence
        best_ix = np.argmin(obj_val_list)
        if np.isnan(obj_val_list[best_ix]):
            return None

        self.edge_cost, self.edges_unique = self.get_edge_costs_for_setup(self.mz[self.mzPermutations[best_ix]])
        self.E_in, self.E_out = self.get_constraints_for_setup(self.edge_cost, self.edges_unique)
        return self.mzPermutations[best_ix], output_list[best_ix], selected_edges_list[best_ix], obj_val_list[best_ix], self.edge_cost, self.edges_unique
