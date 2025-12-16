# import networkx as nx
# import numpy as np
# size =4
# x = np.array([1/size] * size)
# topo = np.empty((size, size))
# for i in range(size):
#      topo[i] = np.roll(x, i)
# G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
# print(topo,x)
# print(G.edges.data('weight'))
# print(f"节点数: {G.number_of_nodes()}")
# print(f"边数: {G.number_of_edges()}")
# print(f"节点: {list(G.nodes())}")
# print(f"边: {list(G.edges())}")
# print(f"图的度: {G.degree()}")  # 每个节点的总度数
from typing import Dict, List, Set
from dataclasses import dataclass
import networkx as nx
from loguru import logger

@dataclass
class CommunicationEdge:
    """Communication edge representing a group of workers that communicate together.
    
    Args:
        ranks (List[int]): List of ranks that communicate in this edge
        weight (float): Weight for the communication (0-1)
    """
    ranks: List[int]
    weight: float

class SimpleGraphTopology:
    """Simple topology based on an undirected graph, ensuring no duplicate communication."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.world_size = len(graph.nodes())
        self.edges: List[CommunicationEdge] = []
        self._build_communication_edges()
    
    def _build_communication_edges(self):
        """Build communication edges by processing each node exactly once."""
        used_nodes: Set[int] = set()
        
        # Process nodes in order
        for node in sorted(self.graph.nodes()):
            if node in used_nodes:
                continue
                
            # Find neighbors that haven't been used yet
            available_neighbors = [
                neighbor for neighbor in self.graph.neighbors(node) 
                if neighbor not in used_nodes
            ]
            
            if available_neighbors:
                # Use the first available neighbor
                neighbor = available_neighbors[0]
                
                # Simple weight calculation
                weight = 0.5
                
                self.edges.append(CommunicationEdge(
                    ranks=sorted([node, neighbor]),
                    weight=weight
                ))
                used_nodes.add(node)
                used_nodes.add(neighbor)
            else:
                # No available neighbors, node communicates alone
                self.edges.append(CommunicationEdge(
                    ranks=[node],
                    weight=1.0
                ))
                used_nodes.add(node)
    
    def get_communication_edges(self) -> List[CommunicationEdge]:
        """Get all communication edges."""
        return self.edges
    
    def get_communication_groups(self) -> List[List[int]]:
        """Get all communication groups as lists of ranks."""
        return [edge.ranks for edge in self.edges]
    
    def validate_topology(self) -> bool:
        """Validate that no rank appears in multiple edges."""
        used_ranks: Set[int] = set()
        
        for edge in self.edges:
            for rank in edge.ranks:
                if rank in used_ranks:
                    logger.error(f"Rank {rank} appears in multiple edges!")
                    return False
                used_ranks.add(rank)
        
        logger.info(f"Topology validation passed: {len(used_ranks)} unique ranks used")
        return True
    
    def print_topology(self):
        """Print the topology information."""
        print(f"Graph Topology with {self.world_size} nodes")
        print(f"Number of communication edges: {len(self.edges)}")
        
        for i, edge in enumerate(self.edges):
            if len(edge.ranks) == 1:
                print(f"  Edge {i}: Isolated node {edge.ranks[0]} (weight: {edge.weight})")
            else:
                print(f"  Edge {i}: Nodes {edge.ranks} communicate (weight: {edge.weight})")

class DegreeBasedTopology(SimpleGraphTopology):
    """Topology that prioritizes high-degree nodes first."""
    
    def _build_communication_edges(self):
        """Build communication edges, processing high-degree nodes first."""
        used_nodes: Set[int] = set()
        
        # Sort nodes by degree (descending)
        nodes_by_degree = sorted(
            self.graph.nodes(),
            key=lambda node: self.graph.degree(node),
            reverse=True
        )
        
        for node in nodes_by_degree:
            if node in used_nodes:
                continue
                
            # Find neighbors that haven't been used yet
            available_neighbors = [
                neighbor for neighbor in self.graph.neighbors(node) 
                if neighbor not in used_nodes
            ]
            
            if available_neighbors:
                # Use the highest-degree available neighbor
                neighbor = max(
                    available_neighbors,
                    key=lambda n: self.graph.degree(n)
                )
                
                # Weight based on degrees
                node_degree = self.graph.degree(node)
                neighbor_degree = self.graph.degree(neighbor)
                total_degree = node_degree + neighbor_degree
                weight = node_degree / total_degree if total_degree > 0 else 0.5
                
                self.edges.append(CommunicationEdge(
                    ranks=sorted([node, neighbor]),
                    weight=weight
                ))
                used_nodes.add(node)
                used_nodes.add(neighbor)
            else:
                # No available neighbors, node communicates alone
                self.edges.append(CommunicationEdge(
                    ranks=[node],
                    weight=1.0
                ))
                used_nodes.add(node)

class ConnectedComponentTopology(SimpleGraphTopology):
    """Topology that processes connected components separately."""
    
    def _build_communication_edges(self):
        """Build communication edges for each connected component."""
        used_nodes: Set[int] = set()
        
        # Process each connected component
        for component in nx.connected_components(self.graph):
            component_nodes = sorted(component)
            
            # Simple pairing within component
            i = 0
            while i < len(component_nodes):
                node = component_nodes[i]
                
                if i + 1 < len(component_nodes):
                    # Pair with next node in component
                    neighbor = component_nodes[i + 1]
                    
                    self.edges.append(CommunicationEdge(
                        ranks=sorted([node, neighbor]),
                        weight=0.5
                    ))
                    used_nodes.add(node)
                    used_nodes.add(neighbor)
                    i += 2
                else:
                    # Last node in component, communicates alone
                    self.edges.append(CommunicationEdge(
                        ranks=[node],
                        weight=1.0
                    ))
                    used_nodes.add(node)
                    i += 1

# Utility functions for common graph types
def create_ring_topology(n_nodes: int) -> SimpleGraphTopology:
    """Create a ring topology with n nodes."""
    G = nx.cycle_graph(n_nodes)
    return SimpleGraphTopology(G)

def create_star_topology(n_nodes: int) -> SimpleGraphTopology:
    """Create a star topology with n nodes (center at node 0)."""
    G = nx.star_graph(n_nodes - 1)
    return SimpleGraphTopology(G)

def create_complete_topology(n_nodes: int) -> SimpleGraphTopology:
    """Create a complete graph topology."""
    G = nx.complete_graph(n_nodes)
    return SimpleGraphTopology(G)

def create_line_topology(n_nodes: int) -> SimpleGraphTopology:
    """Create a line topology with n nodes."""
    G = nx.path_graph(n_nodes)
    return SimpleGraphTopology(G)


# Example usage
if __name__ == "__main__":
    # Example 1: Create a custom graph
    import numpy as np
#     G = nx.Graph()
#     G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2),()])
    size =4
    x = np.array([1/size] * size)
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
        G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    topology = SimpleGraphTopology(G)
    topology.print_topology()
    topology.validate_topology()
    
#     print("\nCommunication edges:")
#     for edge in topology.get_communication_edges():
#         print(f"  Nodes: {edge.ranks}, Weight: {edge.weight}")
    
#     # Example 2: Use different strategies
#     print("\n=== Advanced Topology with Coloring ===")
#     advanced_topology = AdvancedGraphTopology(G, strategy="coloring")
#     advanced_topology.print_topology()
    
#     # Example 3: Predefined topologies
#     print("\n=== Ring Topology (6 nodes) ===")
#     ring_topology = create_ring_topology(6)
#     ring_topology.print_topology()
    
#     print("\n=== Star Topology (5 nodes) ===")
#     star_topology = create_star_topology(5)
#     star_topology.print_topology()
    
#     print("\n=== Mesh Topology (2x3) ===")
#     mesh_topology = create_mesh_topology(2, 3)
#     mesh_topology.print_topology()