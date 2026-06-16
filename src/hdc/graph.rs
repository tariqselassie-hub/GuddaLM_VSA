// Copyright (C) 2025 guddalm_vsa contributors.
// SPDX-License-Identifier: AGPL-3.0
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
/// Graph Encoding with Hyperdimensional Vectors.
///
/// This module provides tools for encoding graph structures (nodes and edges)
/// into distributed hypervector representations and for querying them.
///
/// ## Encoding Strategy
///
/// ### MAP (bipolar) graph encoding
/// Each node is assigned a random **role vector**. Each edge
/// `source --relation--> target` is encoded as:
///
///   `edge = bind(role[source], bind(relation, role[target]))`
///
/// The full graph is the **bundle** of all edge vectors:
///
///   `graph = bundle over all edges of edge_vector`
///
/// This is the VSA analog of an adjacency matrix: the graph vector
/// preserves information about which nodes are connected and via what
/// relation, in a distributed, superposition-resilient form.
///
/// ### GHRR graph encoding
/// When using `GHRRVector` (non-commutative U(2) matrices), edges
/// naturally encode direction because matrix multiplication does not
/// commute: `bind(A, B) ≠ bind(B, A)`. This makes GHRR particularly
/// suitable for directed graphs.
///
/// ## Querying
/// - **Node neighborhood**: unbind the relation and target role to
///   recover source nodes that connect through it.
/// - **Path finding**: composition of relations via binding enables
///   multi-hop queries.
/// - **Subgraph isomorphism**: compare graph vectors via cosine
///   similarity — structurally similar graphs yield higher similarity.
use crate::hdc::vector::HDVector;
use crate::hdc::ghrr::GHRRVector;
/// MAP-based graph encoder for directed graphs.
///
/// Each node is assigned a unique random HD vector (role). Edges are
/// encoded as `bind(source, bind(relation, target))`. The graph is the
/// bundle of all edges.
pub struct GraphEncoder {
    /// Node role vectors: index → HDVector.
    pub node_roles: Vec<HDVector>,
    /// Dimensionality.
    dim: usize,
}

impl GraphEncoder {
    /// Create a new graph encoder with `num_nodes` random role vectors.
    pub fn new(num_nodes: usize, dim: usize) -> Self {
        let node_roles: Vec<HDVector> = (0..num_nodes).map(|_| HDVector::random(dim)).collect();
        GraphEncoder {
            node_roles,
            dim,
        }
    }

    /// Create a graph encoder from pre-existing role vectors.
    pub fn from_roles(node_roles: Vec<HDVector>) -> Self {
        let dim = node_roles[0].dim();
        GraphEncoder {
            node_roles,
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn num_nodes(&self) -> usize {
        self.node_roles.len()
    }

    /// Encode a single directed edge: `source --relation--> target`.
    ///
    /// Returns `bind(role[source], bind(relation, role[target]))`.
    pub fn encode_edge(
        &self,
        source: usize,
        relation: &HDVector,
        target: usize,
    ) -> HDVector {
        let s_role = &self.node_roles[source];
        let t_role = &self.node_roles[target];
        let bound_rt = relation.bind(t_role);
        s_role.bind(&bound_rt)
    }

    /// Encode an entire graph from a list of edges.
    ///
    /// Each edge is `(source, relation, target)` where relation is an
    /// HDVector. All edge vectors are bundled into a single graph vector.
    ///
    /// The result is binarized for noise robustness.
    pub fn encode_graph(
        &self,
        edges: &[(usize, HDVector, usize)],
    ) -> HDVector {
        if edges.is_empty() {
            return HDVector::zeros(self.dim);
        }
        let mut acc = crate::hdc::stream::BundleAccumulator::new(self.dim);
        for (src, rel, tgt) in edges {
            let edge = self.encode_edge(*src, rel, *tgt);
            acc.add(&edge);
        }
        acc.binarize()
    }

    /// Encode a graph with a single shared relation type.
    ///
    /// Convenience wrapper for graphs where all edges share the same
    /// relation. `adjacency` is a list of `(source, target)` pairs.
    pub fn encode_graph_unary(
        &self,
        adjacency: &[(usize, usize)],
        relation: &HDVector,
    ) -> HDVector {
        let edges: Vec<(usize, HDVector, usize)> = adjacency
            .iter()
            .map(|&(s, t)| (s, relation.clone(), t))
            .collect();
        self.encode_graph(&edges)
    }

    /// Query: given a relation and target node, recover the source nodes.
    ///
    /// Performs `unbind(graph, bind(relation, role[target]))` to recover
    /// a superposition of source role vectors. The result can be cleaned
    /// up against the node role codebook to identify specific sources.
    pub fn query_predecessors(
        &self,
        graph: &HDVector,
        relation: &HDVector,
        target: usize,
    ) -> HDVector {
        let t_role = &self.node_roles[target];
        let bound_rt = relation.bind(t_role);
        graph.unbind(&bound_rt)
    }

    /// Query: given a source node and relation, recover the target nodes.
    ///
    /// Unbinds source role, then relation, from the graph.
    pub fn query_successors(
        &self,
        graph: &HDVector,
        source: usize,
        relation: &HDVector,
    ) -> HDVector {
        let s_role = &self.node_roles[source];
        // unbind(graph, source) = bind(relation, target_superposition)
        let without_source = graph.unbind(s_role);
        // unbind(without_source, relation) = target_superposition
        without_source.unbind(relation)
    }

    /// Query whether a specific edge exists in the graph.
    ///
    /// Returns a similarity score indicating the strength of evidence
    /// for the edge `source --relation--> target`.
    pub fn query_edge(
        &self,
        graph: &HDVector,
        source: usize,
        relation: &HDVector,
        target: usize,
    ) -> f64 {
        let edge = self.encode_edge(source, relation, target);
        graph.cosine_similarity(&edge)
    }

    /// Structural similarity between two encoded graphs.
    ///
    /// Two graphs that share similar edge structure will have high
    /// cosine similarity between their graph vectors.
    pub fn structural_similarity(a: &HDVector, b: &HDVector) -> f64 {
        a.cosine_similarity(b)
    }
}

/// Encode a simple path as a chain of bound relations with positional permutation.
///
/// Given nodes `[n₀, n₁, ..., nₖ]` and relations `[r₁, r₂, ..., rₖ]`
/// where `rᵢ` connects `nᵢ₋₁ → nᵢ`, the path is encoded as:
///
///   `path = permute(role[n₀] ⊛ r₁, 0) ⊛ permute(role[n₁] ⊛ r₂, 1) ⊛ ... ⊛ role[nₖ]`
///
/// Position-dependent permutation breaks MAP binding's commutativity,
/// ensuring that `0→1→2` and `0→2→1` produce distinct path vectors.
pub fn encode_path(
    encoder: &GraphEncoder,
    nodes: &[usize],
    relations: &[HDVector],
) -> HDVector {
    assert_eq!(nodes.len(), relations.len() + 1,
        "path needs one more node than relations: N nodes, N-1 edges");

    let mut combined = encoder.node_roles[nodes[0]].clone();
    for i in 0..relations.len() {
        let edge_segment = relations[i].bind(&encoder.node_roles[nodes[i + 1]]);
        let permuted = edge_segment.permute(i); // position-dependent permutation
        combined = combined.bind(&permuted);
    }
    combined
}

/// Compare two encoded paths for structural similarity.
pub fn path_similarity(path_a: &HDVector, path_b: &HDVector) -> f64 {
    path_a.cosine_similarity(path_b)
}

// ═══════════════════════════════════════════════════════════════
// GHRR Graph Encoder — non-commutative directed graph encoding
//
// GHRR's non-commutative bind(a,b) ≠ bind(b,a) naturally captures
// edge direction.  Each edge is encoded via bundled superposition:
//
//   edge(u, r, v)  = bind(role[u], bind(relation, role[v]))
//   graph(G)       = ⊕_{edges} edge(u, r, v)
//
// Query successors (recover target from source+relation):
//   inv(rel) ⊛ (inv(role[source]) ⊛ graph)  ≈  role[target]
//
// Because matrix multiplication is associative, the inverses cancel
// in the correct position regardless of other superimposed edges.
// Non-commutativity ensures bind(a,b) and bind(b,a) are distinct,
// so directed edges A→B and B→A produce different edge vectors.
// ═══════════════════════════════════════════════════════════════

/// GHRR-based graph encoder for directed graphs.
///
/// Each node is assigned a random GHRRVector (U(2) unitary blocks).
/// Edges are encoded as `bind(role[source], bind(relation, role[target]))`
/// and the graph is the **bundled superposition** of all edges.
///
/// Querying uses the non-commutative inverse to peel off layers:
/// `inv(rel) ⊛ (inv(source) ⊛ graph)` recovers the target superposition.
pub struct GhrrGraphEncoder {
    pub node_roles: Vec<GHRRVector>,
    dim: usize,
}

impl GhrrGraphEncoder {
    /// Create a new GHRR graph encoder with `num_nodes` random role vectors.
    pub fn new(num_nodes: usize, dim: usize) -> Self {
        let node_roles: Vec<GHRRVector> = (0..num_nodes).map(|_| GHRRVector::random(dim)).collect();
        GhrrGraphEncoder { node_roles, dim }
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn num_nodes(&self) -> usize { self.node_roles.len() }

    /// Encode a single directed edge: `source --relation--> target`.
    ///
    /// Returns `bind(role[source], bind(relation, role[target]))`.
    /// For GHRR this equals: source ⊛ rel ⊛ target  (associative).
    pub fn encode_edge(
        &self,
        source: usize,
        relation: &GHRRVector,
        target: usize,
    ) -> GHRRVector {
        let s = &self.node_roles[source];
        let t = &self.node_roles[target];
        s.bind(&relation.bind(t))
    }

    /// Encode an entire graph from a list of edges.
    ///
    /// All edge vectors are **bundled** (superposed), not chained.
    /// The result is projected back onto the unitary group.
    pub fn encode_graph(
        &self,
        edges: &[(usize, GHRRVector, usize)],
    ) -> GHRRVector {
        if edges.is_empty() {
            return GHRRVector::identity(self.dim);
        }
        let edges: Vec<GHRRVector> = edges
            .iter()
            .map(|&(ref s, ref r, ref t)| self.encode_edge(*s, r, *t))
            .collect();
        let mut acc = edges[0].clone();
        for e in &edges[1..] {
            acc = acc.bundle(e);
        }
        acc.project()
    }

    /// Encode a graph with a single shared relation type.
    pub fn encode_graph_unary(
        &self,
        adjacency: &[(usize, usize)],
        relation: &GHRRVector,
    ) -> GHRRVector {
        let edges: Vec<(usize, GHRRVector, usize)> = adjacency
            .iter()
            .map(|&(s, t)| (s, relation.clone(), t))
            .collect();
        self.encode_graph(&edges)
    }

    /// Query successors: given source node and relation, recover target nodes.
    ///
    /// Computes `inv(rel) ⊛ (inv(role[source]) ⊛ graph)`.
    /// In the noiseless single-edge case this exactly recovers role[target].
    pub fn query_successors(
        &self,
        graph: &GHRRVector,
        source: usize,
        relation: &GHRRVector,
    ) -> GHRRVector {
        let inv_source = self.node_roles[source].inverse();
        let inv_rel = relation.inverse();
        // inv(rel) ⊛ (inv(source) ⊛ graph)
        inv_rel.bind(&inv_source.bind(graph))
    }

    /// Query whether a specific edge exists in the graph.
    pub fn query_edge(
        &self,
        graph: &GHRRVector,
        source: usize,
        relation: &GHRRVector,
        target: usize,
    ) -> f64 {
        let edge = self.encode_edge(source, relation, target);
        graph.cosine_similarity(&edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;

    #[test]
    fn test_single_edge_roundtrip() {
        let dim = 512;
        let encoder = GraphEncoder::new(5, dim);
        let relation = HDVector::random(dim);

        // Encode edge: 0 --rel--> 2
        let edge = encoder.encode_edge(0, &relation, 2);

        // Query: which source connects to target 2 via this relation?
        let graph = edge.clone(); // graph is just this one edge
        let recovered = encoder.query_predecessors(&graph, &relation, 2).binarize();

        let sim = recovered.cosine_similarity(&encoder.node_roles[0]);
        assert!(
            sim > 0.3,
            "must recover source node 0 from single-edge graph (sim = {})",
            sim
        );
    }

    #[test]
    fn test_graph_encode_unary() {
        let dim = 256;
        let encoder = GraphEncoder::new(4, dim);
        let rel = HDVector::random(dim);

        // Triangle: 0→1, 1→2, 2→0
        let adjacency = vec![(0, 1), (1, 2), (2, 0)];
        let graph = encoder.encode_graph_unary(&adjacency, &rel);

        // Verify each edge has positive similarity
        for &(s, t) in &adjacency {
            let sim = encoder.query_edge(&graph, s, &rel, t);
            assert!(
                sim > 0.1,
                "edge {}→{} must be detectable (sim = {})",
                s,
                t,
                sim
            );
        }
    }

    #[test]
    fn test_query_successors() {
        let dim = 512;
        let encoder = GraphEncoder::new(3, dim);
        let rel = HDVector::random(dim);

        let graph = encoder.encode_edge(0, &rel, 1);

        let successors = encoder.query_successors(&graph, 0, &rel).binarize();
        let sim_to_1 = successors.cosine_similarity(&encoder.node_roles[1]);
        assert!(
            sim_to_1 > 0.25,
            "must recover target node 1 from query_successors (sim = {})",
            sim_to_1
        );
    }

    #[test]
    fn test_structural_similarity_self() {
        let dim = 256;
        let encoder = GraphEncoder::new(3, dim);
        let rel = HDVector::random(dim);
        let adjacency = vec![(0, 1), (1, 2)];
        let graph = encoder.encode_graph_unary(&adjacency, &rel);

        let sim = GraphEncoder::structural_similarity(&graph, &graph);
        assert!((sim - 1.0).abs() < 0.001, "graph must be identical to itself");
    }

    #[test]
    fn test_path_encoding() {
        let dim = 256;
        let encoder = GraphEncoder::new(4, dim);
        let rel = HDVector::random(dim);

        // Path: 0 → 1 → 2 → 3
        let nodes = vec![0, 1, 2, 3];
        let relations = vec![rel.clone(), rel.clone(), rel.clone()];
        let path = encode_path(&encoder, &nodes, &relations);

        // A different path should have lower similarity
        let other_nodes = vec![0, 2, 1, 3];
        let other_path = encode_path(&encoder, &other_nodes, &relations);

        let sim = path_similarity(&path, &other_path);
        assert!(
            sim < 0.7,
            "different paths must have lower similarity (sim = {})",
            sim
        );
    }

    #[test]
    fn test_different_relations_distinguish_edges() {
        let dim = 256;
        let encoder = GraphEncoder::new(3, dim);
        let rel_a = HDVector::random(dim);
        let rel_b = HDVector::random(dim);

        let edge_a = encoder.encode_edge(0, &rel_a, 1);
        let edge_b = encoder.encode_edge(0, &rel_b, 1);

        let sim = edge_a.cosine_similarity(&edge_b);
        assert!(
            sim < 0.3,
            "edges with different relations must be dissimilar (sim = {})",
            sim
        );
    }

    #[test]
    fn test_empty_graph() {
        let dim = 128;
        let encoder = GraphEncoder::new(5, dim);
        let graph = encoder.encode_graph(&[]);
        assert_eq!(graph.dim(), dim);
    }
}
