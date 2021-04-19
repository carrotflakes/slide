use std::collections::HashMap;

use rand::{seq::SliceRandom, Rng};

use crate::{
    hasher::Hasher,
    lsh::Lsh,
    node::{Node, NodeType},
};

pub struct InferContext {
    pub active_nodes_per_layer: Vec<Vec<usize>>,
    pub active_values_per_layer: Vec<Vec<f32>>,
    pub sizes: Vec<usize>,
}

impl InferContext {
    pub fn new(size: usize) -> Self {
        Self {
            active_nodes_per_layer: vec![Vec::new(); size],
            active_values_per_layer: vec![Vec::new(); size],
            sizes: vec![0; size],
        }
    }
}

pub struct Layer<H: Hasher> {
    node_type: NodeType,
    pub nodes: Vec<Node>,
    rand_node: Vec<usize>,
    pub normalization_constants: Vec<f32>,
    k: usize,
    l: usize,
    previous_layer_num_of_nodes: usize,
    pub sparsity: f32,
    pub hash_tables: Lsh,
    pub hasher: H,
}

impl<H: Hasher> Layer<H> {
    pub fn new(
        number_of_nodes: usize,
        previous_layer_num_of_nodes: usize,
        node_type: NodeType,
        batch_size: usize,
        k: usize,
        l: usize,
        range_pow: usize,
        sparsity: f32,
    ) -> Self {
        let hasher = H::new(k * l, previous_layer_num_of_nodes);
        let mut hash_tables = Lsh::new(k, l, range_pow);

        let mut rand_node: Vec<_> = (0..number_of_nodes).collect();
        let mut rng = rand::thread_rng();
        rand_node.shuffle(&mut rng);

        let mut nodes = Vec::new();
        for i in 0..number_of_nodes {
            let mut weights = vec![0.0; previous_layer_num_of_nodes];
            weights.fill_with(|| rng.gen_range(0.0..0.01));

            let node = Node::new(
                previous_layer_num_of_nodes,
                i,
                node_type,
                batch_size,
                weights,
                rng.gen_range(0.0..0.01),
            );

            {
                // add to hash table
                let hashes = hasher.get_hash(&node.weights, previous_layer_num_of_nodes);
                let hash_indices = hash_tables.hashes_to_indices::<H>(&hashes);
                hash_tables.add(&hash_indices, i + 1);
            }

            nodes.push(node);
        }

        Self {
            node_type,
            nodes,
            rand_node,
            normalization_constants: vec![
                0.0;
                if matches!(node_type, NodeType::Softmax) {
                    batch_size
                } else {
                    0
                }
            ],
            k,
            l,
            previous_layer_num_of_nodes,
            hash_tables,
            hasher,
            sparsity,
        }
    }

    pub fn update_table(&mut self) {
        self.hasher = H::new(self.k * self.l, self.previous_layer_num_of_nodes);
    }

    pub fn random_nodes(&mut self) {
        let mut rng = rand::thread_rng();
        self.rand_node.shuffle(&mut rng);
    }

    pub fn query_active_node_and_compute_activations(
        &mut self,
        InferContext {
            active_nodes_per_layer,
            active_values_per_layer,
            sizes: lengths,
        }: &mut InferContext,
        layer_index: usize,
        input_id: usize,
        label: &[usize],
        sparsity: f32,
    ) -> usize {
        let mut _in = 0;
        let len;
        if sparsity == 1.0 {
            len = self.nodes.len();
            active_nodes_per_layer[layer_index + 1] = (0..len).collect();
        } else {
            let hashes = self.hasher.get_hash_sparse(
                &active_values_per_layer[layer_index],
                lengths[layer_index],
                &active_nodes_per_layer[layer_index],
            );
            let hash_indices = self.hash_tables.hashes_to_indices::<H>(&hashes);
            let actives = self.hash_tables.get_raw(&hash_indices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashtable
            let mut counts = HashMap::<usize, usize>::new(); // TODO: use set
                                                             // Make sure that the true label node is in candidates
            if matches!(self.node_type, NodeType::Softmax) && label.len() > 0 {
                for label in label.iter() {
                    counts.insert(*label, self.l);
                }
            }

            for i in 0..self.l {
                // copy sparse array into (dense) map
                for id in &actives[i] {
                    *counts.get_mut(&(id - 1)).unwrap() += 1;
                }
            }
            _in = counts.len();
            if _in < 1500 {
                let mut rng = rand::thread_rng();
                for i in rng.gen::<usize>() % self.nodes.len()..self.nodes.len() {
                    if counts.len() >= 1000 {
                        break;
                    }
                    counts.entry(self.rand_node[i]).or_insert(0);
                }
            }

            len = counts.len();
            active_nodes_per_layer[layer_index + 1] = counts.keys().cloned().collect();
        }
        lengths[layer_index + 1] = len;

        active_values_per_layer[layer_index + 1] = vec![0.0; len];

        // find activation for all ACTIVE nodes in layer
        for i in 0..len {
            active_values_per_layer[layer_index + 1][i] =
                self.nodes[active_nodes_per_layer[layer_index + 1][i]].get_activation(
                    &active_nodes_per_layer[layer_index],
                    &active_values_per_layer[layer_index],
                    lengths[layer_index],
                    input_id,
                );
        }

        if matches!(self.node_type, NodeType::Softmax) {
            self.normalization_constants[input_id] = 0.0;
            let max_value = active_values_per_layer[layer_index + 1]
                .iter()
                .fold(0.0f32, |a, b| a.max(*b));
            for i in 0..len {
                let real_activation =
                    (active_values_per_layer[layer_index + 1][i] - max_value).exp();
                active_values_per_layer[layer_index + 1][i] = real_activation;
                self.nodes[active_nodes_per_layer[layer_index + 1][i]]
                    .set_last_activation(input_id, real_activation);
                self.normalization_constants[input_id] += real_activation;
            }
        }
        _in
    }
}
