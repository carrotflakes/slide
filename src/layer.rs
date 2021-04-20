use std::collections::HashSet;

use rand::{seq::SliceRandom, Rng};

use crate::{
    hasher::Hasher,
    lsh::Lsh,
    node::{Node, NodeType},
};

pub struct LayerStatus {
    pub active_nodes: Vec<usize>,
    pub active_values: Vec<f32>,
}

impl LayerStatus {
    pub fn size(&self) -> usize {
        self.active_nodes.len()
    }
}

pub struct Layer<H: Hasher> {
    node_type: NodeType,
    pub nodes: Vec<Node>,
    rand_node: Vec<u32>,
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
        k: usize,
        l: usize,
        range_pow: usize,
        sparsity: f32,
        batch_size: usize,
    ) -> Self {
        let hasher = H::new(k * l, previous_layer_num_of_nodes);
        let mut hash_tables = Lsh::new(k, l, range_pow);

        let mut rand_node: Vec<_> = (0..number_of_nodes as u32).collect();
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
                hash_tables.add(&hash_indices, i as u32 + 1);
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
        status: &LayerStatus,
        input_id: usize,
        labels: &[u32],
        sparsity: f32,
    ) -> LayerStatus {
        let active_nodes: Vec<_> = if sparsity == 1.0 {
            (0..self.nodes.len()).collect()
        } else {
            let hashes = self.hasher.get_hash_sparse(
                &status.active_values,
                status.size(),
                &status.active_nodes,
            );
            let hash_indices = self.hash_tables.hashes_to_indices::<H>(&hashes);
            let actives = self.hash_tables.get_raw(&hash_indices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashset
            let mut active_nodes = HashSet::<u32>::new();

            // Make sure that the true label node is in candidates
            if matches!(self.node_type, NodeType::Softmax) {
                for label in labels.iter() {
                    active_nodes.insert(*label);
                }
            }

                // copy sparse array into (dense) map
            for id in actives {
                assert!(id > 0);
                active_nodes.insert(id - 1);
                // *active_nodes.get_mut(&(id - 1)).unwrap() += 1;
            }

            let mut rng = rand::thread_rng();
            let offset = rng.gen::<usize>() % self.nodes.len();
            for i in 0..self.nodes.len() {
                if active_nodes.len() >= 1000 {
                    break;
                }
                let i = (i + offset) % self.nodes.len();
                active_nodes.insert(self.rand_node[i]);
            }

            active_nodes.iter().map(|k| *k as usize).collect()
        };

        let mut active_values = vec![0.0; active_nodes.len()];

        // find activation for all ACTIVE nodes in layer
        for i in 0..active_nodes.len() {
            active_values[i] = self.nodes[active_nodes[i]].get_activation(
                &status.active_nodes,
                &status.active_values,
                status.size(),
                input_id,
            );
        }

        if matches!(self.node_type, NodeType::Softmax) {
            self.normalization_constants[input_id] = 0.0;
            let max_value = active_values.iter().fold(0.0f32, |a, b| a.max(*b));
            for i in 0..active_nodes.len() {
                let real_activation = (active_values[i] - max_value).exp();
                active_values[i] = real_activation;
                self.nodes[active_nodes[i]].set_last_activation(input_id, real_activation);
                self.normalization_constants[input_id] += real_activation;
            }
        }

        LayerStatus {
            active_values,
            active_nodes,
        }
    }
}
