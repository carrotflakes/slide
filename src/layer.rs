use std::collections::HashSet;

use rand::{seq::SliceRandom, Rng};

use crate::{
    hasher::Hasher,
    lsh::Lsh,
    node::{Node, NodeType},
};

#[derive(Default)]
pub struct LayerStatus {
    pub active_nodes: Vec<usize>,
    pub active_values: Vec<f32>,
    pub deltas: Vec<f32>,
    pub normalization_constant: f32,
}

impl LayerStatus {
    pub fn from_input(indices: &[usize], values: &[f32]) -> Self {
        LayerStatus {
            active_nodes: indices.to_vec(),
            active_values: values.to_vec(),
            deltas: vec![0.0; indices.len()],
            normalization_constant: 0.0,
        }
    }

    pub fn size(&self) -> usize {
        self.active_nodes.len()
    }
}

pub struct Layer<H: Hasher> {
    node_type: NodeType,
    pub nodes: Vec<Node>,
    rand_ids: Vec<u32>,
    k: usize,
    l: usize,
    previous_layer_num_of_nodes: usize,
    pub sparsity: f32,
    hasher: H,
    hash_tables: Lsh,
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
    ) -> Self {
        let mut rand_ids: Vec<_> = (0..number_of_nodes as u32).collect();
        let mut rng = rand::thread_rng();
        rand_ids.shuffle(&mut rng);

        let mut nodes = Vec::with_capacity(number_of_nodes);
        for _ in 0..number_of_nodes {
            let mut weights = vec![0.0; previous_layer_num_of_nodes];
            weights.fill_with(|| rng.gen_range(0.0..0.01));
            let bias = rng.gen_range(0.0..0.01);

            nodes.push(Node::new(previous_layer_num_of_nodes, weights, bias));
        }

        let hasher = H::new(k * l, previous_layer_num_of_nodes);
        let hash_tables = Lsh::new(k, l, range_pow);

        let mut layer = Self {
            node_type,
            nodes,
            rand_ids,
            k,
            l,
            previous_layer_num_of_nodes,
            hasher,
            hash_tables,
            sparsity,
        };

        layer.rehash();

        layer
    }

    pub fn update_table(&mut self) {
        self.hasher = H::new(self.k * self.l, self.previous_layer_num_of_nodes);
    }

    pub fn rehash(&mut self) {
        self.hash_tables.clear();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let hashes = self.hasher.hash(&node.weights);
            let hash_indices = self.hash_tables.hashes_to_indices::<H>(&hashes);
            self.hash_tables.add(&hash_indices, i as u32);
        }
    }

    pub fn random_nodes(&mut self) {
        self.rand_ids.shuffle(&mut rand::thread_rng());
    }

    pub fn query_active_node_and_compute_activations(
        &self,
        layer_statuses: &mut [LayerStatus],
        labels: &[u32],
        sparsity: f32,
    ) {
        let mut it = layer_statuses.iter_mut();
        let LayerStatus {
            ref active_nodes,
            ref active_values,
            ..
        } = it.next().unwrap();
        let layer_status = it.next().unwrap();

        layer_status.active_nodes = if sparsity == 1.0 {
            (0..self.nodes.len()).collect()
        } else {
            // TODO: implement Modes

            let hashes = self.hasher.hash_sparse(&active_values, &active_nodes);
            let hash_indices = self.hash_tables.hashes_to_indices::<H>(&hashes);
            let actives = self.hash_tables.get_ids(&hash_indices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashset
            let mut active_nodes = HashSet::<u32>::new();

            // Make sure that the true label node is in candidates
            if matches!(self.node_type, NodeType::Softmax) {
                for label in labels.iter() {
                    active_nodes.insert(*label);
                }
            }

            for id in actives {
                active_nodes.insert(id);
            }

            let offset = rand::random::<usize>() % self.nodes.len();
            for i in 0..self.nodes.len() {
                if active_nodes.len() >= 1000 {
                    break;
                }
                let i = (i + offset) % self.nodes.len();
                active_nodes.insert(self.rand_ids[i]);
            }

            active_nodes.iter().map(|v| *v as usize).collect()
        };

        layer_status.active_values.clear();
        for id in layer_status.active_nodes.iter().cloned() {
            layer_status
                .active_values
                .push(self.nodes[id].compute_value(&active_nodes, &active_values));
        }

        layer_status.deltas.clear();
        layer_status
            .deltas
            .resize(layer_status.active_nodes.len(), 0.0);

        // apply activation function
        match self.node_type {
            NodeType::Relu => {
                for value in layer_status.active_values.iter_mut() {
                    *value = value.max(0.0);
                }
            }
            NodeType::Softmax => {
                layer_status.normalization_constant = 0.0;
                let max_value = layer_status
                    .active_values
                    .iter()
                    .fold(0.0f32, |a, b| a.max(*b));
                for i in 0..layer_status.active_nodes.len() {
                    let value = (layer_status.active_values[i] - max_value).exp();
                    layer_status.active_values[i] = value;
                    layer_status.normalization_constant += value;
                }
            }
        }
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        for i in 0..self.nodes.len() {
            let mut node = &mut self.nodes[i];
            for j in 0..node.get_size() {
                node.weights[j] += learning_rate * node.gradients[j].gradient();
            }
            node.bias += learning_rate * node.bias_gradient.gradient();
        }
    }
}
