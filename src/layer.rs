use std::collections::HashSet;

use rand::{seq::SliceRandom, Rng};

use crate::{hasher::Hasher, lsh::Lsh, node::Node, param::Param};

#[derive(Clone, Copy)]
pub enum NodeType {
    Relu,
    Softmax,
    OriginalSoftmax,
}

#[derive(Default)]
pub struct LayerStatus {
    pub active_nodes: Vec<usize>,
    pub active_values: Vec<f32>,
    pub deltas: Vec<f32>,
}

impl LayerStatus {
    pub fn from_input(indices: &[usize], values: &[f32]) -> Self {
        LayerStatus {
            active_nodes: indices.to_vec(),
            active_values: values.to_vec(),
            deltas: vec![0.0; indices.len()],
        }
    }

    pub fn size(&self) -> usize {
        self.active_nodes.len()
    }
}

pub struct Layer<H: Hasher> {
    node_type: NodeType,
    nodes: Vec<Node>,
    rand_ids: Vec<u32>,
    k: usize,
    l: usize,
    previous_layer_num_of_nodes: usize,
    pub sparsity: f32,
    hasher: H,
    hash_tables: Lsh,
    min_active_nodes: usize,
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
            let mut weights = Vec::with_capacity(previous_layer_num_of_nodes);
            weights.resize_with(previous_layer_num_of_nodes, || {
                Param::new(rng.gen_range(0.0..0.01))
            });
            let bias = Param::new(rng.gen_range(0.0..0.01));

            nodes.push(Node::new(weights, bias));
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
            min_active_nodes: 1000,
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
            let hashes = self
                .hasher
                .hash(&node.weights.iter().map(|w| w.value).collect::<Vec<_>>());
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
        force_activate_nodes: &[u32],
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
            active_nodes.extend(force_activate_nodes);
            active_nodes.extend(actives);

            let offset = rand::random::<usize>() % self.nodes.len();
            for i in 0..self.nodes.len() {
                if active_nodes.len() >= self.min_active_nodes {
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
        self.activate(&mut layer_status.active_values);

        layer_status.deltas.clear();
        layer_status
            .deltas
            .resize(layer_status.active_nodes.len(), 0.0);
    }

    pub fn back_propagate(&mut self, layer_statuses: &mut [LayerStatus]) {
        let mut it = layer_statuses.iter_mut();
        let prev_layer_status = it.next().unwrap();
        let layer_status = it.next().unwrap();
        for i in 0..layer_status.size() {
            let id = layer_status.active_nodes[i];
            let value = layer_status.active_values[i];
            let delta = layer_status.deltas[i];
            let delta = match self.node_type {
                NodeType::Relu => {
                    if value > 0.0 {
                        delta
                    } else {
                        0.0
                    }
                }
                NodeType::Softmax => delta,
                NodeType::OriginalSoftmax => delta,
            };
            self.nodes[id].back_propagate(delta, prev_layer_status);
        }
    }

    pub fn update_weights(&mut self, rate: f32) {
        for node in &mut self.nodes {
            for j in 0..node.get_size() {
                node.weights[j].update(rate);
            }
            node.bias.update(rate);
        }
    }

    fn activate(&self, values: &mut [f32]) {
        match self.node_type {
            NodeType::Relu => {
                for value in values {
                    *value = value.max(0.0);
                }
            }
            NodeType::Softmax => {
                let sum_value: f32 = values.iter().map(|v| v.exp()).sum();
                for value in values {
                    *value = value.exp() / sum_value;
                }
            }
            NodeType::OriginalSoftmax => {
                let max_value = values.iter().fold(0.0f32, |a, b| a.max(*b));
                for value in values {
                    *value = (*value - max_value).exp();
                }
            }
        }
    }
}
