use std::collections::HashSet;

use rand::{seq::SliceRandom, Rng};

use crate::{
    hasher::Hasher,
    lsh::Lsh,
    node::{Node, NodeType},
    train::Train,
};

pub struct LayerStatus {
    pub active_nodes: Vec<usize>,
    pub active_values: Vec<f32>,
    pub trains: Vec<Train>,
    pub normalization_constant: f32,
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
    ) -> Self {
        let mut rand_node: Vec<_> = (0..number_of_nodes as u32).collect();
        let mut rng = rand::thread_rng();
        rand_node.shuffle(&mut rng);

        let mut nodes = Vec::with_capacity(number_of_nodes);
        for _ in 0..number_of_nodes {
            let mut weights = vec![0.0; previous_layer_num_of_nodes];
            weights.fill_with(|| rng.gen_range(0.0..0.01));
            let bias = rng.gen_range(0.0..0.01);

            nodes.push(Node::new(
                previous_layer_num_of_nodes,
                node_type,
                weights,
                bias,
            ));
        }

        let hasher = H::new(k * l, previous_layer_num_of_nodes);
        let mut hash_tables = Lsh::new(k, l, range_pow);

        // add to hash table
        for (i, node) in nodes.iter_mut().enumerate() {
            let hashes = hasher.hash(&node.weights);
            let hash_indices = hash_tables.hashes_to_indices::<H>(&hashes);
            hash_tables.add(&hash_indices, i as u32 + 1);
        }

        Self {
            node_type,
            nodes,
            rand_node,
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
        &self,
        layer_statuses: &mut [LayerStatus],
        labels: &[u32],
        sparsity: f32,
    ) {
        let mut it = layer_statuses.iter_mut();
        let LayerStatus {
            active_nodes,
            active_values,
            ..
        } = it.next().unwrap();
        let layer_status = it.next().unwrap();

        layer_status.active_nodes = if sparsity == 1.0 {
            (0..self.nodes.len()).collect()
        } else {
            // TODO: implement Modes

            let hashes = self.hasher.hash_sparse(&active_values, &active_nodes);
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

            for id in actives {
                assert!(id > 0);
                active_nodes.insert(id - 1);
            }

            let offset = rand::random::<usize>() % self.nodes.len();
            for i in 0..self.nodes.len() {
                if active_nodes.len() >= 1000 {
                    break;
                }
                let i = (i + offset) % self.nodes.len();
                active_nodes.insert(self.rand_node[i]);
            }

            active_nodes.iter().map(|v| *v as usize).collect()
        };

        layer_status.active_values = Vec::with_capacity(layer_status.active_nodes.len());
        for i in layer_status.active_nodes.iter().cloned() {
            layer_status
                .active_values
                .push(self.nodes[i].compute_activation(
                    &mut layer_status.trains[i],
                    &active_nodes,
                    &active_values,
                ));
        }

        if matches!(self.node_type, NodeType::Softmax) {
            // softmax
            layer_status.normalization_constant = 0.0;
            let max_value = layer_status
                .active_values
                .iter()
                .fold(0.0f32, |a, b| a.max(*b));
            for i in 0..layer_status.active_nodes.len() {
                let real_activation = (layer_status.active_values[i] - max_value).exp();
                layer_status.active_values[i] = real_activation;
                layer_status.trains[layer_status.active_nodes[i]].activation = real_activation;
                layer_status.normalization_constant += real_activation;
            }
        }
    }
}
