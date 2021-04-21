use crate::{
    adam::{BETA1, BETA2},
    hasher::Hasher,
    layer::{Layer, LayerStatus},
    node::NodeType,
    train::Train,
};

pub struct LayerConfig {
    pub size: usize,
    pub node_type: NodeType,
    pub k: usize,
    pub l: usize,
    pub range_pow: usize,
    pub sparsity: f32,
}

#[derive(Debug)]
pub struct Case {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
    pub labels: Vec<u32>,
}

pub struct Network<H: Hasher> {
    hidden_layers: Vec<Layer<H>>,
    number_of_layers: usize,
    train_contexts: Vec<Vec<LayerStatus>>,
    learning_rate: f32,
}

impl<H: Hasher> Network<H> {
    pub fn new(
        batch_size: usize,
        learning_rate: f32,
        input_size: usize,
        layer_configs: &[LayerConfig],
    ) -> Self {
        let mut hidden_layers = Vec::with_capacity(layer_configs.len());
        let mut previous_layer_size = input_size;
        for config in layer_configs {
            hidden_layers.push(Layer::new(
                config.size,
                previous_layer_size,
                config.node_type,
                config.k,
                config.l,
                config.range_pow,
                config.sparsity,
            ));
            previous_layer_size = config.size;
        }
        Network {
            hidden_layers,
            learning_rate,
            number_of_layers: layer_configs.len(),
            train_contexts: (0..batch_size)
                .map(|_| {
                    let mut v = vec![LayerStatus {
                        active_nodes: Vec::new(),
                        active_values: Vec::new(),
                        trains: (0..input_size).map(|_| Train::new_actived()).collect(),
                        normalization_constant: 0.0,
                    }];
                    for config in layer_configs.iter() {
                        v.push(LayerStatus {
                            active_nodes: Vec::new(),
                            active_values: Vec::new(),
                            trains: (0..config.size).map(|_| Train::new()).collect(),
                            normalization_constant: 0.0,
                        });
                    }
                    v
                })
                .collect(),
        }
    }

    pub fn predict(&mut self, case: &Case, input_id: usize) -> usize {
        let layer_statuses = &mut self.train_contexts[input_id];
        layer_statuses[0].active_nodes = case.indices.clone();
        layer_statuses[0].active_values = case.values.clone();
        // inference
        for j in 0..self.number_of_layers {
            self.hidden_layers[j].query_active_node_and_compute_activations(
                &mut layer_statuses[j..j + 2],
                &[],
                1.0,
            );
        }

        // compute softmax
        let mut max_act = f32::NEG_INFINITY;
        let mut predict_class = 0;
        let last_layer = &layer_statuses[self.number_of_layers];
        for j in 0..last_layer.size() {
            let class = last_layer.active_nodes[j];
            let act = last_layer.trains[class].activation;
            if max_act < act {
                max_act = act;
                predict_class = class;
            }
        }
        predict_class
    }

    pub fn test(&mut self, cases: &[Case]) -> usize {
        let mut correct_pred = 0;
        for i in 0..cases.len().min(self.train_contexts.len()) {
            let predict_class = self.predict(&cases[i], i);
            if cases[i].labels.contains(&(predict_class as u32)) {
                correct_pred += 1;
            }
        }
        correct_pred
    }

    pub fn train(&mut self, cases: &[Case], iter: usize, rehash: bool, rebuild: bool) {
        let batch_size = self.train_contexts.len();
        if iter % 6946 == 6945 {
            self.hidden_layers[1].random_nodes();
        }
        let learning_rate = self.learning_rate * (1.0 - BETA2.powi(iter as i32 + 1)).sqrt()
            / (1.0 - BETA1.powi(iter as i32 + 1));
        for i in 0..batch_size {
            let case = &cases[i % cases.len()];
            let layer_statuses = &mut self.train_contexts[i];
            layer_statuses[0].active_nodes = case.indices.clone();
            layer_statuses[0].active_values = case.values.clone();

            // inference
            for j in 0..self.number_of_layers {
                let sparsity = self.hidden_layers[j].sparsity;
                self.hidden_layers[j].query_active_node_and_compute_activations(
                    &mut layer_statuses[j..j + 2],
                    &case.labels,
                    sparsity,
                );
            }

            // backpropagate
            for j in (0..self.number_of_layers).rev() {
                for id in layer_statuses[j + 1].active_nodes.clone() {
                    let node = &mut self.hidden_layers[j].nodes[id];
                    assert!(layer_statuses[j + 1].trains[id].active);
                    if j == self.number_of_layers - 1 {
                        //TODO: Compute Extra stats: labels[i];
                        let normalization_constant = layer_statuses[j + 1].normalization_constant;
                        node.compute_extra_stats_for_softmax(
                            &mut layer_statuses[j + 1].trains[id],
                            normalization_constant,
                            id as u32,
                            &case.labels,
                            batch_size,
                        );
                    }
                    node.back_propagate(
                        layer_statuses[j + 1].trains[id].delta_for_bp,
                        &mut layer_statuses[j],
                        learning_rate,
                    );
                    layer_statuses[j + 1].trains[id] = Train::new();
                }
            }
        }

        // update weights
        for layer in &mut self.hidden_layers {
            let rehash = rehash && layer.sparsity < 1.0;
            let rebuild = rebuild && layer.sparsity < 1.0;
            if rehash {
                layer.hash_tables.clear();
            }
            if rebuild {
                layer.update_table();
            }
            for i in 0..layer.nodes.len() {
                let mut node = &mut layer.nodes[i];
                for j in 0..node.get_size() {
                    node.weights[j] += learning_rate * node.gradients[j].gradient();
                }
                node.bias += learning_rate * node.bias_gradient.gradient();

                if rehash {
                    let hashes = layer.hasher.hash(&node.weights);
                    let hash_indices = layer.hash_tables.hashes_to_indices::<H>(&hashes);
                    layer.hash_tables.add(&hash_indices, i as u32 + 1);
                }
            }
        }
    }
}
