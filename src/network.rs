use crate::{
    adam::{BETA1, BETA2},
    hasher::Hasher,
    layer::{Layer, LayerStatus, TrainContext},
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
    train_contexts: Vec<Vec<TrainContext>>,
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
                    layer_configs
                        .iter()
                        .map(|config| TrainContext {
                            nodes: (0..config.size).map(|_| Train::new()).collect(),
                            normalization_constant: 0.0,
                        })
                        .collect()
                })
                .collect(),
        }
    }

    pub fn predict(&mut self, case: &Case, input_id: usize) -> usize {
        let train_context = &mut self.train_contexts[input_id];
        let mut layers = Vec::new();
        layers.push(LayerStatus {
            active_nodes: case.indices.clone(),
            active_values: case.values.clone(),
        });
        // inference
        for j in 0..self.number_of_layers {
            let state = self.hidden_layers[j].query_active_node_and_compute_activations(
                &mut train_context[j],
                &layers[j],
                &[],
                1.0,
            );
            layers.push(state);
        }

        // compute softmax
        let mut max_act = f32::NEG_INFINITY;
        let mut predict_class = 0;
        for j in 0..layers[self.number_of_layers].size() {
            let class = layers[self.number_of_layers].active_nodes[j];
            let act = train_context[self.number_of_layers - 1].nodes[class].activation;
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
            let Case {
                indices,
                values,
                labels,
            } = &cases[i % cases.len()];
            let mut layers = Vec::new();
            layers.push(LayerStatus {
                active_nodes: indices.clone(),
                active_values: values.clone(),
            });
            let train_context = &mut self.train_contexts[i];

            // inference
            for j in 0..self.number_of_layers {
                let sparsity = self.hidden_layers[j].sparsity;
                let state = self.hidden_layers[j].query_active_node_and_compute_activations(
                    &mut train_context[j],
                    &layers[j],
                    labels,
                    sparsity,
                );
                layers.push(state);
            }

            // backpropagate
            for j in (0..self.number_of_layers).rev() {
                for id in layers[j + 1].active_nodes.iter().cloned() {
                    let node = &mut self.hidden_layers[j].nodes[id];
                    assert!(train_context[j].nodes[id].active);
                    if j == self.number_of_layers - 1 {
                        //TODO: Compute Extra stats: labels[i];
                        let normalization_constant = train_context[j].normalization_constant;
                        node.compute_extra_stats_for_softmax(
                            &mut train_context[j].nodes[id],
                            normalization_constant,
                            id as u32,
                            labels,
                            batch_size,
                        );
                    }
                    if j > 0 {
                        node.back_propagate(
                            train_context[j].nodes[id].delta_for_bp,
                            &mut train_context[j - 1].nodes,
                            &layers[j].active_nodes,
                            learning_rate,
                        );
                    } else {
                        node.back_propagate_first_layer(
                            train_context[j].nodes[id].delta_for_bp,
                            &indices,
                            &values,
                            learning_rate,
                        );
                    }
                    train_context[j].nodes[id] = Train::new();
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
