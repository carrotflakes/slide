use crate::{
    adam::{BETA1, BETA2},
    hasher::Hasher,
    layer::{Layer, LayerStatus},
    node::NodeType,
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
    train_statuses: Vec<Vec<LayerStatus>>,
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
            train_statuses: (0..batch_size)
                .map(|_| {
                    let mut v = Vec::new();
                    v.resize_with(layer_configs.len() + 1, Default::default);
                    v
                })
                .collect(),
        }
    }

    pub fn predict(&mut self, case: &Case, input_id: usize) -> usize {
        let layer_statuses = &mut self.train_statuses[input_id];
        layer_statuses[0] = LayerStatus::from_input(&case.indices, &case.values);
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
            let act = last_layer.active_values[j];
            if max_act < act {
                max_act = act;
                predict_class = last_layer.active_nodes[j];
            }
        }
        predict_class
    }

    pub fn test(&mut self, cases: &[Case]) -> usize {
        let mut correct_pred = 0;
        for i in 0..cases.len().min(self.train_statuses.len()) {
            let predict_class = self.predict(&cases[i], i);
            if cases[i].labels.contains(&(predict_class as u32)) {
                correct_pred += 1;
            }
        }
        correct_pred
    }

    pub fn train(&mut self, cases: &[Case], iter: usize, rehash: bool, rebuild: bool) {
        let batch_size = self.train_statuses.len().min(cases.len());
        if iter % 6946 == 6945 {
            self.hidden_layers[1].random_nodes();
        }
        let learning_rate = self.learning_rate * (1.0 - BETA2.powi(iter as i32 + 1)).sqrt()
            / (1.0 - BETA1.powi(iter as i32 + 1));

        for i in 0..batch_size {
            let case = &cases[i];
            let layer_statuses = &mut self.train_statuses[i];
            layer_statuses[0] = LayerStatus::from_input(&case.indices, &case.values);

            // inference
            for j in 0..self.number_of_layers {
                let sparsity = self.hidden_layers[j].sparsity;
                let labels = if j == self.number_of_layers - 1 {
                    case.labels.as_slice()
                } else {
                    &[]
                };
                self.hidden_layers[j].query_active_node_and_compute_activations(
                    &mut layer_statuses[j..j + 2],
                    labels,
                    sparsity,
                );
            }

            // backpropagate
            for j in (0..self.number_of_layers).rev() {
                for k in 0..layer_statuses[j + 1].active_nodes.len() {
                    let id = layer_statuses[j + 1].active_nodes[k];
                    if j == self.number_of_layers - 1 {
                        //TODO: Compute Extra stats: labels[i];
                        let normalization_constant = layer_statuses[j + 1].normalization_constant;

                        let activation = layer_statuses[j + 1].active_values[k]
                            / normalization_constant
                            + 0.0000001;

                        // TODO: check gradient
                        let expect = if case.labels.contains(&(id as u32)) {
                            1.0 / case.labels.len() as f32
                        } else {
                            0.0
                        };
                        layer_statuses[j + 1].deltas[k] = (expect - activation) / batch_size as f32;
                    }
                    self.hidden_layers[j].nodes[id].back_propagate(
                        layer_statuses[j + 1].deltas[k],
                        &mut layer_statuses[j],
                        learning_rate,
                    );
                }
            }
        }

        // update weights
        for layer in &mut self.hidden_layers {
            layer.update_weights(learning_rate);
            if rebuild && layer.sparsity < 1.0 {
                layer.update_table();
            }
            if rehash && layer.sparsity < 1.0 {
                layer.rehash();
            }
        }
    }
}
