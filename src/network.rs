use rayon::prelude::*;

use crate::{
    adam::{BETA1, BETA2},
    hasher::Hasher,
    layer::{Layer, LayerStatus, NodeType},
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

    pub fn predict(&mut self, case: &Case) -> usize {
        let layer_statuses = &mut self.train_statuses[0];
        layer_statuses[0] = LayerStatus::from_input(&case.indices, &case.values);
        // inference
        for j in 0..self.number_of_layers {
            self.hidden_layers[j].query_active_node_and_compute_activations(
                &mut layer_statuses[j..j + 2],
                &[],
                1.0,
            );
        }

        // compute top-1
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
        let hidden_layers = &self.hidden_layers;
        let number_of_layers = self.number_of_layers;
        self.train_statuses
            .par_iter_mut()
            .enumerate()
            .map(|(i, layer_statuses)| {
                let case = &cases[i];
                layer_statuses[0] = LayerStatus::from_input(&case.indices, &case.values);
                // inference
                for j in 0..number_of_layers {
                    hidden_layers[j].query_active_node_and_compute_activations(
                        &mut layer_statuses[j..j + 2],
                        &[],
                        1.0,
                    );
                }

                // compute top-1
                let mut max_act = f32::NEG_INFINITY;
                let mut predict_class = 0;
                let last_layer = &layer_statuses[number_of_layers];
                for j in 0..last_layer.size() {
                    let act = last_layer.active_values[j];
                    if max_act < act {
                        max_act = act;
                        predict_class = last_layer.active_nodes[j];
                    }
                }
                if case.labels.contains(&(predict_class as u32)) {
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn train(&mut self, cases: &[Case], iter: usize, rehash: bool, rebuild: bool) {
        let batch_size = self.train_statuses.len().min(cases.len());
        if iter % 6946 == 6945 {
            self.hidden_layers[1].random_nodes();
        }

        // let start = std::time::Instant::now();
        let hidden_layers = &self.hidden_layers;
        let number_of_layers = self.number_of_layers;
        self.train_statuses
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, layer_statuses)| {
                let case = &cases[i];
                layer_statuses[0] = LayerStatus::from_input(&case.indices, &case.values);

                // inference
                for j in 0..number_of_layers {
                    let sparsity = hidden_layers[j].sparsity;
                    let force_activate_nodes = if j == number_of_layers - 1 {
                        case.labels.as_slice()
                    } else {
                        &[]
                    };
                    hidden_layers[j].query_active_node_and_compute_activations(
                        &mut layer_statuses[j..j + 2],
                        force_activate_nodes,
                        sparsity,
                    );
                }

                // compute loss
                for k in 0..layer_statuses[number_of_layers].active_nodes.len() {
                    let id = layer_statuses[number_of_layers].active_nodes[k];
                    let activation = layer_statuses[number_of_layers].active_values[k] + 0.0000001;

                    // TODO: check gradient
                    let expect = if case.labels.contains(&(id as u32)) {
                        1.0 / case.labels.len() as f32
                    } else {
                        0.0
                    };
                    layer_statuses[number_of_layers].deltas[k] =
                        (expect - activation) / batch_size as f32;
                }

                // backpropagate
                for j in (0..number_of_layers).rev() {
                    #[allow(mutable_transmutes)]
                    let layer =
                        unsafe { std::mem::transmute::<_, &mut Layer<H>>(&hidden_layers[j]) };
                    layer.back_propagate(&mut layer_statuses[j..j + 2]);
                }
            });
        // print!("step1: {:?}", start.elapsed());

        let learning_rate = self.learning_rate * (1.0 - BETA2.powi(iter as i32 + 1)).sqrt()
            / (1.0 - BETA1.powi(iter as i32 + 1));

        // update weights
        // let start = std::time::Instant::now();
        for layer in &mut self.hidden_layers {
            layer.update_weights(learning_rate);
            if rebuild && layer.sparsity < 1.0 {
                layer.update_table();
            }
            if rehash && layer.sparsity < 1.0 {
                layer.rehash();
            }
        }
        // println!(", step2: {:?}", start.elapsed());
    }
}
