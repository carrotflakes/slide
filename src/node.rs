use crate::{adam::Adam, layer::LayerStatus, train::Train};

#[derive(Clone, Copy)]
pub enum NodeType {
    Relu,
    Softmax,
}

pub struct Node {
    // active_inputs: usize,
    node_type: NodeType,

    pub weights: Vec<f32>,
    pub bias: f32,
    pub gradients: Vec<Adam>,
    pub bias_gradient: Adam,
}

impl Node {
    pub fn new(size: usize, node_type: NodeType, weights: Vec<f32>, bias: f32) -> Self {
        assert_eq!(weights.len(), size);
        Self {
            // active_inputs: 0,
            node_type,
            weights,
            bias,
            gradients: vec![Adam::default(); size],
            bias_gradient: Adam::default(),
        }
    }

    pub fn get_size(&self) -> usize {
        self.gradients.len()
    }

    pub fn compute_activation(&self, train: &mut Train, indices: &[usize], values: &[f32]) -> f32 {
        // FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
        if !train.active {
            train.active = true;
            // self.active_inputs += 1;
        }

        train.activation = 0.0;
        for i in 0..indices.len() {
            train.activation += self.weights[indices[i]] * values[i];
        }
        train.activation += self.bias;

        match self.node_type {
            NodeType::Relu => {
                if train.activation < 0.0 {
                    train.activation = 0.0;
                    train.delta_for_bp = 0.0;
                }
            }
            NodeType::Softmax => {}
        }
        train.activation
    }

    pub fn compute_extra_stats_for_softmax(
        &self,
        train: &mut Train,
        normalization_constant: f32,
        label: u32,
        labels: &[u32],
        batch_size: usize,
    ) {
        train.activation /= normalization_constant + 0.0000001;

        // TODO: check gradient
        let expect = if labels.contains(&label) {
            1.0 / labels.len() as f32
        } else {
            0.0
        };
        train.delta_for_bp = (expect - train.activation) / batch_size as f32;
    }

    pub fn back_propagate(
        &mut self,
        delta_for_bp: f32,
        layer_status: &mut LayerStatus,
        _learning_rate: f32,
    ) {
        for id in layer_status.active_nodes.iter().cloned() {
            let prev_train = &mut layer_status.trains[id];
            prev_train.increment_delta(delta_for_bp * self.weights[id]);
            let grad_t = delta_for_bp * prev_train.activation;
            self.gradients[id].update(grad_t);
        }
        self.bias_gradient.update(delta_for_bp);
        // self.active_inputs -= 1;
    }
}
