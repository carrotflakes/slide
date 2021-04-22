use crate::{adam::Adam, layer::LayerStatus};

#[derive(Clone, Copy)]
pub enum NodeType {
    Relu,
    Softmax,
}

pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub gradients: Vec<Adam>,
    pub bias_gradient: Adam,
}

impl Node {
    pub fn new(size: usize, weights: Vec<f32>, bias: f32) -> Self {
        assert_eq!(weights.len(), size);
        Self {
            weights,
            bias,
            gradients: vec![Adam::default(); size],
            bias_gradient: Adam::default(),
        }
    }

    pub fn get_size(&self) -> usize {
        self.weights.len()
    }

    pub fn compute_value(&self, indices: &[usize], values: &[f32]) -> f32 {
        let mut value = 0.0;
        for i in 0..indices.len() {
            value += self.weights[indices[i]] * values[i];
        }
        value + self.bias
    }

    pub fn back_propagate(
        &mut self,
        delta: f32,
        layer_status: &mut LayerStatus,
        _learning_rate: f32,
    ) {
        for i in 0..layer_status.active_nodes.len() {
            let id = layer_status.active_nodes[i];
            let value = layer_status.active_values[i];
            if value > 0.0 {
                layer_status.deltas[i] += delta * self.weights[id];
            }
            let grad_t = delta * value;
            self.gradients[id].update(grad_t);
        }
        self.bias_gradient.update(delta);
    }
}
