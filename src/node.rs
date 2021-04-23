use crate::{layer::LayerStatus, param::Param};

pub struct Node {
    pub weights: Vec<Param>,
    pub bias: Param,
}

impl Node {
    pub fn new(weights: Vec<Param>, bias: Param) -> Self {
        Self { weights, bias }
    }

    pub fn get_size(&self) -> usize {
        self.weights.len()
    }

    pub fn compute_value(&self, indices: &[usize], values: &[f32]) -> f32 {
        let mut value = 0.0;
        for i in 0..indices.len() {
            value += self.weights[indices[i]].value * values[i];
        }
        value + self.bias.value
    }

    pub fn back_propagate(&mut self, delta: f32, prev_layer_status: &mut LayerStatus) {
        for i in 0..prev_layer_status.active_nodes.len() {
            let id = prev_layer_status.active_nodes[i];
            let value = prev_layer_status.active_values[i];
            prev_layer_status.deltas[i] += delta * self.weights[id].value;
            self.weights[id].add_error(delta * value);
        }
        self.bias.add_error(delta);
    }
}
