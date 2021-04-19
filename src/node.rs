use crate::{adam::Adam, train::Train};

#[derive(Clone, Copy)]
pub enum NodeType {
    Relu,
    Softmax,
}
pub struct Node {
    active_inputs: usize,
    node_type: NodeType,

    train: Vec<Train>,
    current_batch_size: usize,
    node_id: usize,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub gradients: Vec<Adam>,
    pub bias_gradient: Adam,
}

impl Node {
    pub fn new(
        size: usize,
        node_id: usize,
        node_type: NodeType,
        batch_size: usize,
        weights: Vec<f32>,
        bias: f32,
    ) -> Self {
        assert_eq!(weights.len(), size);
        Self {
            active_inputs: 0,
            node_type,
            train: (0..batch_size).map(|_| Train::new()).collect(),
            current_batch_size: batch_size,
            node_id,
            weights,
            bias,
            gradients: vec![Default::default(); size],
            bias_gradient: Default::default(),
        }
    }

    pub fn get_size(&self) -> usize {
        self.gradients.len()
    }

    pub fn get_last_activation(&self, input_id: usize) -> f32 {
        if self.train[input_id].active_input_ids != 1 {
            0.0
        } else {
            self.train[input_id].last_activations
        }
    }

    pub fn increment_delta(&mut self, input_id: usize, increment_value: f32) {
        assert!(self.train[input_id].active_input_ids == 1);
        if self.train[input_id].last_activations > 0.0 {
            self.train[input_id].last_delta_for_bps += increment_value;
        }
    }

    pub fn get_activation(
        &mut self,
        indices: &[usize],
        values: &[f32],
        length: usize,
        input_id: usize,
    ) -> f32 {
        assert!(input_id <= self.current_batch_size);

        let mut train = &mut self.train[input_id];

        // FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
        if train.active_input_ids != 1 {
            train.active_input_ids = 1;
            self.active_inputs += 1;
        }

        train.last_activations = 0.0;
        for i in 0..length {
            train.last_activations += self.weights[indices[i]] * values[i];
        }
        train.last_activations += self.bias;

        match self.node_type {
            NodeType::Relu => {
                if train.last_activations < 0.0 {
                    train.last_activations = 0.0;
                    train.last_gradients = 1.0;
                    train.last_delta_for_bps = 0.0;
                } else {
                    train.last_gradients = 0.0;
                }
            }
            NodeType::Softmax => {}
        }
        train.last_activations
    }

    pub fn compute_extra_stats_for_softmax(
        &mut self,
        normalization_constant: f32,
        input_id: usize,
        label: &[usize],
    ) {
        let mut train = &mut self.train[input_id];
        assert!(train.active_input_ids == 1);

        train.last_activations /= normalization_constant + 0.0000001;

        // TODO: check gradient
        train.last_gradients = 1.0;
        train.last_delta_for_bps = if label.contains(&self.node_id) {
            (1.0 / label.len() as f32 - train.last_activations) / self.current_batch_size as f32
        } else {
            -train.last_activations / self.current_batch_size as f32
        };
    }

    pub fn back_propagate(
        &mut self,
        previous_nodes: &mut [Node],
        previous_layer_active_node_ids: &[usize],
        _learning_rate: f32,
        input_id: usize,
    ) {
        let mut train = &mut self.train[input_id];
        assert!(train.active_input_ids == 1);

        for id in previous_layer_active_node_ids.iter().cloned() {
            let prev_node = &mut previous_nodes[id];
            prev_node.increment_delta(input_id, train.last_delta_for_bps * self.weights[id]);
            let grad_t = train.last_delta_for_bps * prev_node.get_last_activation(input_id);
            self.gradients[id].update(grad_t);
        }
        self.bias_gradient.update(train.last_delta_for_bps);

        train.active_input_ids = 0;
        train.last_delta_for_bps = 0.0;
        train.last_activations = 0.0;
        self.active_inputs -= 1;
    }

    pub fn back_propagate_first_layer(
        &mut self,
        nnz_indices: &[usize],
        nnz_values: &[f32],
        _learning_rate: f32,
        input_id: usize,
    ) {
        let mut train = &mut self.train[input_id];
        assert!(train.active_input_ids == 1);

        for i in 0..nnz_indices.len() {
            let grad_t = train.last_delta_for_bps * nnz_values[i];
            self.gradients[nnz_indices[i]].update(grad_t);
        }
        self.bias_gradient.update(train.last_delta_for_bps);

        train.active_input_ids = 0;
        train.last_delta_for_bps = 0.0;
        train.last_activations = 0.0;
        self.active_inputs -= 1;
    }

    pub fn set_last_activation(&mut self, input_id: usize, real_activation: f32) {
        self.train[input_id].last_activations = real_activation;
    }
}
