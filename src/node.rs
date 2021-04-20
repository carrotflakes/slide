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
            node_id,
            weights,
            bias,
            gradients: vec![Adam::default(); size],
            bias_gradient: Adam::default(),
        }
    }

    pub fn get_size(&self) -> usize {
        self.gradients.len()
    }

    pub fn get_last_activation(&self, input_id: usize) -> f32 {
        if !self.train[input_id].active {
            0.0
        } else {
            self.train[input_id].last_activation
        }
    }

    pub fn set_last_activation(&mut self, input_id: usize, real_activation: f32) {
        self.train[input_id].last_activation = real_activation;
    }

    pub fn get_activation(
        &mut self,
        indices: &[usize],
        values: &[f32],
        length: usize,
        input_id: usize,
    ) -> f32 {
        // assert!(input_id <= self.batch_size);

        let mut train = &mut self.train[input_id];

        // FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
        if !train.active {
            train.active = true;
            self.active_inputs += 1;
        }

        train.last_activation = 0.0;
        for i in 0..length {
            train.last_activation += self.weights[indices[i]] * values[i];
        }
        train.last_activation += self.bias;

        match self.node_type {
            NodeType::Relu => {
                if train.last_activation < 0.0 {
                    train.last_activation = 0.0;
                    train.last_delta_for_bp = 0.0;
                }
            }
            NodeType::Softmax => {}
        }
        train.last_activation
    }

    pub fn compute_extra_stats_for_softmax(
        &mut self,
        normalization_constant: f32,
        input_id: usize,
        labels: &[u32],
        batch_size: usize,
    ) {
        let mut train = &mut self.train[input_id];
        assert!(train.active);

        train.last_activation /= normalization_constant + 0.0000001;

        // TODO: check gradient
        let expect = if labels.contains(&(self.node_id as u32)) {
            1.0 / labels.len() as f32
        } else {
            0.0
        };
        train.last_delta_for_bp = (expect - train.last_activation) / batch_size as f32;
    }

    pub fn back_propagate(
        &mut self,
        previous_nodes: &mut [Node],
        previous_layer_active_node_ids: &[usize],
        _learning_rate: f32,
        input_id: usize,
    ) {
        let train = &mut self.train[input_id];
        assert!(train.active);

        for id in previous_layer_active_node_ids.iter().cloned() {
            let prev_node = &mut previous_nodes[id];
            prev_node.train[input_id].increment_delta(train.last_delta_for_bp * self.weights[id]);
            let grad_t = train.last_delta_for_bp * prev_node.get_last_activation(input_id);
            self.gradients[id].update(grad_t);
        }
        self.bias_gradient.update(train.last_delta_for_bp);

        *train = Train::new();
        self.active_inputs -= 1;
    }

    pub fn back_propagate_first_layer(
        &mut self,
        nnz_indices: &[usize],
        nnz_values: &[f32],
        _learning_rate: f32,
        input_id: usize,
    ) {
        let train = &mut self.train[input_id];
        assert!(train.active);

        for i in 0..nnz_indices.len() {
            let grad_t = train.last_delta_for_bp * nnz_values[i];
            self.gradients[nnz_indices[i]].update(grad_t);
        }
        self.bias_gradient.update(train.last_delta_for_bp);

        *train = Train::new();
        self.active_inputs -= 1;
    }
}
