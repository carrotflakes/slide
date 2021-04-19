use crate::{
    adam::{BETA1, BETA2},
    hasher::Hasher,
    layer::{InferContext, Layer},
    node::NodeType,
};

pub struct Case {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
    pub size: usize,
    pub labels: Vec<usize>,
}

pub struct Network<H: Hasher> {
    hidden_layers: Vec<Layer<H>>,
    number_of_layers: usize,
    current_batch_size: usize,
    learning_rate: f32,
}

impl<H: Hasher> Network<H> {
    pub fn new(
        batch_size: usize,
        learning_rate: f32,
        input_size: usize,
        number_of_layers: usize,
        sizes_of_layers: &[usize],
        layers_types: Vec<NodeType>,
        k: &[usize],
        l: &[usize],
        range_pow: &[usize],
        sparsity: &[f32],
        // arr: Vec<f32>,
    ) -> Self {
        let mut hidden_layers = Vec::with_capacity(number_of_layers);
        for i in 0..number_of_layers {
            hidden_layers.push(Layer::new(
                sizes_of_layers[i],
                if i != 0 {
                    sizes_of_layers[i - 1]
                } else {
                    input_size
                },
                layers_types[i],
                batch_size,
                k[i],
                l[i],
                range_pow[i],
                sparsity[i],
            ));
        }
        Network {
            hidden_layers,
            learning_rate,
            number_of_layers,
            current_batch_size: batch_size,
        }
    }

    pub fn test(&mut self, cases: &[Case]) -> usize {
        let mut correct_pred = 0;
        for i in 0..self.current_batch_size {
            let Case {
                indices,
                values,
                size,
                labels,
            } = &cases[i];
            let mut ctx = InferContext::new(self.number_of_layers + 1);
            ctx.active_nodes_per_layer[0] = indices.clone();
            ctx.active_values_per_layer[0] = values.clone();
            ctx.sizes[0] = *size;

            // inference
            for j in 0..self.number_of_layers {
                self.hidden_layers[j].query_active_node_and_compute_activations(
                    &mut ctx,
                    j,
                    i,
                    &labels[..0],
                    1.0,
                );
            }

            // compute softmax
            let mut max_act = f32::NEG_INFINITY;
            let mut predict_class = 0;
            for j in 0..ctx.sizes[self.number_of_layers] {
                let class = ctx.active_nodes_per_layer[self.number_of_layers][j];
                let act = self.hidden_layers[self.number_of_layers - 1].nodes[class]
                    .get_last_activation(i);
                if max_act < act {
                    max_act = act;
                    predict_class = class;
                }
            }

            if labels.contains(&predict_class) {
                correct_pred += 1;
            }
        }
        correct_pred
    }

    pub fn train(&mut self, cases: &[Case], iter: usize, rehash: bool, rebuild: bool) {
        if iter % 6946 == 6945 {
            self.hidden_layers[1].random_nodes();
        }
        let learning_rate = self.learning_rate * (1.0 - BETA2.powi(iter as i32 + 1)).sqrt()
            / (1.0 - BETA1.powi(iter as i32 + 1));
        for i in 0..self.current_batch_size {
            let Case {
                indices,
                values,
                size,
                labels,
            } = &cases[i];
            let mut ctx = InferContext::new(self.number_of_layers + 1);
            ctx.active_nodes_per_layer[0] = indices.clone();
            ctx.active_values_per_layer[0] = values.clone();
            ctx.sizes[0] = *size;

            // inference
            for j in 0..self.number_of_layers {
                let sparsity = self.hidden_layers[j].sparsity;
                self.hidden_layers[j]
                    .query_active_node_and_compute_activations(&mut ctx, j, i, labels, sparsity);
            }

            // backpropagate
            for j in (0..self.number_of_layers).rev() {
                for k in 0..ctx.sizes[j + 1] {
                    if j == self.number_of_layers - 1 {
                        //TODO: Compute Extra stats: labels[i];
                        let layer = &mut self.hidden_layers[j];
                        let node = &mut layer.nodes[ctx.active_nodes_per_layer[j + 1][k]];
                        node.compute_extra_stats_for_softmax(
                            layer.normalization_constants[i],
                            i,
                            labels,
                        );
                    }
                    if j != 0 {
                        let mut it = self.hidden_layers.iter_mut().skip(j);
                        let prev_layer = it.next().unwrap();
                        let layer = it.next().unwrap();
                        let node = &mut layer.nodes[ctx.active_nodes_per_layer[j + 1][k]];
                        node.back_propagate(
                            &mut prev_layer.nodes,
                            &ctx.active_nodes_per_layer[j],
                            learning_rate,
                            i,
                        );
                    } else {
                        let node =
                            &mut self.hidden_layers[0].nodes[ctx.active_nodes_per_layer[j + 1][k]];
                        node.back_propagate_first_layer(&indices, &values, learning_rate, i);
                    }
                }
            }
        }

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
                    let hashes = layer.hasher.get_hash(&node.weights, node.get_size());
                    let hash_indices = layer.hash_tables.hashes_to_indices::<H>(&hashes);
                    layer.hash_tables.add(&hash_indices, i + 1);
                }
            }
        }
    }
}
