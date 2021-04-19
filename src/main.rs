mod adam;
mod bucket;
// mod desified_min_hash;
mod desified_wta_hash;
mod hasher;
mod layer;
mod lsh;
mod network;
mod node;
// mod sparse_random_projection;
mod train;
// mod wta_hash;

use desified_wta_hash::DesifiedWtaHash;
use network::{Case, Network};
use node::NodeType;

fn main() {
    let number_of_layers = 3;
    let batch_size = 128;
    let learning_rate = 0.0001;
    let input_size = 1359;
    // let input_size = 135909;
    // let sizes_of_layers = [128, 670091];
    let sizes_of_layers = [128, 6700];
    let k = [2, 6];
    let l = [20, 50];
    let range_pow = [6, 18];
    // let sparsity = [1.0, 0.005, 1.0, 1.0];
    let sparsity = [1.0, 0.005, 1.0, 1.0];
    let mut layers_types = vec![NodeType::Relu; number_of_layers];
    layers_types[number_of_layers - 1] = NodeType::Softmax;

    let mut network = Network::<DesifiedWtaHash>::new(
        batch_size,
        learning_rate,
        input_size,
        number_of_layers,
        &sizes_of_layers,
        layers_types,
        &k,
        &l,
        &range_pow,
        &sparsity,
    );
    dbg!("network built");
    network.test(&[Case {
        indices: Vec::new(),
        values: Vec::new(),
        size: 0,
        labels: Vec::new(),
    }]);
    network.train(
        &[Case {
            indices: Vec::new(),
            values: Vec::new(),
            size: 0,
            labels: Vec::new(),
        }],
        0,
        true,
        true,
    );
}
