use slide::densified_wta_hash::DensifiedWtaHash;
use slide::network::{Case, LayerConfig, Network};
use slide::layer::NodeType;

fn main() {
    let batch_size = 8; //128;
    let learning_rate = 0.001; //0.0001;
    let input_size = 1359;

    let layers = [
        LayerConfig {
            size: 128,
            node_type: NodeType::Relu,
            k: 2,
            l: 20,
            range_pow: 6,
            sparsity: 1.0,
        },
        LayerConfig {
            size: 1000, //670091,
            node_type: NodeType::Softmax,
            k: 6,
            l: 50,
            range_pow: 18,
            sparsity: 0.005,
        },
    ];

    let start = std::time::Instant::now();
    let mut network =
        Network::<DensifiedWtaHash>::new(batch_size, learning_rate, input_size, &layers);
    println!("network built elapsed: {:?}", start.elapsed());

    dbg!(network.predict(
        &Case {
            indices: vec![1],
            values: vec![1.0],
            labels: vec![1],
        },
        0
    ));

    for i in 0..10 {
        dbg!("train...");
        network.train(
            &[Case {
                indices: vec![1],
                values: vec![1.0],
                labels: vec![1],
            }],
            0,
            false,
            false,
        );
        dbg!("train end");

        dbg!(network.predict(
            &Case {
                indices: vec![1],
                values: vec![1.0],
                labels: vec![1],
            },
            0
        ));
    }
}
