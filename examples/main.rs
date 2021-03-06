use slide::densified_wta_hash::DensifiedWtaHash;
use slide::network::{Case, LayerConfig, Network};
use slide::layer::NodeType;

fn main() {
    let batch_size = 64;
    let learning_rate = 0.01; //0.0001;
    let input_size = 100;
    let case_per_rehash = 6400;
    let case_per_rebuild = 128000;

    let layers = [
        LayerConfig {
            size: 128,
            node_type: NodeType::Relu,
            k: 2,
            l: 20,
            range_pow: 6,
            sparsity: 1.0,
        },
        // LayerConfig {
        //     size: 1024,
        //     node_type: NodeType::Relu,
        //     k: 2,
        //     l: 20,
        //     range_pow: 8,
        //     sparsity: 0.1,
        // },
        LayerConfig {
            size: 128,
            node_type: NodeType::Softmax,
            k: 4,
            l: 30,
            range_pow: 10,
            sparsity: 0.8,
        },
    ];

    let mut cases = Vec::new();
    for i in 0..1000 {
        let label = i % 10;
        let mut indices = vec![
            rand::random::<usize>() % 5 * 10 + label,
            rand::random::<usize>() % 5 * 10 + label,
            rand::random::<usize>() % 5 * 10 + label,
            rand::random::<usize>() % 5 * 10 + label,
            rand::random::<usize>() % (5 * 10),
        ];
        indices.sort();
        indices.dedup();
        let mut values = vec![];
        values.resize(indices.len(), 1.0);
        cases.push(Case {
            indices,
            values,
            labels: vec![label as u32],
        });
    }
    dbg!(&cases[..3]);

    let start = std::time::Instant::now();
    let mut network =
        Network::<DensifiedWtaHash>::new(batch_size, learning_rate, input_size, &layers);
    println!("network built elapsed: {:?}", start.elapsed());

    for i in 0..1000 {
        // dbg!("train...");
        let rehash = i % (case_per_rehash / batch_size) == case_per_rehash / batch_size - 1;
        let rebuild = i % (case_per_rebuild / batch_size) == case_per_rebuild / batch_size - 1;
        network.train(&cases[i * batch_size % 900..], i, rehash, rebuild);
        // dbg!("train end");

        println!(
            "{:>5}: {:?}",
            i,
            (0..10)
                .map(|i| network.predict(&cases[i]))
                .collect::<Vec<_>>()
        );
    }
}
