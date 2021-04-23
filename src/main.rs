use slide::densified_wta_hash::DensifiedWtaHash;
use slide::network::{Case, LayerConfig, Network};
use slide::layer::NodeType;

const BATCH_SIZE: usize = 128;
const CASE_PER_REHASH: usize = 6400;
const CASE_PER_REBUILD: usize = 128000;
const STEP_SIZE: usize = 1000;
const TRAIN_FILE: &str = "../Amazon/amazon_train.txt";
const TEST_FILE: &str = "../Amazon/amazon_test.txt";

fn main() {
    let learning_rate = 0.0001;
    let input_size = 135909;

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
            size: 670091,
            node_type: NodeType::Softmax,
            k: 6,
            l: 50,
            range_pow: 18,
            sparsity: 0.005,
        },
    ];

    let start = std::time::Instant::now();
    let mut network =
        Network::<DensifiedWtaHash>::new(BATCH_SIZE, learning_rate, input_size, &layers);
    println!("network built elapsed: {:?}", start.elapsed());

    for epoch in 0..10 {
        println!("epoch {}", epoch);
        train(490449 / BATCH_SIZE, &mut network, epoch);
        test(100, &mut network, (epoch + 1) * (490449 / BATCH_SIZE));
    }
}

fn train(num_batches: usize, network: &mut Network<DensifiedWtaHash>, epoch: usize) {
    use std::io::prelude::*;
    let file = std::fs::File::open(TRAIN_FILE).unwrap();
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines().skip(1);
    for i in 0..num_batches {
        if (i + epoch * num_batches) % STEP_SIZE == 0 {
            test(20, network, epoch * num_batches + i);
        }
        let mut cases = Vec::new();
        while let Some(Ok(str)) = lines.next() {
            let a = str.split(' ').skip(1).map(|s| s.split(':').collect::<Vec<_>>()).collect::<Vec<_>>();
            let b = str.split(' ').next().unwrap().split(',').collect::<Vec<_>>();

            cases.push(Case {
                indices: a.iter().map(|s| s[0].parse::<usize>().unwrap()).collect(),
                values: a.iter().map(|s| s[1].parse::<f32>().unwrap()).collect(),
                labels: b.iter().map(|s| s.parse::<u32>().unwrap()).collect(),
            });
            if cases.len() == BATCH_SIZE {
                break;
            }
        }

        let iter = epoch * num_batches + i;
        let rehash = iter % (CASE_PER_REHASH / BATCH_SIZE) == CASE_PER_REHASH / BATCH_SIZE - 1;
        let rebuild = iter % (CASE_PER_REBUILD / BATCH_SIZE) == CASE_PER_REBUILD / BATCH_SIZE - 1;
        network.train(&cases, iter, rehash, rebuild)
    }
}

fn test(num_batches: usize, network: &mut Network<DensifiedWtaHash>, iter: usize) {
    use std::io::prelude::*;
    let file = std::fs::File::open(TEST_FILE).unwrap();
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines().skip(1);
    for _ in 0..num_batches {
        let mut cases = Vec::new();
        while let Some(Ok(str)) = lines.next() {
            let a = str.split(' ').skip(1).map(|s| s.split(':').collect::<Vec<_>>()).collect::<Vec<_>>();
            let b = str.split(' ').next().unwrap().split(',').collect::<Vec<_>>();

            cases.push(Case {
                indices: a.iter().map(|s| s[0].parse::<usize>().unwrap()).collect(),
                values: a.iter().map(|s| s[1].parse::<f32>().unwrap()).collect(),
                labels: b.iter().map(|s| s.parse::<u32>().unwrap()).collect(),
            });
            if cases.len() == BATCH_SIZE {
                break;
            }
        }

        let correct_pred = network.test(&cases);
        println!("iter {} correct {}/{}", iter, correct_pred, cases.len());
    }
}
