use crate::hasher::Hasher;

const BIN_SIZE: usize = 8;

pub struct WtaHash {
    size: usize,
    indices: Vec<usize>,
}

impl Hasher for WtaHash {
    fn new(size: usize, number_of_bits: usize) -> Self {
        use rand::seq::SliceRandom;

        assert!(BIN_SIZE <= number_of_bits);

        let mut rng = rand::thread_rng();

        let mut n_array: Vec<usize> = (0..number_of_bits).collect();
        let mut indices = vec![0; size * BIN_SIZE];

        for i in 0..size {
            n_array.shuffle(&mut rng);
            for j in 0..BIN_SIZE {
                indices[i * BIN_SIZE + j] = n_array[j];
            }
        }

        WtaHash { size, indices }
    }

    fn hash(&self, weights: &[f32]) -> Vec<usize> {
        // binsize is the number of times the range is larger than the total number of hashes we need.
        let mut hashes = vec![0; self.size];

        for i in 0..self.size {
            let mut w = f32::MIN;
            for j in 0..BIN_SIZE {
                let k = self.indices[i * BIN_SIZE + j];
                if w < weights[k] {
                    w = weights[k];
                    hashes[i] = j;
                }
            }
        }
        hashes
    }

    fn hash_sparse(&self, weights: &[f32], indices: &[usize]) -> Vec<usize> {
        let mut hashes = vec![0; self.size];

        for i in 0..self.size {
            let mut w = f32::MIN;
            for j in 0..BIN_SIZE {
                let k = self.indices[i * BIN_SIZE + j];
                let weight = indices
                    .iter()
                    .position(|i| k == *i)
                    .map(|i| weights[i])
                    .unwrap_or_default();
                if w < weight {
                    w = weight;
                    hashes[i] = j;
                }
            }
        }
        hashes
    }

    fn hashes_to_indices(hashes: &[usize], k: usize, l: usize, range_pow: usize) -> Vec<usize> {
        let bin_size_log2 = (BIN_SIZE as f32).log2() as usize;
        (0..l)
            .map(|i| {
                let mut index = 0;
                for j in 0..k {
                    let h = hashes[k * i + j];
                    index |= h << (bin_size_log2 * j);
                }
                index & ((1 << range_pow) - 1)
            })
            .collect()
    }
}

#[test]
fn test() {
    let hash = WtaHash::new(4, 8);
    assert_eq!(
        hash.hash(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        hash.hash_sparse(&[], &[])
    );
    assert_eq!(
        hash.hash(&[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        hash.hash_sparse(&[0.5], &[0])
    );
    assert_eq!(
        hash.hash(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        hash.hash_sparse(&[1.0], &[0])
    );
    assert_eq!(
        hash.hash(&[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        hash.hash_sparse(&[0.5], &[1])
    );
    assert_eq!(
        hash.hash(&[0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]),
        hash.hash_sparse(&[0.5, 0.4, 0.3, 0.2], &[1, 3, 5, 7])
    );

    let hashes = hash.hash(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    dbg!(&hashes);
    dbg!(WtaHash::hashes_to_indices(&hashes, 2, 2, 10));
    let hashes = hash.hash(&[0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]);
    dbg!(&hashes);
    dbg!(WtaHash::hashes_to_indices(&hashes, 2, 2, 10));

    let hash = WtaHash::new(100, 50);
    let hashes = hash.hash_sparse(&[1.0], &[0]);
    dbg!(&hashes);
}
