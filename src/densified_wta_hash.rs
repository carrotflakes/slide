use crate::hasher::Hasher;

const BIN_SIZE: usize = 8;

pub struct DensifiedWtaHash {
    rand_hash: usize,
    numhashes: usize,
    range_pow: usize,
    lognumhashes: usize,
    indices: Vec<usize>,
    pos: Vec<usize>,
    permute: usize,
}

impl DensifiedWtaHash {
    fn rand_double_hash(&self, binid: usize, count: usize) -> usize {
        let tohash = ((binid + 1) << 6) + count;
        (self.rand_hash * tohash << 3) >> (32 - self.lognumhashes) // lognumhash needs to be ceiled.
    }

    fn densify(&self, hashes: Vec<usize>) -> Vec<usize> {
        let mut densified_hashes = Vec::with_capacity(self.numhashes);
        for i in 0..self.numhashes {
            let mut hash = hashes[i];
            let mut count = 0;
            while hash == usize::MAX {
                hash = hashes[self.rand_double_hash(i, count).min(self.numhashes - 1)];
                count += 1;
                if count > 100 {
                    // Densification failure
                    println!("Densification failure");
                    hash = 0; // Work around...
                    break;
                }
            }
            densified_hashes.push(hash);
        }
        densified_hashes
    }
}

impl Hasher for DensifiedWtaHash {
    fn new(size: usize, number_of_bits: usize) -> Self {
        use rand::{seq::SliceRandom, Rng};

        let mut rng = rand::thread_rng();

        let permute = (size as f32 * BIN_SIZE as f32 / number_of_bits as f32).ceil() as usize;

        let mut n_array: Vec<usize> = (0..number_of_bits).collect();
        let mut indices = vec![0; number_of_bits * permute];
        let mut pos = vec![0; number_of_bits * permute];

        for p in 0..permute {
            n_array.shuffle(&mut rng);
            for i in 0..number_of_bits {
                indices[p * number_of_bits + n_array[i]] = (p * number_of_bits + i) / BIN_SIZE;
                pos[p * number_of_bits + n_array[i]] = (p * number_of_bits + i) % BIN_SIZE;
            }
        }

        DensifiedWtaHash {
            rand_hash: rng.gen::<usize>() | 1,
            numhashes: size,
            range_pow: number_of_bits,
            lognumhashes: (size as f32).log2() as usize,
            indices,
            pos,
            permute,
        }
    }

    fn hash(&self, weights: &[f32]) -> Vec<usize> {
        // binsize is the number of times the range is larger than the total number of hashes we need.
        let mut hashes = vec![usize::MAX; self.numhashes];
        let mut values = vec![f32::MIN; self.numhashes];

        for p in 0..self.permute {
            let bin_index = p * self.range_pow;
            for i in 0..weights.len() {
                let index = bin_index + i;
                let binid = self.indices[index];
                let weight = weights[i];
                if binid < self.numhashes && values[binid] < weight {
                    values[binid] = weight;
                    hashes[binid] = self.pos[index];
                }
            }
        }

        self.densify(hashes)
    }

    fn hash_sparse(&self, weights: &[f32], indices: &[usize]) -> Vec<usize> {
        let mut hashes = vec![usize::MAX; self.numhashes];
        let mut values = vec![f32::MIN; self.numhashes];

        for p in 0..self.permute {
            let bin_index = p * self.range_pow;
            for i in 0..weights.len() {
                let index = bin_index + indices[i];
                let binid = self.indices[index];
                let weight = weights[i];
                if binid < self.numhashes && values[binid] < weight {
                    values[binid] = weight;
                    hashes[binid] = self.pos[index];
                }
            }
        }

        self.densify(hashes)
    }

    fn hashes_to_indices(hashes: &[usize], k: usize, l: usize, _range_pow: usize) -> Vec<usize> {
        (0..l)
            .map(|i| {
                let mut index = 0;
                for j in 0..k {
                    let h = hashes[k * i + j];
                    index += h << ((k - 1 - j) * ((BIN_SIZE as f32).ln().floor() as usize));
                }
                index
            })
            .collect()
    }
}
