use crate::hasher::Hasher;

const BIN_SIZE: usize = 8;

pub struct DesifiedWtaHash {
    rand_hash: usize,
    numhashes: usize,
    range_pow: usize,
    lognumhashes: usize,
    indices: Vec<usize>,
    pos: Vec<usize>,
    permute: usize,
}

impl DesifiedWtaHash {
    fn get_rand_double_hash(&self, binid: usize, count: usize) -> usize {
        let tohash = ((binid + 1) << 6) + count;
        (self.rand_hash * tohash << 3) >> (32 - self.lognumhashes) // lognumhash needs to be ceiled.
    }
}

impl Hasher for DesifiedWtaHash {
    fn new(size: usize, number_of_bits_to_hash: usize) -> Self {
        use rand::{seq::SliceRandom, Rng};

        let mut rng = rand::thread_rng();

        let permute =
            (size as f32 * BIN_SIZE as f32 / number_of_bits_to_hash as f32).ceil() as usize;

        let mut n_array: Vec<usize> = (0..number_of_bits_to_hash).collect();
        let mut indices = vec![0; number_of_bits_to_hash * permute];
        let mut pos = vec![0; number_of_bits_to_hash * permute];

        for p in 0..permute {
            n_array.shuffle(&mut rng);
            for i in 0..number_of_bits_to_hash {
                indices[p * number_of_bits_to_hash + n_array[i]] =
                    (p * number_of_bits_to_hash + i) / BIN_SIZE;
                pos[p * number_of_bits_to_hash + n_array[i]] =
                    (p * number_of_bits_to_hash + i) % BIN_SIZE;
            }
        }

        DesifiedWtaHash {
            rand_hash: rng.gen::<usize>() | 1,
            numhashes: size,
            range_pow: number_of_bits_to_hash,
            lognumhashes: (size as f32).log2() as usize,
            indices,
            pos,
            permute,
        }
    }

    fn get_hash(&self, weights: &[f32]) -> Vec<usize> {
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

        let mut hash_array = vec![0; self.numhashes];
        for i in 0..self.numhashes {
            let mut next = hashes[i];
            let mut count = 0;
            while next == usize::MAX {
                next = hashes[self.get_rand_double_hash(i, count).min(self.numhashes)]; // kills GPU.
                count += 1;
                if count > 100 {
                    // Densification failure
                    break;
                }
            }
            hash_array[i] = next;
        }
        hash_array
    }

    fn get_hash_sparse(&self, weights: &[f32], indices: &[usize]) -> Vec<usize> {
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

        let mut hash_array = vec![0; self.numhashes];
        for i in 0..self.numhashes {
            let mut next = hashes[i];
            let mut count = 0;
            while next == usize::MAX {
                next = hashes[self.get_rand_double_hash(i, count).min(self.numhashes)]; // kills GPU.
                count += 1;
                if count > 100 {
                    // Densification failure
                    break;
                }
            }
            hash_array[i] = next;
        }
        hash_array
    }

    fn hashes_to_index(hashes:  &[usize], k: usize, l: usize, _range_pow: usize) -> Vec<usize> {
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
