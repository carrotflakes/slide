pub trait Hasher {
    fn new(size: usize, number_of_bits_to_hash: usize) -> Self;
    fn get_hash(&self, weights: &[f32], length: usize) -> Vec<usize>;
    fn get_hash_sparse(&self, weights: &[f32], length: usize, indices: &[usize]) -> Vec<usize>;
    fn hashes_to_index(hashes: &[usize], k: usize, l: usize, range_pow: usize) -> Vec<usize>;
}
