pub trait Hasher {
    fn new(size: usize, number_of_bits_to_hash: usize) -> Self;
    fn hash(&self, weights: &[f32]) -> Vec<usize>;
    fn hash_sparse(&self, weights: &[f32], indices: &[usize]) -> Vec<usize>;
    fn hashes_to_indices(hashes: &[usize], k: usize, l: usize, range_pow: usize) -> Vec<usize>;
}
