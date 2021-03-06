use crate::{bucket::Bucket, hasher::Hasher};

pub struct Lsh {
    bucket: Vec<Vec<Bucket>>,
    k: usize,
    l: usize,
    range_pow: usize,
}

impl Lsh {
    pub fn new(k: usize, l: usize, range_pow: usize) -> Self {
        let mut bucket = Vec::with_capacity(l);
        for i in 0..l {
            bucket.push(Vec::with_capacity(1 << range_pow));
            for _ in 0..1 << range_pow {
                bucket[i].push(Bucket::new());
            }
        }

        Self {
            bucket,
            k,
            l,
            range_pow,
        }
    }

    pub fn clear(&mut self) {
        for buckets in &mut self.bucket {
            for bucket in buckets {
                bucket.clear();
            }
        }
    }

    pub fn hashes_to_indices<H: Hasher>(&self, hashes: &[usize]) -> Vec<usize> {
        H::hashes_to_indices(hashes, self.k, self.l, self.range_pow)
    }

    pub fn add(&mut self, indices: &[usize], id: u32) {
        for i in 0..self.l {
            self.bucket[i][indices[i]].add(id);
        }
    }

    pub fn get_ids(&self, indices: &[usize]) -> Vec<u32> {
        (0..self.l)
            .flat_map(|i| self.bucket[i][indices[i]].get_all())
            .cloned()
            .collect()
    }

    #[allow(dead_code)]
    pub fn print_count(&self) {
        for i in 0..self.l {
            let mut total = 0;
            for j in 0..1 << self.range_pow {
                let size = self.bucket[i][j].get_size();
                if size > 0 {
                    print!("{} ", size);
                    total += size;
                }
            }
            println!();
            println!("TABLE {} Total {}", i, total);
        }
    }
}
