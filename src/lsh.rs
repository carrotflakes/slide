use crate::{bucket::Bucket, hasher::Hasher};

pub struct Lsh {
    bucket: Vec<Vec<Bucket>>,
    k: usize,
    l: usize,
    range_pow: usize,
    // rand1: Vec<usize>,
}

impl Lsh {
    pub fn new(k: usize, l: usize, range_pow: usize) -> Self {
        let mut bucket = Vec::with_capacity(l);
        for i in 0..l {
            bucket.push(Vec::with_capacity(1 << range_pow));
            for _ in 0..1<<range_pow {
                bucket[i].push(Bucket::new());
            }
        }

        // let mut rand1 = vec![0; k * l];
        // for i in 0..k*l {
        //     rand1[i] = rand::random::<usize>() | 1;
        // }

        Self {
            bucket,
            k,
            l,
            range_pow,
            // rand1,
        }
    }

    pub fn clear(&mut self) {
        for i in 0..self.l {
            self.bucket[i] = Vec::with_capacity(1 << self.range_pow);
            for _ in 0..1<<self.range_pow {
                self.bucket[i].push(Bucket::new());
            }
        }
    }

    pub fn hashes_to_indices<H: Hasher>(&self, hashes: &[usize]) -> Vec<usize> {
        H::hashes_to_index(hashes, self.k, self.l, self.range_pow)
    }

    pub fn add(&mut self, indices: &[usize], id: usize) -> Vec<usize> {
        let mut second_indices = Vec::with_capacity(self.l);
        for i in 0..self.l {
            second_indices.push(self.bucket[i][indices[i]].add(id));
        }
        second_indices
    }

    pub fn get_raw(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        (0..self.l).map(|i| self.bucket[i][indices[i]].get_all().to_vec()).collect() // TODO: many clone!!!!!
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
