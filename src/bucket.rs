pub const BUCKET_SIZE: usize = 128;
const FIFO: bool = true;

pub struct Bucket {
    arr: [u32; BUCKET_SIZE],
    count: usize,
    next_index: usize,
}

impl Bucket {
    pub fn new() -> Self {
        Self {
            arr: [0; BUCKET_SIZE],
            count: 0,
            next_index: 0,
        }
    }

    pub fn get_size(&self) -> usize {
        self.count
    }

    pub fn add(&mut self, id: u32) -> usize {
        assert!(id > 0);
        self.count += 1;
        // FIFO
        if FIFO {
            let index = (self.count - 1) & (BUCKET_SIZE - 1);
            self.arr[index] = id;
            index
        } else {
            // Reservoir Sampling
            if self.next_index == BUCKET_SIZE {
                let randnum = rand::random::<usize>() % self.count + 1;
                if randnum == 2 {
                    let randidx = rand::random::<usize>() % BUCKET_SIZE;
                    self.arr[randidx] = id;
                    randidx
                } else {
                    usize::MAX
                }
            } else {
                self.arr[self.next_index] = id;
                let index = self.next_index;
                self.next_index += 1;
                index
            }
        }
    }

    pub fn get_all(&self) -> &[u32] {
        &self.arr[..self.count]
    }
}
