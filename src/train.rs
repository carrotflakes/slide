pub struct Train {
    pub delta_for_bp: f32,
    pub activation: f32,
    pub active: bool,
}

impl Train {
    pub fn new() -> Self {
        Self {
            delta_for_bp: 0.0,
            activation: 0.0,
            active: false,
        }
    }

    pub fn new_actived() -> Self {
        Self {
            delta_for_bp: 0.0,
            activation: 1.0,
            active: true,
        }
    }

    pub fn increment_delta(&mut self, value: f32) {
        assert!(self.active);
        if self.activation > 0.0 {
            self.delta_for_bp += value;
        }
    }

    pub fn compute_extra_stats_for_softmax(
        &mut self,
        normalization_constant: f32,
        label: u32,
        labels: &[u32],
        batch_size: usize,
    ) {
        self.activation /= normalization_constant + 0.0000001;

        // TODO: check gradient
        let expect = if labels.contains(&label) {
            1.0 / labels.len() as f32
        } else {
            0.0
        };
        self.delta_for_bp = (expect - self.activation) / batch_size as f32;
    }
}
