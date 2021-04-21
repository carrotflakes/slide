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

    pub fn increment_delta(&mut self, value: f32) {
        assert!(self.active);
        if self.activation > 0.0 {
            self.delta_for_bp += value;
        }
    }
}
