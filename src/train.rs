pub struct Train {
    pub last_delta_for_bp: f32,
    pub last_activation: f32,
    pub active: bool,
}

impl Train {
    pub fn new() -> Self {
        Self {
            last_delta_for_bp: 0.0,
            last_activation: 0.0,
            active: false,
        }
    }

    pub fn increment_delta(&mut self, increment_value: f32) {
        assert!(self.active);
        if self.last_activation > 0.0 {
            self.last_delta_for_bp += increment_value;
        }
    }
}
