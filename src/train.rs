pub struct Train {
    pub last_delta_for_bps: f32,
    pub last_activations: f32,
    pub last_gradients: f32,
    pub active_input_ids: usize,
}

impl Train {
    pub fn new() -> Self {
        Self {
            last_delta_for_bps: 0.0,
            last_activations: 0.0,
            last_gradients: 0.0,
            active_input_ids: 0,
        }
    }
}
