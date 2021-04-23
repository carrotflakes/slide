pub const BETA2: f32 = 0.999;
pub const BETA1: f32 = 0.9;
const EPS: f32 = 0.00000001;

#[derive(Default)]
pub struct Adam {
    avg_mom: f32,
    avg_vel: f32,
}

impl Adam {
    pub fn apply(&mut self, error: f32) -> f32 {
        self.avg_mom = BETA1 * self.avg_mom + (1.0 - BETA1) * error;
        self.avg_vel = BETA2 * self.avg_vel + (1.0 - BETA2) * error.powi(2);
        self.avg_mom / (self.avg_vel.sqrt() + EPS)
    }
}
