pub const BETA2: f32 = 0.999;
pub const BETA1: f32 = 0.9;
const EPS: f32 = 0.00000001;

#[derive(Default, Clone)]
pub struct Adam {
    avg_mom: f32,
    avg_vel: f32,
    t: f32,
}

impl Adam {
    pub fn update(&mut self, dt: f32) {
        self.t += dt;
    }

    pub fn gradient(&mut self) -> f32 {
        self.avg_mom = BETA1 * self.avg_mom + (1.0 - BETA1) * self.t;
        self.avg_vel = BETA2 * self.avg_vel + (1.0 - BETA1) * self.t.powi(2);
        self.t = 0.0;
        self.avg_mom / (self.avg_vel.sqrt() + EPS)
    }
}
