use crate::adam::Adam;

pub struct Param {
    pub value: f32,
    error: f32,
    adam: Adam,
}

impl Param {
    pub fn new(value: f32) -> Self {
        Param {
            value,
            error: 0.0,
            adam: Adam::default(),
        }
    }

    pub fn add_error(&mut self, value: f32) {
        self.error += value;
    }

    pub fn update(&mut self, rate: f32) {
        self.value += rate * self.adam.apply(self.error);
        self.error = 0.0;
    }
}
