pub(crate) struct ElasticMap {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) alpha: f32,
    pub(super) x_field: Vec<f32>,
    pub(super) y_field: Vec<f32>,
}

impl ElasticMap {
    fn displacement(&self, p: [f64; 2]) -> [f64; 2] {
        let x = p[0].clamp(0.0, (self.width - 1) as f64);
        let y = p[1].clamp(0.0, (self.height - 1) as f64);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        [&self.x_field, &self.y_field].map(|field| {
            let a = f64::from(self.alpha * field[y0 * self.width + x0]);
            let b = f64::from(self.alpha * field[y0 * self.width + x1]);
            let c = f64::from(self.alpha * field[y1 * self.width + x0]);
            let d = f64::from(self.alpha * field[y1 * self.width + x1]);
            (a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy
        })
    }
}

impl super::remap::SpatialRemap for ElasticMap {
    fn source(&self, p: [f64; 2]) -> [f64; 2] {
        let index = p[1] as usize * self.width + p[0] as usize;
        [
            p[0] + f64::from(self.alpha * self.x_field[index]),
            p[1] + f64::from(self.alpha * self.y_field[index]),
        ]
    }

    fn destination(&self, p: [f64; 2]) -> [f64; 2] {
        if self.is_identity() {
            return p;
        }
        let mut q = p;
        for _ in 0..32 {
            let [dx, dy] = self.displacement(q);
            let next = [p[0] - dx, p[1] - dy];
            if (next[0] - q[0]).abs().max((next[1] - q[1]).abs()) <= 1e-6 {
                return next;
            }
            q = next;
        }
        // Strong fields need not be invertible. Choose the nearest sampled
        // source coordinate, then break ties by distance to the input point.
        let radius = f64::from(self.alpha) + 1.0;
        let bounds = |v: f64, n: usize| {
            ((v - radius).floor() as usize).min(n - 1)..=((v + radius).ceil() as usize).min(n - 1)
        };
        let mut best = (f64::INFINITY, f64::INFINITY, p);
        for y in bounds(p[1], self.height) {
            for x in bounds(p[0], self.width) {
                let q = [x as f64, y as f64];
                let s = self.source(q);
                let residual = (s[0] - p[0]).powi(2) + (s[1] - p[1]).powi(2);
                let travel = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
                if (residual, travel) < (best.0, best.1) {
                    best = (residual, travel, q);
                }
            }
        }
        best.2
    }

    fn is_identity(&self) -> bool {
        self.alpha == 0.0
    }
}
