//! Sum tree for prioritized sampling.
//!
//! Code is adapted from <https://github.com/jaromiru/AI-blog/blob/master/SumTree.py> and
use rand::{rngs::StdRng, RngCore};
/// <https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py>
use segment_tree::{
    ops::{MaxIgnoreNaN, MinIgnoreNaN},
    SegmentPoint,
};
use serde::{Deserialize, Serialize};

#[derive(Copy, Debug, Clone, Deserialize, Serialize, PartialEq)]
/// Specifies how to normalize the importance weights in a prioritized batch.
pub enum WeightNormalizer {
    /// Normalize weights by the maximum weight of all samples in the buffer.
    All,
    /// Normalize weights by the maximum weight of samples in the batch.
    Batch,
}

/// A sum tree used for prioritized experience replay.
#[derive(Debug)]
pub struct SumTree {
    eps: f32,
    alpha: f32,
    capacity: usize,
    n_samples: usize,
    tree: Vec<f32>,
    min_tree: SegmentPoint<f32, MinIgnoreNaN>,
    max_tree: SegmentPoint<f32, MaxIgnoreNaN>,
    normalize: WeightNormalizer,
    rng: fastrand::Rng,
}

impl SumTree {
    pub fn new(capacity: usize, alpha: f32, normalize: WeightNormalizer) -> Self {
        Self {
            eps: 1e-8,
            alpha,
            capacity,
            n_samples: 0,
            tree: vec![0f32; 2 * capacity - 1],
            min_tree: SegmentPoint::build(vec![f32::MAX; capacity], MinIgnoreNaN),
            max_tree: SegmentPoint::build(vec![1e-8f32; capacity], MaxIgnoreNaN),
            normalize,
            rng: fastrand::Rng::with_seed(0),
        }
    }

    fn propagate(&mut self, ix: usize, change: f32) {
        let parent = (ix - 1) / 2;
        self.tree[parent] += change;
        if parent != 0 {
            self.propagate(parent, change);
        }
    }

    fn retrieve(&self, ix: usize, s: f32) -> usize {
        let left = 2 * ix + 1;
        let right = left + 1;

        if left >= self.tree.len() {
            return ix;
        }

        if s <= self.tree[left] || self.tree[right] == 0f32 {
            return self.retrieve(left, s);
        } else {
            return self.retrieve(right, s - self.tree[left]);
        }
    }

    pub fn total(&self) -> f32 {
        return self.tree[0];
    }

    pub fn max(&self) -> f32 {
        self.max_tree
            .query(0, self.max_tree.len())
            .powf(1.0 / self.alpha)
    }

    /// Add priority value at `ix`-th element in the sum tree.
    ///
    /// The alpha-th power of the priority value is taken when addition.
    pub fn add(&mut self, ix: usize, p: f32) {
        debug_assert!(ix <= self.n_samples);

        self.update(ix, p);

        if self.n_samples < self.capacity {
            self.n_samples += 1;
        }
    }

    /// Update priority value at `ix`-th element in the sum tree.
    pub fn update(&mut self, ix: usize, p: f32) {
        debug_assert!(ix < self.capacity);

        let p = (p + self.eps).powf(self.alpha);
        self.min_tree.modify(ix, p);
        self.max_tree.modify(ix, p);
        let ix = ix + self.capacity - 1;
        let change = p - self.tree[ix];
        if change.is_nan() {
            println!("{:?}, {:?}", p, self.tree[ix]);
            panic!();
        }
        self.tree[ix] = p;
        self.propagate(ix, change);
    }

    /// Get the maximal index of the sum tree where the sum of priority values is less than `s`.
    pub fn get(&self, s: f32) -> usize {
        let ix = self.retrieve(0, s);
        debug_assert!(ix >= (self.capacity - 1));
        ix + 1 - self.capacity
    }

    /// Samples indices for batch and returns normalized weights.
    ///
    /// The weight is $w_i=\left(N^{-1}P(i)^{-1}\right)^{\beta}$
    /// and it will be normalized by $max_i w_i$.
    pub fn sample(&mut self, batch_size: usize, beta: f32) -> (Vec<i64>, Vec<f32>) {
        let p_sum = &self.total();
        let ps = (0..batch_size)
            .map(|_| p_sum * self.rng.f32())
            .collect::<Vec<_>>();
        let indices = ps.iter().map(|&p| self.get(p)).collect::<Vec<_>>();
        let (ws, w_max_inv) = self.weights(&indices, beta);

        // let n = self.n_samples as f32 / p_sum;
        // let ws = indices
        //     .iter()
        //     .map(|ix| self.tree[ix + self.capacity - 1])
        //     .map(|p| (n * p).powf(-beta))
        //     .collect::<Vec<_>>();

        // // normalizer within all samples
        // let w_max_inv = match self.normalize {
        //     WeightNormalizer::All => (n * self.min_tree.query(0, self.n_samples)).powf(beta),
        //     WeightNormalizer::Batch => 1f32 / ws.iter().fold(0.0 / 0.0, |m, v| v.max(m)),
        // };
        // let ws = ws.iter().map(|w| w * w_max_inv).collect::<Vec<f32>>();

        // debug
        // if self.n_samples % 100 == 0 || p_sum.is_nan() || w_max.is_nan() {
        if p_sum.is_nan() || w_max_inv.is_nan() || ws.iter().sum::<f32>().is_nan() {
            println!("self.n_samples: {:?}", self.n_samples);
            println!("p_sum: {:?}", p_sum);
            println!("w_max_inv: {:?}", w_max_inv);
            println!("ps: {:?}", ps);
            println!("indices: {:?}", indices);
            println!("{:?}", ws);
            panic!();
        }

        let ixs = indices.iter().map(|&ix| ix as i64).collect();

        (ixs, ws)
    }

    #[allow(dead_code)]
    pub fn print_tree(&self) {
        let mut nl = 1;

        for i in 0..self.tree.len() {
            print!("{} ", self.tree[i]);
            if i == 2 * nl - 2 {
                println!();
                nl *= 2;
            }
        }
        println!("max   = {}", self.max());
        // println!("min   = {}", self.min());
        println!("total = {}", self.total());
    }

    fn weights(&self, ixs: &Vec<usize>, beta: f32) -> (Vec<f32>, f32) {
        let n = self.n_samples as f32 / self.total();
        let ws = ixs
            .iter()
            .map(|ix| self.tree[ix + self.capacity - 1])
            .map(|p| (n * p).powf(-beta))
            .collect::<Vec<_>>();

        // normalizer within all samples
        let w_max_inv = match self.normalize {
            WeightNormalizer::All => (n * self.min_tree.query(0, self.n_samples)).powf(beta),
            WeightNormalizer::Batch => 1f32 / ws.iter().fold(0.0 / 0.0, |m, v| v.max(m)),
        };
        let ws = ws.iter().map(|w| w * w_max_inv).collect::<Vec<f32>>();

        (ws, w_max_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::{SumTree, WeightNormalizer::Batch};

    #[test]
    fn test_sum_tree() {
        // 7 samples
        let data = vec![0.5f32, 0.2, 0.8, 0.3, 1.1, 2.5, 3.9];

        // Capacity is 16, alpha is 1
        let mut sum_tree = SumTree::new(16, 1.0, Batch);

        // Check the number of samples in the sum tree
        assert_eq!(sum_tree.n_samples, 0);

        // Push samples
        for ix in 0..data.len() {
            sum_tree.add(ix, data[ix]);
        }
        sum_tree.print_tree();
        println!();

        // Check the importance weights
        println!("Importance weights");
        println!("{:?}", sum_tree.weights(&vec![0, 1, 2, 3, 4], 0.1));
        println!();

        // Check the number of samples in the sum tree
        assert_eq!(sum_tree.n_samples, data.len());

        // Check the weigths
        assert_eq!(sum_tree.get(0.0), 0);
        assert_eq!(sum_tree.get(0.4), 0);
        assert_eq!(sum_tree.get(0.5), 0);
        assert_eq!(sum_tree.get(0.6), 1);
        assert_eq!(sum_tree.get(1.2), 2);
        assert_eq!(sum_tree.get(1.6), 3);
        assert_eq!(sum_tree.get(2.0), 4);
        assert_eq!(sum_tree.get(2.8), 4);
        assert_eq!(sum_tree.get(sum_tree.total()), sum_tree.n_samples - 1);

        // Updates weights of 2nd and 7th samples
        sum_tree.update(1, 3.3);
        sum_tree.update(6, 2.0);
        sum_tree.print_tree();
        println!();

        // Check the weigths after updates
        assert_eq!(sum_tree.get(0.0), 0);
        assert_eq!(sum_tree.get(3.8), 1);
        assert_eq!(sum_tree.get(3.81), 2);
        assert_eq!(sum_tree.get(4.6), 2);
        assert_eq!(sum_tree.get(sum_tree.total()), sum_tree.n_samples - 1);
    }

    #[test]
    fn test_sum_tree_with_alpha() {
        // 7 samples
        let data = vec![0.5f32, 0.2, 0.8, 0.3, 1.1, 2.5, 3.9];

        // Capacity is 16, alpha is 0.5
        let mut sum_tree = SumTree::new(16, 0.5, Batch);

        // Check the number of samples in the sum tree
        assert_eq!(sum_tree.n_samples, 0);

        // Push samples
        for ix in 0..data.len() {
            sum_tree.add(ix, data[ix]);
        }
        sum_tree.print_tree();
        println!();

        // Check the number of samples in the sum tree
        assert_eq!(sum_tree.n_samples, data.len());

        // Check the weigths
        assert_eq!(sum_tree.get(0.0), 0);
        assert_eq!(sum_tree.get(0.6), 0);
        assert_eq!(sum_tree.get(1.1), 1);
        assert_eq!(sum_tree.get(2.0), 2);
        assert_eq!(sum_tree.get(2.5), 3);
        assert_eq!(sum_tree.get(3.5), 4);
        assert_eq!(sum_tree.get(sum_tree.total()), sum_tree.n_samples - 1);
    }

    #[test]
    fn test_sum_tree_sampling() {
        // 5 samples
        let data = vec![0.5f32, 0.2, 0.8, 0.3, 1.1];
        let sum: f32 = data.iter().sum();
        let probs = data.iter().map(|&x| x as f32 / sum as f32).collect::<Vec<_>>();
        println!("Theoretical probabilities: {:?}", probs);

        // Capacity is 16, alpha is 1.0
        let mut sum_tree = SumTree::new(16, 1.0, Batch);

        // Push samples
        for ix in 0..data.len() {
            sum_tree.add(ix, data[ix]);
        }

        // Sampling many times
        let n_samples = data.len();
        let mut n_counts = vec![0; n_samples];
        let batch_size = 32;
        let beta = 1.0;
        let n_sampling_times = 100000;
        for _ in 0..n_sampling_times {
            let (ixs, _) = sum_tree.sample(batch_size, beta);
            ixs.iter().for_each(|&ix| n_counts[ix as usize] += 1);
        }

        // Empirical probability
        let n_total = (n_sampling_times * batch_size) as f32;
        let emps = n_counts.iter().map(|&c| c as f32 / n_total).collect::<Vec<_>>();
        println!("Empirical probabilities: {:?}", emps);

        // Relative precision less than 0.1
        probs.iter().zip(emps.iter()).for_each(|(&p1, &p2)| {
            approx::assert_relative_eq!(p1, p2, max_relative = 0.01);
        })
    }
}
