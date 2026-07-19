use crate::BatchBase;
use ::tch::{Device, Tensor};

/// Adds capability of constructing [`Tensor`] with a static method.
///
/// [`Tensor`]: https://docs.rs/tch/0.24.0/tch/struct.Tensor.html
pub trait ZeroTensor {
    /// Constructs zero tensor.
    fn zeros(shape: &[i64]) -> Tensor;
}

impl ZeroTensor for u8 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(shape, (::tch::kind::Kind::Uint8, Device::Cpu))
    }
}

impl ZeroTensor for i32 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(shape, (::tch::kind::Kind::Int, Device::Cpu))
    }
}

impl ZeroTensor for f32 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(shape, ::tch::kind::FLOAT_CPU)
    }
}

impl ZeroTensor for i64 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(shape, (::tch::kind::Kind::Int64, Device::Cpu))
    }
}

/// A buffer consisting of a [`Tensor`].
///
/// The internal buffer of this struct has the shape of `[n_capacity, shape[1..]]`,
/// where `shape` is obtained from the data pushed at the first time via
/// [`TensorBatch::push`] method. `[1..]` means that the first axis of the
/// given data is ignored as it might be batch size.
///
/// [`Tensor`]: https://docs.rs/tch/0.24.0/tch/struct.Tensor.html
pub struct TensorBatch {
    buf: Option<Tensor>,
    capacity: i64,
}

impl Clone for TensorBatch {
    fn clone(&self) -> Self {
        let buf = match self.buf.is_none() {
            true => None,
            false => Some(self.buf.as_ref().unwrap().copy()),
        };

        Self {
            buf,
            capacity: self.capacity,
        }
    }
}

impl TensorBatch {
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.size()[0] as _;
        Self {
            buf: Some(t),
            capacity,
        }
    }
}

impl BatchBase for TensorBatch {
    fn new(capacity: usize) -> Self {
        Self {
            buf: None,
            capacity: capacity as _,
        }
    }

    /// Pushes given data.
    ///
    /// If the internal buffer is empty, it will be initialized with the shape
    /// `[capacity, data.buf.size()[1..]]`.
    fn push(&mut self, index: usize, data: Self) {
        if data.buf.is_none() {
            return;
        }

        let batch_size = data.buf.as_ref().unwrap().size()[0];
        if batch_size == 0 {
            return;
        }

        if self.buf.is_none() {
            let mut shape = data.buf.as_ref().unwrap().size().clone();
            shape[0] = self.capacity;
            let kind = data.buf.as_ref().unwrap().kind();
            let device = Device::Cpu;
            self.buf = Some(Tensor::zeros(&shape, (kind, device)));
        }

        let index = index as i64;
        let val: Tensor = data.buf.as_ref().unwrap().copy();

        for i_ in 0..batch_size {
            let i = (i_ + index) % self.capacity;
            self.buf.as_ref().unwrap().get(i).copy_(&val.get(i_));
        }
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let ixs = ixs.iter().map(|&ix| ix as i64).collect::<Vec<_>>();
        let batch_indexes = Tensor::from_slice(&ixs);
        let buf = Some(self.buf.as_ref().unwrap().index_select(0, &batch_indexes));
        Self {
            buf,
            capacity: ixs.len() as i64,
        }
    }
}

impl From<TensorBatch> for Tensor {
    fn from(b: TensorBatch) -> Self {
        b.buf.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BatchBase;

    fn col_tensor(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals).reshape(&[vals.len() as i64, 1])
    }

    #[test]
    fn from_tensor_and_into_roundtrip() {
        let t = col_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let batch = TensorBatch::from_tensor(t.copy());
        let back: Tensor = batch.into();
        assert_eq!(Vec::<f32>::try_from(back.flatten(0, 1)).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn push_then_sample() {
        let mut buf = TensorBatch::new(4);
        buf.push(0, TensorBatch::from_tensor(col_tensor(&[1.0, 2.0])));
        buf.push(2, TensorBatch::from_tensor(col_tensor(&[3.0, 4.0])));

        let sampled: Tensor = buf.sample(&vec![0, 1, 2, 3]).into();
        assert_eq!(
            Vec::<f32>::try_from(sampled.flatten(0, 1)).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );

        let sampled: Tensor = buf.sample(&vec![3, 0]).into();
        assert_eq!(
            Vec::<f32>::try_from(sampled.flatten(0, 1)).unwrap(),
            vec![4.0, 1.0]
        );
    }

    #[test]
    fn push_wraps_around() {
        let mut buf = TensorBatch::new(3);
        buf.push(0, TensorBatch::from_tensor(col_tensor(&[1.0, 2.0])));
        // push two rows starting at index 2: one at 2, one wraps to 0
        buf.push(2, TensorBatch::from_tensor(col_tensor(&[3.0, 4.0])));

        let sampled: Tensor = buf.sample(&vec![0, 1, 2]).into();
        assert_eq!(
            Vec::<f32>::try_from(sampled.flatten(0, 1)).unwrap(),
            vec![4.0, 2.0, 3.0]
        );
    }

    #[test]
    fn clone_is_deep() {
        let mut buf = TensorBatch::new(2);
        buf.push(0, TensorBatch::from_tensor(col_tensor(&[1.0, 2.0])));
        let cloned = buf.clone();
        // Mutate original in place; clone must be unaffected.
        buf.push(0, TensorBatch::from_tensor(col_tensor(&[9.0, 9.0])));

        let orig: Tensor = buf.sample(&vec![0, 1]).into();
        let cln: Tensor = cloned.sample(&vec![0, 1]).into();
        assert_eq!(Vec::<f32>::try_from(orig.flatten(0, 1)).unwrap(), vec![9.0, 9.0]);
        assert_eq!(Vec::<f32>::try_from(cln.flatten(0, 1)).unwrap(), vec![1.0, 2.0]);
    }
}
