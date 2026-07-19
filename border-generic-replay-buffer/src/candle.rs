use crate::BatchBase;
use candle_core::{error::Result, Device, IndexOp, Tensor};

/// A buffer consisting of a [`Tensor`].
///
/// The internal buffer is a single contiguous [`Tensor`] of shape
/// `[capacity, shape[1..]]`, where `shape` is obtained from the data pushed at
/// the first time via [`TensorBatch::push`]. `[1..]` means that the first axis
/// of the given data is ignored as it might be the batch size.
///
/// [`Tensor`]: https://docs.rs/candle-core/0.4.1/candle_core/struct.Tensor.html
#[derive(Clone, Debug)]
pub struct TensorBatch {
    buf: Option<Tensor>,
    capacity: usize,
}

impl TensorBatch {
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.dims()[0] as _;
        Self {
            buf: Some(t),
            capacity,
        }
    }

    pub fn to(&mut self, device: &Device) -> Result<()> {
        if let Some(buf) = &self.buf {
            self.buf = Some(buf.to_device(device)?);
        }
        Ok(())
    }
}

impl BatchBase for TensorBatch {
    fn new(capacity: usize) -> Self {
        Self {
            buf: None,
            capacity: capacity,
        }
    }

    /// Pushes given data.
    ///
    /// If the internal buffer is empty, it will be initialized with the shape
    /// `[capacity, data.buf.dims()[1..]]`.
    fn push(&mut self, index: usize, data: Self) {
        if data.buf.is_none() {
            return;
        }

        let batch_size = data.buf.as_ref().unwrap().dims()[0];
        if batch_size == 0 {
            return;
        }

        if self.buf.is_none() {
            let mut shape = data.buf.as_ref().unwrap().dims().to_vec();
            shape[0] = self.capacity;
            let dtype = data.buf.as_ref().unwrap().dtype();
            let device = Device::Cpu;
            self.buf = Some(Tensor::zeros(shape, dtype, &device).unwrap());
        }

        if index + batch_size > self.capacity {
            let batch_size = self.capacity - index;
            let data = &data.buf.unwrap();
            let data1 = data.i((..batch_size,)).unwrap();
            let data2 = data.i((batch_size..,)).unwrap();
            self.buf
                .as_mut()
                .unwrap()
                .slice_set(&data1, 0, index)
                .unwrap();
            self.buf.as_mut().unwrap().slice_set(&data2, 0, 0).unwrap();
        } else {
            self.buf
                .as_mut()
                .unwrap()
                .slice_set(&data.buf.unwrap(), 0, index)
                .unwrap();
        }
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let capacity = ixs.len();
        let ixs = {
            let device = self.buf.as_ref().unwrap().device();
            let ixs = ixs.iter().map(|x| *x as u32).collect();
            Tensor::from_vec(ixs, &[capacity], device).unwrap()
        };
        let buf = Some(self.buf.as_ref().unwrap().index_select(&ixs, 0).unwrap());
        Self { buf, capacity }
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

    fn tensor_from(vals: &[f32], rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(vals.to_vec(), &[rows, cols], &Device::Cpu).unwrap()
    }

    #[test]
    fn from_tensor_and_into_roundtrip() {
        let t = tensor_from(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let batch = TensorBatch::from_tensor(t.clone());
        let back: Tensor = batch.into();
        assert_eq!(
            back.to_vec2::<f32>().unwrap(),
            t.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn push_then_sample() {
        let mut buf = TensorBatch::new(4);
        // push two rows at index 0
        buf.push(0, TensorBatch::from_tensor(tensor_from(&[1.0, 2.0], 2, 1)));
        // push two rows at index 2
        buf.push(2, TensorBatch::from_tensor(tensor_from(&[3.0, 4.0], 2, 1)));

        let sampled: Tensor = buf.sample(&vec![0, 1, 2, 3]).into();
        assert_eq!(
            sampled.to_vec2::<f32>().unwrap(),
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]
        );

        // sample a subset in arbitrary order
        let sampled: Tensor = buf.sample(&vec![3, 0]).into();
        assert_eq!(
            sampled.to_vec2::<f32>().unwrap(),
            vec![vec![4.0], vec![1.0]]
        );
    }

    #[test]
    fn push_wraps_around() {
        let mut buf = TensorBatch::new(3);
        buf.push(0, TensorBatch::from_tensor(tensor_from(&[1.0, 2.0], 2, 1)));
        // push two rows starting at index 2: one at 2, one wraps to 0
        buf.push(2, TensorBatch::from_tensor(tensor_from(&[3.0, 4.0], 2, 1)));

        let sampled: Tensor = buf.sample(&vec![0, 1, 2]).into();
        assert_eq!(
            sampled.to_vec2::<f32>().unwrap(),
            vec![vec![4.0], vec![2.0], vec![3.0]]
        );
    }
}
