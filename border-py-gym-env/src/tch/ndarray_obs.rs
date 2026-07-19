use super::TensorBatch;
use ndarray::ArrayD;
use std::convert::TryFrom;
use tch::Tensor;

#[derive(Clone, Debug)]
/// Observation.
pub struct NdarrayObs(pub ArrayD<f32>);

impl border_core::Obs for NdarrayObs {
    fn len(&self) -> usize {
        self.0.shape()[0]
    }
}

impl Into<Tensor> for NdarrayObs {
    fn into(self) -> Tensor {
        Tensor::try_from(&self.0).unwrap()
    }
}

impl From<NdarrayObs> for TensorBatch {
    fn from(o: NdarrayObs) -> Self {
        TensorBatch::from_tensor(o.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    // Regression test for the tch 0.24 bump: `NdarrayObs` must convert into a
    // `tch::Tensor` preserving shape, dtype and values. This fails to compile
    // when border's ndarray version does not match the one tch links against.
    #[test]
    fn ndarray_obs_into_tensor_preserves_shape_and_values() {
        let arr =
            ArrayD::<f32>::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();

        let t: Tensor = NdarrayObs(arr).into();

        assert_eq!(t.size(), vec![2, 3]);
        assert_eq!(t.kind(), tch::Kind::Float);
        let v = Vec::<f32>::try_from(&t.flatten(0, -1)).unwrap();
        assert_eq!(v, vec![1., 2., 3., 4., 5., 6.]);
    }
}
