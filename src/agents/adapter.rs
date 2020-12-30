use tch::Tensor;
use crate::core::Obs;

pub trait ModuleObsAdapter<T: Obs> {
    /// Converts [crate::core::Env::Obs] to tch tensor.
    fn convert(&self, obs: &T) -> Tensor;

    /// Return the shape of tensors of observation.
    fn shape(&self) -> &[i64];
}

pub trait ModuleActAdapter<T> {
    /// Converts tch tensor to [crate::core::Env::Act].
    fn convert(&self, act: &Tensor) -> T;

    /// Return the shape of tensors of action.
    fn shape(&self) -> &[i64];
}
