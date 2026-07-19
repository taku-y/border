use anyhow::Result;
use border_core::{Act, Obs};
use border_generic_replay_buffer::BatchBase;
use pyo3::prelude::*;

/// Conversion trait for observation and action.
pub trait MinariConverter {
    /// Observation type.
    type Obs: Obs;

    /// Action type.
    type Act: Act;

    /// Batch of observation.
    type ObsBatch: BatchBase + From<Self::Obs>;

    /// Batch of action.
    type ActBatch: BatchBase + From<Self::Act>;

    /// Converts [`PyObject`] into [`Self::Obs`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_observation(&self, obj: &Bound<'_, PyAny>) -> Result<Self::Obs>;

    /// Converts [`Self::Act`] into [`PyObject`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_action(&self, act: Self::Act) -> Result<PyObject>;

    /// Converts [`PyObject`] of an episode into [`Self::ObsBatch`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_observation_batch(&self, obj: &Bound<'_, PyAny>) -> Result<Self::ObsBatch>;

    /// Converts [`PyObject`] of an episode into [`Self::ObsBatch`] for `next_obs`.
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_observation_batch_next(&self, obj: &Bound<'_, PyAny>) -> Result<Self::ObsBatch>;

    /// Converts [`PyObject`] of an episode into [`Self::ActBatch`].
    ///
    /// [`PyObject`]: pyo3::PyObject
    fn convert_action_batch(&self, obj: &Bound<'_, PyAny>) -> Result<Self::ActBatch>;

    /// Returns optional parameters when recovering the environment for evaluation from the dataset.
    fn env_params(&self, py: Python<'_>) -> Vec<(&str, PyObject)>;
}
