use crate::{util, MinariConverter, MinariEnv};
use anyhow::Result;
use border_core::{ExperienceBuffer, ReplayBuffer};
use border_generic_replay_buffer::{
    GenericReplayBuffer, GenericReplayBufferConfig, GenericTransitionBatch,
};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

/// Common interface for Minari datasets.
pub struct MinariDataset {
    pub(crate) dataset: PyObject,
}

impl MinariDataset {
    /// Loads dataset.
    ///
    /// * `dataset_id`: name id of Minari dataset.
    /// * `download`: if `true`, download the dataset. if it is not found locally.
    pub fn load_dataset(dataset_id: impl AsRef<str>, download: bool) -> Result<Self> {
        Python::with_gil(|py| {
            let minari = py.import("minari").unwrap();
            let dataset = minari
                .getattr("load_dataset")?
                .call1((dataset_id.as_ref(), download))?
                .unbind();
            Ok(Self { dataset })
        })
    }

    /// Gets the number of transitions over all episodes.
    pub fn get_num_transitions(&self, episode_indices: Option<Vec<usize>>) -> Result<usize> {
        Python::with_gil(|py| {
            let episodes = self
                .dataset
                .call_method1(py, "iterate_episodes", (episode_indices,))?;
            let mut total = 0;

            // Iterate over episodes
            for ep in episodes.bind(py).try_iter()? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;
                total += ep
                    .getattr("rewards")?
                    .call_method0("__len__")?
                    .extract::<usize>()?;
            }
            Ok(total)
        })
    }

    /// Creates replay buffer from the dataset.
    ///
    /// The order of transitions in the original dataset is preserved,
    /// but the boundary between episodes is discarded.
    ///
    /// * `converter`: converter for observation and action.
    /// * `episode_indices`: indices of episodes to be included in the replay buffer.
    ///   If `None`, all episodes are included.
    ///   If `Some(indices)`, only episodes with the indices are included.
    ///   All transitions in the episodes are flattened.
    pub fn create_replay_buffer<T: MinariConverter>(
        &self,
        converter: &mut T,
        episode_indices: Option<Vec<usize>>,
    ) -> Result<GenericReplayBuffer<T::ObsBatch, T::ActBatch>>
    where
        T::ObsBatch: std::fmt::Debug,
        T::ActBatch: std::fmt::Debug,
    {
        Python::with_gil(|py| {
            // The total number of transitions
            let num_transitions = self.get_num_transitions(episode_indices.clone())?;

            // Prepare replay buffer
            let mut replay_buffer = GenericReplayBuffer::build(&GenericReplayBufferConfig {
                capacity: num_transitions,
                seed: 0,
                per_config: None,
            });

            let episodes = self
                .dataset
                .call_method1(py, "iterate_episodes", (episode_indices,))?;

            // Iterate over episodes
            for ep in episodes.bind(py).try_iter()? {
                // ep is minari.dataset.episode_data.EpisodeData
                let ep = ep?;

                // Extract transitions in the episode as a batch
                let batch = Self::extract_transitions_in_episode(&ep, converter)?;

                // Push the batch to the replay buffer
                replay_buffer.push(batch)?;
            }

            // Stats in the replay buffer
            log::info!("In replay buffer:");
            log::info!("{} transitions", num_transitions);
            log::info!("{} terminated flags", replay_buffer.num_terminated_flags());
            log::info!("{} truncated flags", replay_buffer.num_truncated_flags());
            log::info!("{} reward sum", replay_buffer.sum_rewards());

            Ok(replay_buffer)
        })
    }

    fn extract_transitions_in_episode<T: MinariConverter>(
        ep: &Bound<'_, PyAny>,
        converter: &T,
    ) -> Result<GenericTransitionBatch<T::ObsBatch, T::ActBatch>>
    where
        T::ObsBatch: std::fmt::Debug,
        T::ActBatch: std::fmt::Debug,
    {
        // Extract episode as batch
        let obs = ep.getattr("observations")?;
        let act = ep.getattr("actions")?;
        let rew = ep.getattr("rewards")?;
        let trm = ep.getattr("terminations")?;
        let trn = ep.getattr("truncations")?;

        // Creates batch
        let next_obs = converter.convert_observation_batch_next(&obs)?;
        let obs = converter.convert_observation_batch(&obs)?;
        let act = converter.convert_action_batch(&act)?;
        let reward = util::vec::pyany_to_f32vec(&rew)?;
        let is_terminated = util::vec::pyany_to_vec::<i8>(&trm)?;
        let is_truncated = util::vec::pyany_to_vec::<i8>(&trn)?;

        Ok(GenericTransitionBatch {
            obs,
            act,
            next_obs,
            reward,
            is_terminated,
            is_truncated,
            weight: None,
            ix_sample: None,
        })
    }

    /// Recovers the environment.
    ///
    /// * `converter`: converter for observation and action.
    /// * `eval_env`: if `true`, the environment is for evaluation.
    ///   See [Minari API documentation](https://minari.farama.org/api/minari_dataset/minari_dataset/#minari.MinariDataset.recover_environment).
    /// * `render_mode`: render mode for the environment.
    pub fn recover_environment<'a, T: MinariConverter>(
        &self,
        converter: T,
        eval_env: bool,
        render_mode: impl Into<Option<&'a str>>,
    ) -> Result<MinariEnv<T>> {
        let env = {
            Python::with_gil(|py| {
                let render_mode_obj = match render_mode.into() {
                    Some(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
                    None => py.None(),
                };
                let mut kwargs: Vec<(&str, PyObject)> = vec![("render_mode", render_mode_obj)];
                kwargs.extend(converter.env_params(py));
                let kwargs = kwargs.into_py_dict(py)?;
                self.dataset
                    .call_method(py, "recover_environment", (eval_env,), Some(&kwargs))
            })?
        };

        let ref_score_minmax = Python::with_gil(|py| {
            let min = self
                .dataset
                .getattr(py, "storage")
                .unwrap()
                .getattr(py, "metadata")
                .unwrap()
                .call_method1(py, "get", ("ref_min_score",));
            let max = self
                .dataset
                .getattr(py, "storage")
                .unwrap()
                .getattr(py, "metadata")
                .unwrap()
                .call_method1(py, "get", ("ref_max_score",));

            if min.is_err() || max.is_err() {
                log::info!("Either ref_min_score or ref_max_score is missing. Normalized score cannot be calculated.");
                None
            } else {
                let min = min.unwrap().extract(py);
                let max = max.unwrap().extract(py);
                if min.is_ok() && max.is_ok() {
                    let min = min.unwrap();
                    let max = max.unwrap();
                    log::info!("ref_min_score = {min}");
                    log::info!("ref_max_score = {max}");
                    Some((min, max))
                } else {
                    log::info!("Either ref_min_score or ref_max_score could not be converted to f32. Normalized score cannot be calculated.");
                    None
                }
            }
        });

        Ok(MinariEnv {
            converter,
            env,
            initial_seed: None,
            count_steps: 0,
            max_steps: None,
            ref_score_minmax,
            // dataset: self.dataset.clone(),
        })
    }
}
