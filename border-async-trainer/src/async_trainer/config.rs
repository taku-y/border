use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

/// Configuration of [AsyncTrainer](crate::AsyncTrainer)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AsyncTrainerConfig {
    /// The maximum number of optimization steps.
    pub max_opts: usize,

    /// Where to save the trained model.
    pub model_dir: Option<String>,

    /// Interval of evaluation in training steps.
    pub eval_interval: usize,

    /// Interval of flushing records in optimization steps.
    pub flush_record_interval: usize,

    /// Interval of recording agent information in optimization steps.
    pub record_compute_cost_interval: usize,

    /// Interval of saving the model in optimization steps.
    pub save_interval: usize,

    /// Interval of synchronizing model parameters in training steps.
    pub sync_interval: usize,

    /// Warmup period, for filling replay buffer, in environment steps
    pub warmup_period: usize,

    /// Used for logging.
    #[serde(default)]
    pub n_actors: Option<usize>,
}

impl AsyncTrainerConfig {
    /// Constructs [AsyncTrainerConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [AsyncTrainerConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }

    /// Sets the directory the trained model being saved.
    pub fn model_dir<T: Into<String>>(mut self, model_dir: T) -> Result<Self> {
        self.model_dir = Some(model_dir.into());
        Ok(self)
    }

    /// Sets the number of actors.
    pub fn n_actors(mut self, n_actors: usize) -> Result<Self> {
        self.n_actors = Some(n_actors);
        Ok(self)
    }
}
