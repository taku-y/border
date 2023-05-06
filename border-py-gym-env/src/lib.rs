//! A wrapper of [gym](https://gym.openai.com) environments on Python.
//!
//! This crate uses python>=3.8 and [`gym`]>=0.26 via [`PyO3`].
//! 
//! ## Example
//! 
//! This example applies a random controller ([`Policy`]) to [`cartpole-v1`] environment of [Gym].
//! 
//! ```no_run
//! use anyhow::Result;
//! use border_core::{
//!     record::{BufferedRecorder, Record},
//!     util, Env as _, Policy,
//! };
//! use border_py_gym_env::{
//!     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct,
//!     PyGymEnvDiscreteActRawFilter, PyGymEnvObs, PyGymEnvObsFilter,
//!     PyGymEnvObsRawFilter,
//! };
//! use serde::Serialize;
//! use std::{convert::TryFrom, fs::File};
//!
//! type PyObsDtype = f32;
//!
//! type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! type Act = PyGymEnvDiscreteAct;
//! type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//!
//! #[derive(Clone)]
//! struct RandomPolicyConfig;
//!
//! struct RandomPolicy;
//!
//! impl Policy<Env> for RandomPolicy {
//!     type Config = RandomPolicyConfig;
//!
//!     fn build(_config: Self::Config) -> Self {
//!         Self
//!     }
//!
//!     fn sample(&mut self, _: &Obs) -> Act {
//!         let v = fastrand::u32(..=1);
//!         Act::new(vec![v as i32])
//!     }
//! }
//!
//! #[derive(Debug, Serialize)]
//! struct CartpoleRecord {
//!     episode: usize,
//!     step: usize,
//!     reward: f32,
//!     obs: Vec<f64>,
//! }
//!
//! impl TryFrom<&Record> for CartpoleRecord {
//!     type Error = anyhow::Error;
//!
//!     fn try_from(record: &Record) -> Result<Self> {
//!         Ok(Self {
//!             episode: record.get_scalar("episode")? as _,
//!             step: record.get_scalar("step")? as _,
//!             reward: record.get_scalar("reward")?,
//!             obs: record
//!                 .get_array1("obs")?
//!                 .iter()
//!                 .map(|v| *v as f64)
//!                 .collect(),
//!         })
//!     }
//! }
//!
//! fn main() -> Result<()> {
//!     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
//!         .init();
//!     fastrand::seed(42);
//!
//!     let env_config = PyGymEnvConfig::default()
//!         .name("CartPole-v1".to_string())
//!         .render_mode(Some("human".to_string()))
//!         .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
//!         .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
//!     let mut env = Env::build(&env_config, 0)?;
//!     let mut recorder = BufferedRecorder::new();
//!     env.set_render(true);
//!     let mut policy = RandomPolicy;
//!
//!     let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;
//!
//!     let mut wtr = csv::WriterBuilder::new()
//!         .has_headers(false)
//!         .from_writer(File::create(
//!             "border-py-gym-env/examples/random_cartpole_eval.csv",
//!         )?);
//!     for record in recorder.iter() {
//!         wtr.serialize(CartpoleRecord::try_from(record)?)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//! 
//! ### Relevant types
//! 
//! First, we define aliases of relevant data types. `PyObsDtype` is the data type
//! of arrays of continuous observations emitted from the environment in Python.
//! `Obs` and `Act` are the aliases of observation and action. The type parameter
//! `f32` means that `PyObsDtype` is converted to `f32`. So an agent must be able to
//! handle `f32` array as an observation. The type of [`PyGymEnvDiscreteAct`] is a set of `i32`.
//! 
//! ```no_run
//! use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs};
//! 
//! type PyObsDtype = f32;
//!
//! type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! type Act = PyGymEnvDiscreteAct;
//! ```
//! 
//! `ObsFilter` is an alias of [`PyGymEnvObsRawFilter`], which convertes Python objects of
//! observations from the environment into `Obs` ([`PyGymEnvObs`]) by using 
//! [`PyGymEnvObsFilter::filt`]. Users can define a filter by implementing [`PyGymEnvObsFilter`]
//! trait with arbitrary data processing.
//! 
//! `ActFilter` is an alias of [`PyGymEnvDiscreteActRawFilter`], implementing
//! [`PyGymEnvActFilter`] trait. This trait has a method that converts `Act`
//! into Python objects in order to be sent to the environment.
//! In this example, `Act` itself is a type compatible with Python object,
//! so no processing happens.
//! 
//! ```no_run
//! # use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs};
//! use border_py_gym_env::{PyGymEnvDiscreteActRawFilter, PyGymEnvObsRawFilter};
//! # 
//! # type PyObsDtype = f32;
//! # 
//! # type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! # type Act = PyGymEnvDiscreteAct;
//! type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! ```
//! 
//! [`PyGymEnv`] has four type parameters: observation, action, observation filter and
//! action filter.
//! 
//! ```no_run
//! # use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs};
//! # use border_py_gym_env::{PyGymEnvDiscreteActRawFilter, PyGymEnvObsRawFilter};
//! use border_py_gym_env::PyGymEnv;
//! # 
//! # type PyObsDtype = f32;
//! # 
//! # type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! # type Act = PyGymEnvDiscreteAct;
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! ```
//! 
//! ### Policy
//! 
//! `RandomPolicy` is a controller and implements [`Policy`] trait. Here the policy has no
//! parameter and is dedined as a struct without any field. Structs implementing [`Policy`]
//! should be able to be initialized with a configuration, regardless of if there are configurable
//! parameters, like `RandomPolicyConfig`.
//! It should be noted that [`Policy::Config`] must implements [`Clone`] trait.
//! 
//! There are two methods in [`Policy`] trait. [`Policy::build`] builds an instance
//! given a `config`. [`Policy::sample`] emits an action (discrete action in this example).
//! 
//! ```no_run
//! use border_core::Policy;
//! # use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs, PyGymEnv};
//! # use border_py_gym_env::{PyGymEnvDiscreteActRawFilter, PyGymEnvObsRawFilter};
//! # 
//! # type PyObsDtype = f32;
//! # type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! # type Act = PyGymEnvDiscreteAct;
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! 
//! #[derive(Clone)]
//! struct RandomPolicyConfig;
//! 
//! struct RandomPolicy;
//! 
//! impl Policy<Env> for RandomPolicy {
//!     type Config = RandomPolicyConfig;
//!
//!     fn build(_config: Self::Config) -> Self {
//!         Self
//!     }
//!
//!     fn sample(&mut self, _: &Obs) -> Act {
//!         let v = fastrand::u32(..=1);
//!         Act::new(vec![v as i32])
//!     }
//! }
//! ```
//! 
//! ### Recording
//! 
//! In order to record sequences of states in episodes, we define `CartpoleRecord`.
//! This type has four fields: index of episode (`episode`), index of steps of interactions
//! (`step`), immediate reward (`reward`) and observation (`obs`).
//! 
//! ```no_run
//! # use serde::Serialize;
//! # 
//! #[derive(Debug, Serialize)]
//! struct CartpoleRecord {
//!     episode: usize,
//!     step: usize,
//!     reward: f32,
//!     obs: Vec<f64>,
//! }
//! ```
//! 
//! The sequence of states in an episode will be recorded as a sequence of 
//! [`border_core::record::Record`]. Each record is converted to `CartpoleRecord`
//! with a custom converter as show below: 
//! 
//! ```no_run
//! # use anyhow::Result;
//! use border_core::record::Record;
//! use std::convert::TryFrom;
//! # use serde::Serialize;
//!
//! # #[derive(Debug, Serialize)]
//! # struct CartpoleRecord {
//! #     episode: usize,
//! #     step: usize,
//! #     reward: f32,
//! #     obs: Vec<f64>,
//! # }
//! # 
//! impl TryFrom<&Record> for CartpoleRecord {
//!     type Error = anyhow::Error;
//!
//!     fn try_from(record: &Record) -> Result<Self> {
//!         Ok(Self {
//!             episode: record.get_scalar("episode")? as _,
//!             step: record.get_scalar("step")? as _,
//!             reward: record.get_scalar("reward")?,
//!             obs: record
//!                 .get_array1("obs")?
//!                 .iter()
//!                 .map(|v| *v as f64)
//!                 .collect(),
//!         })
//!     }
//! }
//! ```
//! 
//! ### Run episodes
//! 
//! Having defined required components, let us implement the main function.
//! [`PyGymEnvConfig`] is configuration of an environment.
//! We set some configurations of, e.g., observation and action filters,
//! to [`PyGymEnvConfig`]. We also give a string "CartPole-v1", the name of the 
//! environment for this example. These names are registered in [`gym`].
//! 
//! ```no_run
//! use border_py_gym_env::{PyGymEnvConfig, PyGymEnvObsFilter, PyGymEnvActFilter};
//! # use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs, PyGymEnv};
//! # use border_py_gym_env::{PyGymEnvDiscreteActRawFilter, PyGymEnvObsRawFilter};
//! # 
//! # type PyObsDtype = f32;
//! # type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! # type Act = PyGymEnvDiscreteAct;
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # 
//! let env_config = PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//!     .name("CartPole-v1".to_string())
//!     .render_mode(Some("human".to_string()))
//!     .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
//!     .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
//! ```
//! 
//! Next, we instantiate an environment, recorder and policy.
//! [`BufferedRecorder`] implements [`Recorder`], which has a method used to record some
//! internal variables of functions in the Border library.
//! 
//! In this example, `RandomPolicy` can be initialized without `RandomPolicyConfig`,
//! because `RandomPolicy` has no parameter.
//! 
//! ```no_run
//! use border_core::{Env as _, record::BufferedRecorder};
//! # use border_core::Policy;
//! # use anyhow::Result;
//! # use border_py_gym_env::{PyGymEnvConfig, PyGymEnvObsFilter, PyGymEnvActFilter};
//! # use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs, PyGymEnv};
//! # use border_py_gym_env::{PyGymEnvDiscreteActRawFilter, PyGymEnvObsRawFilter};
//! # 
//! # fn main() -> Result<()> {
//! # 
//! # type PyObsDtype = f32;
//! # type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! # type Act = PyGymEnvDiscreteAct;
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # 
//! # #[derive(Clone)]
//! # struct RandomPolicyConfig;
//! # 
//! # struct RandomPolicy;
//! # 
//! # impl Policy<Env> for RandomPolicy {
//! #     type Config = RandomPolicyConfig;
//! #
//! #     fn build(_config: Self::Config) -> Self {
//! #         Self
//! #     }
//! #
//! #     fn sample(&mut self, _: &Obs) -> Act {
//! #         let v = fastrand::u32(..=1);
//! #         Act::new(vec![v as i32])
//! #     }
//! # }
//! # 
//! # let env_config = PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//! #     .name("CartPole-v1".to_string())
//! #     .render_mode(Some("human".to_string()))
//! #     .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
//! #     .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
//! # 
//! let mut env = Env::build(&env_config, 0)?;
//! let mut recorder = BufferedRecorder::new();
//! env.set_render(true);
//! let mut policy = RandomPolicy;
//! # Ok(())
//! # }
//! ```
//! 
//! With these instances, we can run episodes. `5` means the number of episodes.
//! The sequence of states for 5 episodes will be recorded in `recorder`.
//! After evaluation is finished, we convert sequence of [`Record`]s and
//! write them into a CSV file.
//! 
//! ```no_run
//! use std::fs::File;
//! # use border_core::record::Record;
//! # use std::convert::TryFrom;
//! # use serde::Serialize;
//! # use border_core::util;
//! # use border_core::{Env as _, record::BufferedRecorder};
//! # use border_core::Policy;
//! # use anyhow::Result;
//! # use border_py_gym_env::{PyGymEnvConfig, PyGymEnvObsFilter, PyGymEnvActFilter};
//! # use border_py_gym_env::{PyGymEnvDiscreteAct, PyGymEnvObs, PyGymEnv};
//! # use border_py_gym_env::{PyGymEnvDiscreteActRawFilter, PyGymEnvObsRawFilter};
//! # 
//! # fn main() -> Result<()> {
//! # 
//! # type PyObsDtype = f32;
//! # type Obs = PyGymEnvObs<PyObsDtype, f32>;
//! # type Act = PyGymEnvDiscreteAct;
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # 
//! # #[derive(Clone)]
//! # struct RandomPolicyConfig;
//! # 
//! # struct RandomPolicy;
//! # 
//! # impl Policy<Env> for RandomPolicy {
//! #     type Config = RandomPolicyConfig;
//! #
//! #     fn build(_config: Self::Config) -> Self {
//! #         Self
//! #     }
//! #
//! #     fn sample(&mut self, _: &Obs) -> Act {
//! #         let v = fastrand::u32(..=1);
//! #         Act::new(vec![v as i32])
//! #     }
//! # }
//! # 
//! # #[derive(Debug, Serialize)]
//! # struct CartpoleRecord {
//! #     episode: usize,
//! #     step: usize,
//! #     reward: f32,
//! #     obs: Vec<f64>,
//! # }
//! # 
//! # impl TryFrom<&Record> for CartpoleRecord {
//! #     type Error = anyhow::Error;
//! #
//! #     fn try_from(record: &Record) -> Result<Self> {
//! #         Ok(Self {
//! #             episode: record.get_scalar("episode")? as _,
//! #             step: record.get_scalar("step")? as _,
//! #             reward: record.get_scalar("reward")?,
//! #             obs: record
//! #                 .get_array1("obs")?
//! #                 .iter()
//! #                 .map(|v| *v as f64)
//! #                 .collect(),
//! #         })
//! #     }
//! # }
//! # 
//! # let env_config = PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//! #     .name("CartPole-v1".to_string())
//! #     .render_mode(Some("human".to_string()))
//! #     .obs_filter_config(<ObsFilter as PyGymEnvObsFilter<Obs>>::Config::default())
//! #     .act_filter_config(<ActFilter as PyGymEnvActFilter<Act>>::Config::default());
//! # 
//! # let mut env = Env::build(&env_config, 0)?;
//! # let mut recorder = BufferedRecorder::new();
//! # env.set_render(true);
//! # let mut policy = RandomPolicy;
//! # 
//! let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;
//! 
//! let mut wtr = csv::WriterBuilder::new()
//!     .has_headers(false)
//!     .from_writer(File::create(
//!         "border-py-gym-env/examples/random_cartpole_eval.csv",
//!     )?);
//! for record in recorder.iter() {
//!     wtr.serialize(CartpoleRecord::try_from(record)?)?;
//! }
//! # 
//! # Ok(())
//! # }
//! ```
//! 
//! [`PyO3`]: https://github.com/PyO3/pyo3
//! [`cartpole-v1`]: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
//! [Gym]: https://www.gymlibrary.dev/#
//! [`Policy`]: border_core::Policy
//! [`Policy::Config`]: border_core::Policy::Config
//! [`Policy::build`]: border_core::Policy::build
//! [`Policy::sample`]: border_core::Policy::sample
//! [`BufferedRecorder`]: border_core::record::BufferedRecorder
//! [`Recorder`]: border_core::record::Recorder
//! [`Record`]: border_core::record::Record
//! [`gym`]: https://pypi.org/project/gym/
//! 

/// [`PyGymEnv`] is a wrapper of [gym](https://gym.openai.com) based on [PyO3](https://github.com/PyO3/pyo3).
/// It supports some [classic control](https://gym.openai.com/envs/#classic_control),
/// [Atari](https://gym.openai.com/envs/#atari), and [PyBullet](https://github.com/benelot/pybullet-gym)
/// environments.
///
/// This wrapper accepts array-like observation and action
/// ([Box](https://github.com/openai/gym/blob/master/gym/spaces/box.py) spaces), and
/// discrete action. In order to interact with Python interpreter where gym is running,
/// [`PyGymEnvObsFilter`] and [`PyGymEnvActFilter`] provides interfaces for converting Python object
/// (numpy array) to/from ndarrays in Rust. [`PyGymEnvObsRawFilter`],
/// [`PyGymEnvContinuousActRawFilter`] and [`PyGymEnvDiscreteActRawFilter`] do the conversion for environments
/// where observation and action are arrays. In addition to the data conversion between Python and Rust,
/// we can implements arbitrary preprocessing in these filters. For example, [`FrameStackFilter`] keeps
/// four consevutive observation frames (images) and outputs a stack of these frames.
///
/// For Atari environments, a tweaked version of
/// [atari_wrapper.py](https://github.com/taku-y/border/blob/main/examples/atari_wrappers.py)
/// is required to be in `PYTHONPATH`. The frame stacking preprocessing is implemented in
/// [`FrameStackFilter`] as an [`PyGymEnvObsFilter`].
///
/// Examples with a random controller ([`Policy`](border_core::Policy)) are in
/// [examples](https://github.com/taku-y/border/blob/main/border-py-gym-env/examples) directory.
/// Examples with `border-tch-agents`, which are collections of RL agents implemented with tch-rs,
/// are in [here](https://github.com/taku-y/border/blob/main/border/examples).

mod act_c;
mod act_d;
mod atari;
mod base;
mod config;
mod obs;
mod vec;
pub use act_c::{to_pyobj, PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter};
pub use act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter};
pub use atari::AtariWrapper;
pub use base::{PyGymEnv, PyGymEnvActFilter, PyGymEnvObsFilter, PyGymInfo};
pub use config::PyGymEnvConfig;
pub use obs::{pyobj_to_arrayd, FrameStackFilter, PyGymEnvObs, PyGymEnvObsRawFilter};
// pub use vec::{PyVecGymEnv, PyVecGymEnvConfig};
