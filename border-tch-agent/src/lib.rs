//! RL agents implemented with [tch](https://crates.io/crates/tch).
//!
//! ## Example
//!
//! The below code is an example where [`Dqn`] agent is trained and evaluated on the cartpole environment.
//!
//! ```no_run
//! use anyhow::Result;
//! use border_core::{
//!     record::{BufferedRecorder, Record, TensorboardRecorder},
//!     replay_buffer::{
//!         SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
//!         SimpleStepProcessorConfig, SubBatch,
//!     },
//!     util, Agent, Env as _, Policy, Trainer, TrainerConfig,
//! };
//! use border_py_gym_env::{
//!     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
//!     PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
//! };
//! use border_tch_agent::{
//!     dqn::{Dqn, DqnConfig, DqnModelConfig},
//!     mlp::{Mlp, MlpConfig},
//!     TensorSubBatch,
//! };
//! use clap::{App, Arg};
//! use csv::WriterBuilder;
//! use serde::Serialize;
//! use std::{convert::TryFrom, fs::File};
//! use tch::Tensor;
//!
//! const DIM_OBS: i64 = 4;
//! const DIM_ACT: i64 = 2;
//! const LR_CRITIC: f64 = 0.001;
//! const DISCOUNT_FACTOR: f64 = 0.99;
//! const BATCH_SIZE: usize = 64;
//! const N_TRANSITIONS_WARMUP: usize = 100;
//! const N_UPDATES_PER_OPT: usize = 1;
//! const TAU: f64 = 0.005;
//! const OPT_INTERVAL: usize = 50;
//! const MAX_OPTS: usize = 1000;
//! const EVAL_INTERVAL: usize = 50;
//! const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! const N_EPISODES_PER_EVAL: usize = 5;
//! const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";
//!
//! type PyObsDtype = f32;
//!
//! #[derive(Clone, Debug)]
//! struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//!
//! impl border_core::Obs for Obs {
//!     fn dummy(n: usize) -> Self {
//!         Obs(PyGymEnvObs::dummy(n))
//!     }
//!
//!     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//!         Obs(self.0.merge(obs_reset.0, is_done))
//!     }
//!
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//!
//! impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//!     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//!         Obs(obs)
//!     }
//! }
//!
//! impl From<Obs> for Tensor {
//!     fn from(obs: Obs) -> Tensor {
//!         Tensor::try_from(&obs.0.obs).unwrap()
//!     }
//! }
//!
//! struct ObsBatch(TensorSubBatch);
//!
//! impl SubBatch for ObsBatch {
//!     fn new(capacity: usize) -> Self {
//!         Self(TensorSubBatch::new(capacity))
//!     }
//!
//!     fn push(&mut self, i: usize, data: &Self) {
//!         self.0.push(i, &data.0)
//!     }
//!
//!     fn sample(&self, ixs: &Vec<usize>) -> Self {
//!         let buf = self.0.sample(ixs);
//!         Self(buf)
//!     }
//! }
//!
//! impl From<Obs> for ObsBatch {
//!     fn from(obs: Obs) -> Self {
//!         let tensor = obs.into();
//!         Self(TensorSubBatch::from_tensor(tensor))
//!     }
//! }
//!
//! impl From<ObsBatch> for Tensor {
//!     fn from(b: ObsBatch) -> Self {
//!         b.0.into()
//!     }
//! }
//!
//! #[derive(Clone, Debug)]
//! struct Act(PyGymEnvDiscreteAct);
//!
//! impl border_core::Act for Act {
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//!
//! impl Into<PyGymEnvDiscreteAct> for Act {
//!     fn into(self) -> PyGymEnvDiscreteAct {
//!         self.0
//!     }
//! }
//!
//! struct ActBatch(TensorSubBatch);
//!
//! impl SubBatch for ActBatch {
//!     fn new(capacity: usize) -> Self {
//!         Self(TensorSubBatch::new(capacity))
//!     }
//!
//!     fn push(&mut self, i: usize, data: &Self) {
//!         self.0.push(i, &data.0)
//!     }
//!
//!     fn sample(&self, ixs: &Vec<usize>) -> Self {
//!         let buf = self.0.sample(ixs);
//!         Self(buf)
//!     }
//! }
//!
//! impl From<Act> for Tensor {
//!     fn from(act: Act) -> Tensor {
//!         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//!         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//!
//!         // The first dimension of the action tensor is the number of processes,
//!         // which is 1 for the non-vectorized environment.
//!         t.unsqueeze(0)
//!     }
//! }
//!
//! impl From<Act> for ActBatch {
//!     fn from(act: Act) -> Self {
//!         let tensor = act.into();
//!         Self(TensorSubBatch::from_tensor(tensor))
//!     }
//! }
//!
//! impl From<ActBatch> for Tensor {
//!     fn from(act: ActBatch) -> Self {
//!         act.0.into()
//!     }
//! }
//!
//! impl From<Tensor> for Act {
//!     // `t` must be a 1-dimentional tensor of `f32`
//!     fn from(t: Tensor) -> Self {
//!         let data: Vec<i64> = t.into();
//!         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//!         Act(PyGymEnvDiscreteAct::new(data))
//!     }
//! }
//!
//! type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//! type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
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
//! fn create_agent(in_dim: i64, out_dim: i64) -> Dqn<Env, Mlp, ReplayBuffer> {
//!     let device = tch::Device::cuda_if_available();
//!     let config = {
//!         let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
//!         let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim, true);
//!         let model_config = DqnModelConfig::default()
//!             .q_config(mlp_config)
//!             .out_dim(out_dim)
//!             .opt_config(opt_config);
//!         DqnConfig::default()
//!             .n_updates_per_opt(N_UPDATES_PER_OPT)
//!             .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//!             .batch_size(BATCH_SIZE)
//!             .discount_factor(DISCOUNT_FACTOR)
//!             .tau(TAU)
//!             .model_config(model_config)
//!             .device(device)
//!     };
//!
//!     Dqn::build(config)
//! }
//!
//! fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
//!     PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//!         .name("CartPole-v0".to_string())
//!         .obs_filter_config(ObsFilter::default_config())
//!         .act_filter_config(ActFilter::default_config())
//! }
//!
//! fn train(max_opts: usize, model_dir: &str) -> Result<()> {
//!     let mut trainer = {
//!         let env_config = env_config();
//!         let step_proc_config = SimpleStepProcessorConfig {};
//!         let replay_buffer_config =
//!             SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
//!         let config = TrainerConfig::default()
//!             .max_opts(max_opts)
//!             .opt_interval(OPT_INTERVAL)
//!             .eval_interval(EVAL_INTERVAL)
//!             .record_interval(EVAL_INTERVAL)
//!             .save_interval(EVAL_INTERVAL)
//!             .eval_episodes(N_EPISODES_PER_EVAL)
//!             .model_dir(model_dir);
//!         let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
//!             config,
//!             env_config,
//!             None,
//!             step_proc_config,
//!             replay_buffer_config,
//!         );
//!
//!         trainer
//!     };
//!
//!     let mut recorder = TensorboardRecorder::new(model_dir);
//!     let mut agent = create_agent(DIM_OBS, DIM_ACT);
//!
//!     trainer.train(&mut agent, &mut recorder)?;
//!
//!     Ok(())
//! }
//!
//! fn eval(model_dir: &str, render: bool) -> Result<()> {
//!     let mut env_config = env_config();
//!     if render {
//!         env_config = env_config.render_mode(Some("human".to_string()));
//!     }
//!     let mut env = Env::build(&env_config, 0)?;
//!     let mut agent = create_agent(DIM_OBS, DIM_ACT);
//!     let mut recorder = BufferedRecorder::new();
//!     env.set_render(render);
//!     if render {
//!         env.set_wait_in_step(std::time::Duration::from_millis(10));
//!     }
//!     agent.load(model_dir)?;
//!     agent.eval();
//!
//!     let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;
//!
//!     // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
//!     let mut wtr = WriterBuilder::new()
//!         .has_headers(false)
//!         .from_writer(File::create(model_dir.to_string() + "/eval.csv")?);
//!     for record in recorder.iter() {
//!         wtr.serialize(CartpoleRecord::try_from(record)?)?;
//!     }
//!
//!     Ok(())
//! }
//!
//! fn main() -> Result<()> {
//!     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
//!     tch::manual_seed(42);
//!
//!     let matches = App::new("dqn_cartpole")
//!         .version("0.1.0")
//!         .author("Taku Yoshioka <yoshioka@laboro.ai>")
//!         .arg(
//!             Arg::with_name("train")
//!                 .long("train")
//!                 .takes_value(false)
//!                 .help("Do training only"),
//!         )
//!         .arg(
//!             Arg::with_name("eval")
//!                 .long("eval")
//!                 .takes_value(false)
//!                 .help("Do evaluation only"),
//!         )
//!         .get_matches();
//!
//!     let do_train = (matches.is_present("train") && !matches.is_present("eval"))
//!         || (!matches.is_present("train") && !matches.is_present("eval"));
//!     let do_eval = (!matches.is_present("train") && matches.is_present("eval"))
//!         || (!matches.is_present("train") && !matches.is_present("eval"));
//!
//!     if do_train {
//!         train(MAX_OPTS, MODEL_DIR)?;
//!     }
//!     if do_eval {
//!         eval(&(MODEL_DIR.to_owned() + "/best"), true)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Observation
//!
//! First, we define observation type `Obs`, which wraps [`PyGymEnvObs`].
//! This type is required in order to implement some conversion traits.
//! `Obs` must implement [`border_core::Obs`] to satisfy trait bound of
//! [`border_core::Env::Obs`].
//!
//! ```no_run
//! use border_py_gym_env::PyGymEnvObs;
//!
//! type PyObsDtype = f32;
//!
//! #[derive(Clone, Debug)]
//! struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//!
//! impl border_core::Obs for Obs {
//!     fn dummy(n: usize) -> Self {
//!         Obs(PyGymEnvObs::dummy(n))
//!     }
//!
//!     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//!         Obs(self.0.merge(obs_reset.0, is_done))
//!     }
//!
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//! ```
//!
//! ### Conversion
//!
//! Here is the code for conversion from [`PyGymEnvObs`] to `Obs`.
//! This conversion is used in [`PyGymEnvObsRawFilter`].
//!
//! ```no_run
//! # use border_py_gym_env::PyGymEnvObs;
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//!     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//!         Obs(obs)
//!     }
//! }
//! ```
//!
//! In order for `Obs` to be given to the model of [`Dqn`] agent,
//! which is implemented with [`tch`], `Obs` should be able to be
//! converted to [`tch::Tensor`].
//!
//! ```no_run
//! use std::convert::TryFrom;
//! use tch::Tensor;
//! #
//! # use border_py_gym_env::PyGymEnvObs;
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//!
//! impl From<Obs> for Tensor {
//!     fn from(obs: Obs) -> Tensor {
//!         Tensor::try_from(&obs.0.obs).unwrap()
//!     }
//! }
//! ```
//!
//! ### Minibatch
//!
//! [`border_core::replay_buffer::StdBatch`] is an implementation of minibatch and
//! can be pushed to [`border_core::replay_buffer::SimpleReplayBuffer`].
//! [`StdBatch`] accepts arbitraty observation and action types.
//! These types (type parameters `O` and `A` of [`StdBatch`]) must implement
//! [`border_core::replay_buffer::SubBatch`] to be used to compose [`StdBatch`].
//! Since [`TensorSubBatch`] implements [`SubBatch`], it can be used to store observations.
//! In order to define conversion traits, we wrap [`TensorSubBatch`] and define type `ObsBatch`.
//!
//! ```no_run
//! use border_core::replay_buffer::SubBatch;
//! use border_tch_agent::TensorSubBatch;
//!
//! struct ObsBatch(TensorSubBatch);
//!
//! impl SubBatch for ObsBatch {
//!     fn new(capacity: usize) -> Self {
//!         Self(TensorSubBatch::new(capacity))
//!     }
//!
//!     fn push(&mut self, i: usize, data: &Self) {
//!         self.0.push(i, &data.0)
//!     }
//!
//!     fn sample(&self, ixs: &Vec<usize>) -> Self {
//!         let buf = self.0.sample(ixs);
//!         Self(buf)
//!     }
//! }
//! ```
//!
//! In every interaction steps, `Obs` in [`border_core::Step`] object is converted into `ObsBatch`
//! and used to create [`StdBatch`] (a minibatch with a single triplet (o, a, o')).
//! Then, the minibatch is pushed into [`SimpleReplayBuffer`].
//! The conversion trait below is used in the above process. This conversion trait is required by
//! [`border_core::replay_buffer::SimpleStepProcessor`]. See "Interaction of objects" section in
//! [`border_core::Trainer`].
//!
//! ```no_run
//! # use std::convert::TryFrom;
//! # use tch::Tensor;
//! # use border_core::replay_buffer::SubBatch;
//! # use border_tch_agent::TensorSubBatch;
//! #
//! # struct ObsBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ObsBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # use border_py_gym_env::PyGymEnvObs;
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         Tensor::try_from(&obs.0.obs).unwrap()
//! #     }
//! # }
//! impl From<Obs> for ObsBatch {
//!     fn from(obs: Obs) -> Self {
//!         let tensor = obs.into();
//!         Self(TensorSubBatch::from_tensor(tensor))
//!     }
//! }
//! ```
//!
//! The conversion trait defined here is required by [`Dqn`] in order to train the model of the
//! action value function (Q-function). [`Tensor`] corresponds to `Q::Input` in the
//! trait bound of [`Dqn`], where type parameter `Q` is a model implemented with [`tch`].
//!
//! ```no_run
//! # use tch::Tensor;
//! # use border_tch_agent::TensorSubBatch;
//! # struct ObsBatch(TensorSubBatch);
//! #
//! impl From<ObsBatch> for Tensor {
//!     fn from(b: ObsBatch) -> Self {
//!         b.0.into()
//!     }
//! }
//! ```
//!
//! ## Action
//!
//! Similarly to the case of observation, type `Act` is defined for action with
//! type [`PyGymEnvDiscreteAct`] provided by the library and [`border_core::Act`]
//! is implemented on `Act`.
//!
//! ```no_run
//! use border_py_gym_env::PyGymEnvDiscreteAct;
//!
//! #[derive(Clone, Debug)]
//! struct Act(PyGymEnvDiscreteAct);
//!
//! impl border_core::Act for Act {
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//! ```
//!
//! This conversion trait is required by [`PyGymEnvDiscreteActRawFilter`] for
//! applying action, sampled from the agent, to the environment.
//!
//! ```no_run
//! # use border_py_gym_env::PyGymEnvDiscreteAct;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! impl Into<PyGymEnvDiscreteAct> for Act {
//!     fn into(self) -> PyGymEnvDiscreteAct {
//!         self.0
//!     }
//! }
//! ```
//!
//! `Act` and [`Tensor`] are converted with each other in order to use
//! the model of the action value function of [`Dqn`] agent.
//!
//! ```no_run
//! use std::convert::TryFrom;
//! # use tch::Tensor;
//! #
//! # use border_py_gym_env::PyGymEnvDiscreteAct;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! impl From<Act> for Tensor {
//!     fn from(act: Act) -> Tensor {
//!         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//!         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//!
//!         // The first dimension of the action tensor is the number of processes,
//!         // which is 1 for the non-vectorized environment.
//!         t.unsqueeze(0)
//!     }
//! }
//!
//! impl From<Tensor> for Act {
//!     // `t` must be a 1-dimentional tensor of `f32`
//!     fn from(t: Tensor) -> Self {
//!         let data: Vec<i64> = t.into();
//!         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//!         Act(PyGymEnvDiscreteAct::new(data))
//!     }
//! }
//! ```
//!
//! A set of actions is handled as `ActBatch`, which composes
//! [`StdBatch`] with `ObsBatch`.
//!
//! ```no_run
//! use border_tch_agent::TensorSubBatch;
//! use border_core::replay_buffer::SubBatch;
//!
//! struct ActBatch(TensorSubBatch);
//!
//! impl SubBatch for ActBatch {
//!     fn new(capacity: usize) -> Self {
//!         Self(TensorSubBatch::new(capacity))
//!     }
//!
//!     fn push(&mut self, i: usize, data: &Self) {
//!         self.0.push(i, &data.0)
//!     }
//!
//!     fn sample(&self, ixs: &Vec<usize>) -> Self {
//!         let buf = self.0.sample(ixs);
//!         Self(buf)
//!     }
//! }
//! ```
//!
//! We implements conversion traits for `ActBatch`.
//!
//! ```no_run
//! # use std::convert::TryFrom;
//! # use tch::Tensor;
//! # use border_tch_agent::TensorSubBatch;
//! # use border_core::replay_buffer::SubBatch;
//! #
//! # use border_py_gym_env::PyGymEnvDiscreteAct;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! #
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! #
//! # impl From<Tensor> for Act {
//! #     // `t` must be a 1-dimentional tensor of `f32`
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! # struct ActBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ActBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! impl From<Act> for ActBatch {
//!     fn from(act: Act) -> Self {
//!         let tensor = act.into();
//!         Self(TensorSubBatch::from_tensor(tensor))
//!     }
//! }
//!
//! impl From<ActBatch> for Tensor {
//!     fn from(act: ActBatch) -> Self {
//!         act.0.into()
//!     }
//! }
//! ```
//!
//! ## Type aliases
//!
//! For brevity of the code, we define type aliases as below.
//! `StepProc` is used to generate minibathces from observation and
//! action. Two type parameters `ObsBatch` and `ActBatch` composes
//! [`StdBatch`]. `ReplayBuffer` is an aliase of [`SimpleReplayBuffer`]
//! that accepts [`StdBatch`] composed of `ObsBatch` and `ActBatch`.
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     replay_buffer::{
//! #         SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
//! #         SimpleStepProcessorConfig, SubBatch,
//! #     },
//! #     util, Agent, Env as _, Policy, Trainer, TrainerConfig,
//! # };
//! # use border_py_gym_env::{
//! #     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
//! #     PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
//! # };
//! # use border_tch_agent::{
//! #     dqn::{Dqn, DqnConfig, DqnModelConfig},
//! #     mlp::{Mlp, MlpConfig},
//! #     TensorSubBatch,
//! # };
//! # use clap::{App, Arg};
//! # use csv::WriterBuilder;
//! # use serde::Serialize;
//! # use std::{convert::TryFrom, fs::File};
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # const DIM_ACT: i64 = 2;
//! # const LR_CRITIC: f64 = 0.001;
//! # const DISCOUNT_FACTOR: f64 = 0.99;
//! # const BATCH_SIZE: usize = 64;
//! # const N_TRANSITIONS_WARMUP: usize = 100;
//! # const N_UPDATES_PER_OPT: usize = 1;
//! # const TAU: f64 = 0.005;
//! # const OPT_INTERVAL: usize = 50;
//! # const MAX_OPTS: usize = 1000;
//! # const EVAL_INTERVAL: usize = 50;
//! # const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! # const N_EPISODES_PER_EVAL: usize = 5;
//! # const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//! #     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//! #         Obs(obs)
//! #     }
//! # }
//! #
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         Tensor::try_from(&obs.0.obs).unwrap()
//! #     }
//! # }
//! #
//! # struct ObsBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ObsBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Obs> for ObsBatch {
//! #     fn from(obs: Obs) -> Self {
//! #         let tensor = obs.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ObsBatch> for Tensor {
//! #     fn from(b: ObsBatch) -> Self {
//! #         b.0.into()
//! #     }
//! # }
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! # impl border_core::Act for Act {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl Into<PyGymEnvDiscreteAct> for Act {
//! #     fn into(self) -> PyGymEnvDiscreteAct {
//! #         self.0
//! #     }
//! # }
//! #
//! # struct ActBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ActBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! #
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! #
//! # impl From<Act> for ActBatch {
//! #     fn from(act: Act) -> Self {
//! #         let tensor = act.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ActBatch> for Tensor {
//! #     fn from(act: ActBatch) -> Self {
//! #         act.0.into()
//! #     }
//! # }
//! #
//! # impl From<Tensor> for Act {
//! #     // `t` must be a 1-dimentional tensor of `f32`
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//! type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! ```
//!
//! ## Create agent
//!
//! The configuration of [`Dqn`] agent consists of the configurations of optimizer and
//! model architecture. In this example, there are two linear layers of 256 units.
//! Nonlinear activation functions are inserted between layers when constructing
//! [`Mlp`].
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     replay_buffer::{
//! #         SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
//! #         SimpleStepProcessorConfig, SubBatch,
//! #     },
//! #     util, Agent, Env as _, Policy, Trainer, TrainerConfig,
//! # };
//! # use border_py_gym_env::{
//! #     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
//! #     PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
//! # };
//! # use border_tch_agent::{
//! #     dqn::{Dqn, DqnConfig, DqnModelConfig},
//! #     mlp::{Mlp, MlpConfig},
//! #     TensorSubBatch,
//! # };
//! # use clap::{App, Arg};
//! # use csv::WriterBuilder;
//! # use serde::Serialize;
//! # use std::{convert::TryFrom, fs::File};
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # const DIM_ACT: i64 = 2;
//! # const LR_CRITIC: f64 = 0.001;
//! # const DISCOUNT_FACTOR: f64 = 0.99;
//! # const BATCH_SIZE: usize = 64;
//! # const N_TRANSITIONS_WARMUP: usize = 100;
//! # const N_UPDATES_PER_OPT: usize = 1;
//! # const TAU: f64 = 0.005;
//! # const OPT_INTERVAL: usize = 50;
//! # const MAX_OPTS: usize = 1000;
//! # const EVAL_INTERVAL: usize = 50;
//! # const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! # const N_EPISODES_PER_EVAL: usize = 5;
//! # const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//! #     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//! #         Obs(obs)
//! #     }
//! # }
//! #
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         Tensor::try_from(&obs.0.obs).unwrap()
//! #     }
//! # }
//! #
//! # struct ObsBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ObsBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Obs> for ObsBatch {
//! #     fn from(obs: Obs) -> Self {
//! #         let tensor = obs.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ObsBatch> for Tensor {
//! #     fn from(b: ObsBatch) -> Self {
//! #         b.0.into()
//! #     }
//! # }
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! # impl border_core::Act for Act {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl Into<PyGymEnvDiscreteAct> for Act {
//! #     fn into(self) -> PyGymEnvDiscreteAct {
//! #         self.0
//! #     }
//! # }
//! #
//! # struct ActBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ActBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! #
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! #
//! # impl From<Act> for ActBatch {
//! #     fn from(act: Act) -> Self {
//! #         let tensor = act.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ActBatch> for Tensor {
//! #     fn from(act: ActBatch) -> Self {
//! #         act.0.into()
//! #     }
//! # }
//! #
//! # impl From<Tensor> for Act {
//! #     // `t` must be a 1-dimentional tensor of `f32`
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//! # type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! #
//! fn create_agent(in_dim: i64, out_dim: i64) -> Dqn<Env, Mlp, ReplayBuffer> {
//!     let device = tch::Device::cuda_if_available();
//!     let config = {
//!         let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
//!         let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim, true);
//!         let model_config = DqnModelConfig::default()
//!             .q_config(mlp_config)
//!             .out_dim(out_dim)
//!             .opt_config(opt_config);
//!         DqnConfig::default()
//!             .n_updates_per_opt(N_UPDATES_PER_OPT)
//!             .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//!             .batch_size(BATCH_SIZE)
//!             .discount_factor(DISCOUNT_FACTOR)
//!             .tau(TAU)
//!             .model_config(model_config)
//!             .device(device)
//!     };
//!
//!     Dqn::build(config)
//! }
//! ```
//! ## Configuration of environment
//!
//! The configuration of environment is created from name of a registered environment,
//! observation and action filters.
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     replay_buffer::{
//! #         SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
//! #         SimpleStepProcessorConfig, SubBatch,
//! #     },
//! #     util, Agent, Env as _, Policy, Trainer, TrainerConfig,
//! # };
//! # use border_py_gym_env::{
//! #     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
//! #     PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
//! # };
//! # use border_tch_agent::{
//! #     dqn::{Dqn, DqnConfig, DqnModelConfig},
//! #     mlp::{Mlp, MlpConfig},
//! #     TensorSubBatch,
//! # };
//! # use clap::{App, Arg};
//! # use csv::WriterBuilder;
//! # use serde::Serialize;
//! # use std::{convert::TryFrom, fs::File};
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # const DIM_ACT: i64 = 2;
//! # const LR_CRITIC: f64 = 0.001;
//! # const DISCOUNT_FACTOR: f64 = 0.99;
//! # const BATCH_SIZE: usize = 64;
//! # const N_TRANSITIONS_WARMUP: usize = 100;
//! # const N_UPDATES_PER_OPT: usize = 1;
//! # const TAU: f64 = 0.005;
//! # const OPT_INTERVAL: usize = 50;
//! # const MAX_OPTS: usize = 1000;
//! # const EVAL_INTERVAL: usize = 50;
//! # const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! # const N_EPISODES_PER_EVAL: usize = 5;
//! # const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//! #     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//! #         Obs(obs)
//! #     }
//! # }
//! #
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         Tensor::try_from(&obs.0.obs).unwrap()
//! #     }
//! # }
//! #
//! # struct ObsBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ObsBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Obs> for ObsBatch {
//! #     fn from(obs: Obs) -> Self {
//! #         let tensor = obs.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ObsBatch> for Tensor {
//! #     fn from(b: ObsBatch) -> Self {
//! #         b.0.into()
//! #     }
//! # }
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! # impl border_core::Act for Act {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl Into<PyGymEnvDiscreteAct> for Act {
//! #     fn into(self) -> PyGymEnvDiscreteAct {
//! #         self.0
//! #     }
//! # }
//! #
//! # struct ActBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ActBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! #
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! #
//! # impl From<Act> for ActBatch {
//! #     fn from(act: Act) -> Self {
//! #         let tensor = act.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ActBatch> for Tensor {
//! #     fn from(act: ActBatch) -> Self {
//! #         act.0.into()
//! #     }
//! # }
//! #
//! # impl From<Tensor> for Act {
//! #     // `t` must be a 1-dimentional tensor of `f32`
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//! # type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! #
//! fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
//!     PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//!         .name("CartPole-v0".to_string())
//!         .obs_filter_config(ObsFilter::default_config())
//!         .act_filter_config(ActFilter::default_config())
//! }
//! ```
//!
//! ## Training
//!
//! [`border_core::Trainer`] is used to train the agent. The configuration of
//! [`Trainer`] consists of configurations of environment, step processor and
//! replay buffer. [`TensorboardRecorder`] is used to record some metrices during
//! training process. [`Trainer::train`] method execute training with given
//! agent and recorder. During the training, metrices and model parameters are
//! saved in directory `model_dir`.
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     replay_buffer::{
//! #         SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
//! #         SimpleStepProcessorConfig, SubBatch,
//! #     },
//! #     util, Agent, Env as _, Policy, Trainer, TrainerConfig,
//! # };
//! # use border_py_gym_env::{
//! #     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
//! #     PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
//! # };
//! # use border_tch_agent::{
//! #     dqn::{Dqn, DqnConfig, DqnModelConfig},
//! #     mlp::{Mlp, MlpConfig},
//! #     TensorSubBatch,
//! # };
//! # use clap::{App, Arg};
//! # use csv::WriterBuilder;
//! # use serde::Serialize;
//! # use std::{convert::TryFrom, fs::File};
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # const DIM_ACT: i64 = 2;
//! # const LR_CRITIC: f64 = 0.001;
//! # const DISCOUNT_FACTOR: f64 = 0.99;
//! # const BATCH_SIZE: usize = 64;
//! # const N_TRANSITIONS_WARMUP: usize = 100;
//! # const N_UPDATES_PER_OPT: usize = 1;
//! # const TAU: f64 = 0.005;
//! # const OPT_INTERVAL: usize = 50;
//! # const MAX_OPTS: usize = 1000;
//! # const EVAL_INTERVAL: usize = 50;
//! # const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! # const N_EPISODES_PER_EVAL: usize = 5;
//! # const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//! #     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//! #         Obs(obs)
//! #     }
//! # }
//! #
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         Tensor::try_from(&obs.0.obs).unwrap()
//! #     }
//! # }
//! #
//! # struct ObsBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ObsBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Obs> for ObsBatch {
//! #     fn from(obs: Obs) -> Self {
//! #         let tensor = obs.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ObsBatch> for Tensor {
//! #     fn from(b: ObsBatch) -> Self {
//! #         b.0.into()
//! #     }
//! # }
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! # impl border_core::Act for Act {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl Into<PyGymEnvDiscreteAct> for Act {
//! #     fn into(self) -> PyGymEnvDiscreteAct {
//! #         self.0
//! #     }
//! # }
//! #
//! # struct ActBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ActBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! #
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! #
//! # impl From<Act> for ActBatch {
//! #     fn from(act: Act) -> Self {
//! #         let tensor = act.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ActBatch> for Tensor {
//! #     fn from(act: ActBatch) -> Self {
//! #         act.0.into()
//! #     }
//! # }
//! #
//! # impl From<Tensor> for Act {
//! #     // `t` must be a 1-dimentional tensor of `f32`
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//! # type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! #
//! # fn create_agent(in_dim: i64, out_dim: i64) -> Dqn<Env, Mlp, ReplayBuffer> {
//! #     let device = tch::Device::cuda_if_available();
//! #     let config = {
//! #         let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
//! #         let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim, true);
//! #         let model_config = DqnModelConfig::default()
//! #             .q_config(mlp_config)
//! #             .out_dim(out_dim)
//! #             .opt_config(opt_config);
//! #         DqnConfig::default()
//! #             .n_updates_per_opt(N_UPDATES_PER_OPT)
//! #             .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//! #             .batch_size(BATCH_SIZE)
//! #             .discount_factor(DISCOUNT_FACTOR)
//! #             .tau(TAU)
//! #             .model_config(model_config)
//! #             .device(device)
//! #     };
//! #
//! #     Dqn::build(config)
//! # }
//! #
//! # fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
//! #     PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//! #         .name("CartPole-v0".to_string())
//! #         .obs_filter_config(ObsFilter::default_config())
//! #         .act_filter_config(ActFilter::default_config())
//! # }
//! fn train(max_opts: usize, model_dir: &str) -> Result<()> {
//!     let mut trainer = {
//!         let env_config = env_config();
//!         let step_proc_config = SimpleStepProcessorConfig {};
//!         let replay_buffer_config =
//!             SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
//!         let config = TrainerConfig::default()
//!             .max_opts(max_opts)
//!             .opt_interval(OPT_INTERVAL)
//!             .eval_interval(EVAL_INTERVAL)
//!             .record_interval(EVAL_INTERVAL)
//!             .save_interval(EVAL_INTERVAL)
//!             .eval_episodes(N_EPISODES_PER_EVAL)
//!             .model_dir(model_dir);
//!         let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
//!             config,
//!             env_config,
//!             None,
//!             step_proc_config,
//!             replay_buffer_config,
//!         );
//!
//!         trainer
//!     };
//!
//!     let mut recorder = TensorboardRecorder::new(model_dir);
//!     let mut agent = create_agent(DIM_OBS, DIM_ACT);
//!
//!     trainer.train(&mut agent, &mut recorder)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Evaluation
//!
//! When evaluating the trained agent, the parameters of the model is loaded
//! with [`Agent::load()`] method. Then [`Agent::eval()`] method is called to
//! make sure that the agent is in evaluation mode.
//!
//! In this code, the states of the environment recorded during evaluation are
//! saved as a CSV file.
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     replay_buffer::{
//! #         SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
//! #         SimpleStepProcessorConfig, SubBatch,
//! #     },
//! #     util, Agent, Env as _, Policy, Trainer, TrainerConfig,
//! # };
//! # use border_py_gym_env::{
//! #     PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
//! #     PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
//! # };
//! # use border_tch_agent::{
//! #     dqn::{Dqn, DqnConfig, DqnModelConfig},
//! #     mlp::{Mlp, MlpConfig},
//! #     TensorSubBatch,
//! # };
//! # use clap::{App, Arg};
//! # use csv::WriterBuilder;
//! # use serde::Serialize;
//! # use std::{convert::TryFrom, fs::File};
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # const DIM_ACT: i64 = 2;
//! # const LR_CRITIC: f64 = 0.001;
//! # const DISCOUNT_FACTOR: f64 = 0.99;
//! # const BATCH_SIZE: usize = 64;
//! # const N_TRANSITIONS_WARMUP: usize = 100;
//! # const N_UPDATES_PER_OPT: usize = 1;
//! # const TAU: f64 = 0.005;
//! # const OPT_INTERVAL: usize = 50;
//! # const MAX_OPTS: usize = 1000;
//! # const EVAL_INTERVAL: usize = 50;
//! # const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! # const N_EPISODES_PER_EVAL: usize = 5;
//! # const MODEL_DIR: &str = "./border/examples/model/dqn_cartpole";
//! #
//! # type PyObsDtype = f32;
//! #
//! # #[derive(Clone, Debug)]
//! # struct Obs(PyGymEnvObs<PyObsDtype, f32>);
//! #
//! # impl border_core::Obs for Obs {
//! #     fn dummy(n: usize) -> Self {
//! #         Obs(PyGymEnvObs::dummy(n))
//! #     }
//! #
//! #     fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
//! #         Obs(self.0.merge(obs_reset.0, is_done))
//! #     }
//! #
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl From<PyGymEnvObs<PyObsDtype, f32>> for Obs {
//! #     fn from(obs: PyGymEnvObs<PyObsDtype, f32>) -> Self {
//! #         Obs(obs)
//! #     }
//! # }
//! #
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         Tensor::try_from(&obs.0.obs).unwrap()
//! #     }
//! # }
//! #
//! # struct ObsBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ObsBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Obs> for ObsBatch {
//! #     fn from(obs: Obs) -> Self {
//! #         let tensor = obs.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ObsBatch> for Tensor {
//! #     fn from(b: ObsBatch) -> Self {
//! #         b.0.into()
//! #     }
//! # }
//! #
//! # #[derive(Clone, Debug)]
//! # struct Act(PyGymEnvDiscreteAct);
//! #
//! # impl border_core::Act for Act {
//! #     fn len(&self) -> usize {
//! #         self.0.len()
//! #     }
//! # }
//! #
//! # impl Into<PyGymEnvDiscreteAct> for Act {
//! #     fn into(self) -> PyGymEnvDiscreteAct {
//! #         self.0
//! #     }
//! # }
//! #
//! # struct ActBatch(TensorSubBatch);
//! #
//! # impl SubBatch for ActBatch {
//! #     fn new(capacity: usize) -> Self {
//! #         Self(TensorSubBatch::new(capacity))
//! #     }
//! #
//! #     fn push(&mut self, i: usize, data: &Self) {
//! #         self.0.push(i, &data.0)
//! #     }
//! #
//! #     fn sample(&self, ixs: &Vec<usize>) -> Self {
//! #         let buf = self.0.sample(ixs);
//! #         Self(buf)
//! #     }
//! # }
//! #
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! #
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! #
//! # impl From<Act> for ActBatch {
//! #     fn from(act: Act) -> Self {
//! #         let tensor = act.into();
//! #         Self(TensorSubBatch::from_tensor(tensor))
//! #     }
//! # }
//! #
//! # impl From<ActBatch> for Tensor {
//! #     fn from(act: ActBatch) -> Self {
//! #         act.0.into()
//! #     }
//! # }
//! #
//! # impl From<Tensor> for Act {
//! #     // `t` must be a 1-dimentional tensor of `f32`
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! # type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
//! # type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
//! # type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
//! #
//! # fn create_agent(in_dim: i64, out_dim: i64) -> Dqn<Env, Mlp, ReplayBuffer> {
//! #     let device = tch::Device::cuda_if_available();
//! #     let config = {
//! #         let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
//! #         let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim, true);
//! #         let model_config = DqnModelConfig::default()
//! #             .q_config(mlp_config)
//! #             .out_dim(out_dim)
//! #             .opt_config(opt_config);
//! #         DqnConfig::default()
//! #             .n_updates_per_opt(N_UPDATES_PER_OPT)
//! #             .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//! #             .batch_size(BATCH_SIZE)
//! #             .discount_factor(DISCOUNT_FACTOR)
//! #             .tau(TAU)
//! #             .model_config(model_config)
//! #             .device(device)
//! #     };
//! #
//! #     Dqn::build(config)
//! # }
//! #
//! # fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
//! #     PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
//! #         .name("CartPole-v0".to_string())
//! #         .obs_filter_config(ObsFilter::default_config())
//! #         .act_filter_config(ActFilter::default_config())
//! # }
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
//! fn eval(model_dir: &str, render: bool) -> Result<()> {
//!     let mut env_config = env_config();
//!     if render {
//!         env_config = env_config.render_mode(Some("human".to_string()));
//!     }
//!     let mut env = Env::build(&env_config, 0)?;
//!     let mut agent = create_agent(DIM_OBS, DIM_ACT);
//!     let mut recorder = BufferedRecorder::new();
//!     env.set_render(render);
//!     if render {
//!         env.set_wait_in_step(std::time::Duration::from_millis(10));
//!     }
//!     agent.load(model_dir)?;
//!     agent.eval();
//!
//!     let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;
//!
//!     // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
//!     let mut wtr = WriterBuilder::new()
//!         .has_headers(false)
//!         .from_writer(File::create(model_dir.to_string() + "/eval.csv")?);
//!     for record in recorder.iter() {
//!         wtr.serialize(CartpoleRecord::try_from(record)?)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! [`Dqn`]: crate::dqn::Dqn
//! [`PyGymEnvObs`]: border_py_gym_env::PyGymEnvObs
//! [`PyGymEnvObsRawFilter`]: border_py_gym_env::PyGymEnvObsRawFilter
//! [`StdBatch`]: border_core::replay_buffer::StdBatch
//! [`Tensor`]: tch::Tensor
//! [`SimpleReplayBuffer`]: border_core::replay_buffer::SimpleReplayBuffer
//! [`SubBatch`]: border_core::replay_buffer::SubBatch
//! [`PyGymEnvDiscreteAct`]: border_py_gym_env::PyGymEnvDiscreteAct
//! [`PyGymEnvDiscreteActRawFilter`]: border_py_gym_env::PyGymEnvDiscreteActRawFilter
//! [`Mlp`]: crate::mlp::Mlp
//! [`Trainer`]: border_core::Trainer
//! [`Trainer::train`]: border_core::Trainer::train
//! [`TensorboardRecorder`]: border_core::record::TensorboardRecorder
//! [`Agent::load()`]: border_core::Agent::load
//! [`Agent::eval()`]: border_core::Agent::eval
pub mod cnn;
pub mod dqn;
pub mod iqn;
pub mod mlp;
pub mod model;
pub mod opt;
pub mod sac;
mod tensor_batch;
// pub mod replay_buffer;
pub mod util;
use serde::{Deserialize, Serialize};
pub use tensor_batch::{TensorSubBatch, ZeroTensor};

#[derive(Clone, Debug, Copy, Deserialize, Serialize, PartialEq)]
/// Device for using tch-rs.
///
/// This enum is added because `tch::Device` does not support serialization.
pub enum Device {
    /// The main CPU device.
    Cpu,

    /// The main GPU device.
    Cuda(usize),
}

impl From<tch::Device> for Device {
    fn from(device: tch::Device) -> Self {
        match device {
            tch::Device::Cpu => Self::Cpu,
            tch::Device::Cuda(n) => Self::Cuda(n),
        }
    }
}

impl Into<tch::Device> for Device {
    fn into(self) -> tch::Device {
        match self {
            Self::Cpu => tch::Device::Cpu,
            Self::Cuda(n) => tch::Device::Cuda(n),
        }
    }
}
