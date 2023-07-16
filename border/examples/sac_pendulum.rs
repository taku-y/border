use anyhow::Result;
use border_core::{
    record::{/*BufferedRecorder,*/ Record, RecordValue, TensorboardRecorder},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    Agent, DefaultEvaluator, Evaluator as _, Policy, Trainer, TrainerConfig,
};
use border_derive::{Act, Obs, SubBatch};
use border_py_gym_env::{
    to_pyobj, PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvContinuousAct, PyGymEnvObs,
    PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};
use border_tch_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, Sac, SacConfig},
    TensorSubBatch,
};
use clap::{App, Arg};
// use csv::WriterBuilder;
use pyo3::PyObject;
use serde::Serialize;
use std::convert::TryFrom;

const DIM_OBS: i64 = 3;
const DIM_ACT: i64 = 1;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 128;
const N_TRANSITIONS_WARMUP: usize = 1000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 40_000;
const EVAL_INTERVAL: usize = 2_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;

type PyObsDtype = f32;

#[derive(Clone, Debug, Obs)]
struct Obs(PyGymEnvObs<PyObsDtype, f32>);

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvContinuousAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

// Custom activation filter
#[derive(Clone, Debug)]
struct ActFilter {}

impl PyGymEnvActFilter<Act> for ActFilter {
    type Config = ();

    fn build(_config: &Self::Config) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {})
    }

    fn filt(&mut self, act: Act) -> (PyObject, Record) {
        let act_filt = 2f32 * &act.0.act;
        let record = Record::from_slice(&[
            (
                "act_org",
                RecordValue::Array1(act.0.act.iter().cloned().collect()),
            ),
            (
                "act_filt",
                RecordValue::Array1(act_filt.iter().cloned().collect()),
            ),
        ]);
        (to_pyobj(act_filt), record)
    }
}

type ObsFilter = PyGymEnvObsRawFilter<PyObsDtype, f32, Obs>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Evaluator = DefaultEvaluator<Env, Sac<Env, Mlp, Mlp2, ReplayBuffer>>;

#[derive(Debug, Serialize)]
struct PendulumRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f32>,
    act_org: Vec<f32>,
    act_filt: Vec<f32>,
}

impl TryFrom<&Record> for PendulumRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record.get_array1("obs")?.to_vec(),
            act_org: record.get_array1("act_org")?.to_vec(),
            act_filt: record.get_array1("act_filt")?.to_vec(),
        })
    }
}

fn create_agent(in_dim: i64, out_dim: i64) -> Sac<Env, Mlp, Mlp2, ReplayBuffer> {
    let device = tch::Device::cuda_if_available();
    let actor_config = ActorConfig::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
        .out_dim(out_dim)
        .pi_config(MlpConfig::new(in_dim, vec![64, 64], out_dim, true));
    let critic_config = CriticConfig::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
        .q_config(MlpConfig::new(in_dim + out_dim, vec![64, 64], 1, true));
    let sac_config = SacConfig::default()
        .batch_size(BATCH_SIZE)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .actor_config(actor_config)
        .critic_config(critic_config)
        .device(device);
    Sac::build(sac_config)
}

fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("Pendulum-v1".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
}

fn train(max_opts: usize, model_dir: &str, eval_interval: usize) -> Result<()> {
    let mut trainer = {
        let env_config = env_config();
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
        let config = TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(eval_interval)
            .record_interval(eval_interval)
            .save_interval(eval_interval)
            .model_dir(model_dir);
        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            config,
            env_config,
            step_proc_config,
            replay_buffer_config,
        );

        trainer
    };
    let mut agent = create_agent(DIM_OBS, DIM_ACT);
    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut evaluator = Evaluator::new(&env_config(), 0, N_EPISODES_PER_EVAL)?;

    trainer.train(&mut agent, &mut recorder, &mut evaluator)?;

    Ok(())
}

fn eval(n_episodes: usize, render: bool, model_dir: &str) -> Result<()> {
    let env_config = {
        let mut env_config = env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        };
        env_config
    };
    let mut agent = {
        let mut agent = create_agent(DIM_OBS, DIM_ACT);
        agent.load(model_dir)?;
        agent.eval();
        agent
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, n_episodes)?.evaluate(&mut agent);

    // // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    // let mut wtr = WriterBuilder::new()
    //     .has_headers(false)
    //     .from_writer(File::create(model_dir.to_string() + "/eval.csv")?);
    // for record in recorder.iter() {
    //     wtr.serialize(PendulumRecord::try_from(record)?)?;
    // }

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("sac_pendulum")
        .version("0.1.0")
        .author("Taku Yoshioka <yoshioka@laboro.ai>")
        .arg(
            Arg::with_name("train")
                .long("train")
                .takes_value(false)
                .help("Do training only"),
        )
        .arg(
            Arg::with_name("eval")
                .long("eval")
                .takes_value(false)
                .help("Do evaluation only"),
        )
        .get_matches();

    let do_train = (matches.is_present("train") && !matches.is_present("eval"))
        || (!matches.is_present("train") && !matches.is_present("eval"));
    let do_eval = (!matches.is_present("train") && matches.is_present("eval"))
        || (!matches.is_present("train") && !matches.is_present("eval"));

    if do_train {
        train(
            MAX_OPTS,
            "./border/examples/model/sac_pendulum",
            EVAL_INTERVAL,
        )?;
    }
    if do_eval {
        eval(5, true, "./border/examples/model/sac_pendulum/best")?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_sac_pendulum() -> Result<()> {
        tch::manual_seed(42);

        let model_dir = TempDir::new("sac_pendulum")?;
        let model_dir = model_dir.path().to_str().unwrap();
        train(100, model_dir, 100)?;
        eval(1, false, (model_dir.to_string() + "/best").as_str())?;

        Ok(())
    }
}
