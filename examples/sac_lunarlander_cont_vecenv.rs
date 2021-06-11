use anyhow::Result;
use border::try_from;
use border_core::{
    record::{BufferedRecorder, Record, TensorboardRecorder},
    shape,
    util::eval_with_recorder,
    Agent, Shape, TrainerBuilder,
};
use border_py_gym_env::{
    newtype_act_c, newtype_obs, PyVecGymEnv, PyVecGymEnvBuilder, PyGymEnvContinuousAct,
};
use border_tch_agent::{
    replay_buffer::TchTensorBuffer,
    sac::{EntCoefMode, SACBuilder},
    util::{create_actor, create_critic, CriticLoss, OptInterval},
};
use clap::{App, Arg};
use ndarray::{Array1, IxDyn};
use serde::Serialize;
use std::{convert::TryFrom, fs::File};
use tch::Tensor;

const N_PROCS: usize = 4;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_CRITICS: usize = 1;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 128;
const N_TRANSITIONS_WARMUP: usize = 1000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.001;
const ALPHA: f64 = 0.5;
// const TARGET_ENTROPY: f64 = -(ACT_DIM as f64);
// const LR_ENT_COEF: f64 = 3e-4;
const REWARD_SCALE: f32 = 1.0;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 200_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const MAX_STEPS_IN_EPISODE: usize = 1000;
const MODEL_DIR: &str = "./examples/model/sac_lunarlander_cont_vec";

shape!(ObsShape, [8]);
shape!(ActShape, [2]);
newtype_obs!(Obs, ObsFilter, ObsShape, f32, f32);
newtype_act_c!(Act, ActFilter, ActShape);

impl From<Obs> for Tensor {
    fn from(obs: Obs) -> Tensor {
        try_from(obs.0.obs).unwrap()
    }
}

impl From<Act> for Tensor {
    fn from(act: Act) -> Tensor {
        let v = act.0.act.iter().map(|e| *e as f32).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<f32>>::try_from(v).unwrap();

        // The first dimension of the action tensor is the number of processes,
        // which is 1 for the non-vectorized environment.
        t.unsqueeze(0)
    }
}

impl From<Tensor> for Act {
    /// `t` must be a 1-dimentional tensor of `f32`.
    fn from(t: Tensor) -> Self {
        // In non-vectorized environment, the batch dimension is not required, thus dropped.
        let shape = t.size()[1..]
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();
        let act: Vec<f32> = t.into();

        let act = Array1::<f32>::from(act).into_shape(IxDyn(&shape)).unwrap();

        Act(PyGymEnvContinuousAct::new(act))
    }
}

type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
type ActBuffer = TchTensorBuffer<f32, ActShape, Act>;

fn create_agent() -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(
        ObsShape::shape()[0] as _,
        ActShape::shape()[0] as _,
        LR_ACTOR,
        vec![64, 64],
        device,
    )?;
    let critics = (0..N_CRITICS)
        .map(|_| {
            create_critic(
                (ObsShape::shape()[0] + ActShape::shape()[0]) as _,
                1,
                LR_CRITIC,
                vec![64, 64],
                device,
            )
            .expect("Cannot create critic")
        })
        .collect::<Vec<_>>();

    Ok(SACBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        // .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .ent_coef_mode(EntCoefMode::Fix(ALPHA))
        .reward_scale(REWARD_SCALE)
        .critic_loss(CRITIC_LOSS)
        .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY)
        .build::<_, _, _, ObsBuffer, ActBuffer>(critics, actor, device))
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    PyVecGymEnvBuilder::default()
        .build("LunarLanderContinuous-v2", obs_filter, act_filter)
        .unwrap()
        .max_steps(Some(MAX_STEPS_IN_EPISODE)) // TODO: consider moving the method to the builder
}

// fn create_env(n_procs: usize) -> Env {
//     let obs_filter = ObsFilter { vectorized: true, ..ObsFilter::default() };
//     let act_filter = ActFilter { vectorized: true };
//     Env::new("LunarLanderContinuous-v2", n_procs, obs_filter, act_filter).unwrap()
// }

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    tch::manual_seed(42);

    let env = create_env(N_PROCS);
    let env_eval = create_env(1);
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(MAX_OPTS)
        .eval_interval(EVAL_INTERVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL);

    trainer.train();
    trainer.get_agent().save("./examples/model/sac_lunarlander_cont_vec")?;

    trainer.get_env().close();
    trainer.get_env_eval().close();
    // let mut env = create_env();
    // let mut agent = create_agent();
    // env.set_render(true);
    // agent.load("./examples/model/sac_lunarlander_cont")?;
    // agent.eval();
    // util::eval(&env, &mut agent, 5, None);

    Ok(())
}
