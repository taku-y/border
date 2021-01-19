use log::trace;
use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{no_grad, Tensor};
use crate::{core::{Policy, Agent, Step, Env}};
use crate::agents::OptInterval;
use crate::agents::tch::{ReplayBuffer, TchBuffer, TchBatch};
use crate::agents::tch::model::{Model1, Model2};
use crate::agents::tch::util::track;

type ActionValue = Tensor;

struct ActionNoise {
    mu: f64,
    theta: f64,
    sigma: f64,
    state: Tensor
}

impl ActionNoise {
    pub fn new() -> Self {
        let n_procs = 1;
        Self {
            mu: 0.0,
            theta: 0.15,
            sigma: 0.1,
            state: Tensor::ones(&[n_procs, 1], tch::kind::FLOAT_CPU),
        }
    }

    pub fn update(&mut self) {
        // self.var = (self.var * 0.999).max(0.01);
    }

    pub fn apply(&mut self, t: &Tensor) -> Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(&self.state.size(), tch::kind::FLOAT_CPU);
        self.state += dx;
        t + &self.state
        //self.var * Tensor::randn(t.size().as_slice(), tch::kind::FLOAT_CPU)
    }
}

// adapted from ddpg.rs in tch-rs RL examples
pub struct DDPG<E, Q, P, O, A> where
    E: Env,
    O: TchBuffer<Item = E::Obs>,
    A: TchBuffer<Item = E::Act>,
{
    critic: Q,
    critic_tgt: Q,
    actor: P,
    actor_tgt: P,
    action_noise: ActionNoise,
    replay_buffer: ReplayBuffer<E, O, A>,
    gamma: f64,
    tau: f64,
    opt_interval: OptInterval,
    n_updates_per_opt: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    count_opt_interval: usize,
    train: bool,
    prev_obs: RefCell<Option<E::Obs>>,
    phantom: PhantomData<E>
}

impl<E, Q, P, O, A> DDPG<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = A::SubBatch> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    pub fn new(critic: Q, actor: P, replay_buffer: ReplayBuffer<E, O, A>) -> Self {
        let critic_tgt = critic.clone();
        let actor_tgt = actor.clone();
        DDPG {
            critic,
            critic_tgt,
            actor,
            actor_tgt,
            action_noise: ActionNoise::new(),
            replay_buffer,
            gamma: 0.99,
            tau: 0.005,
            opt_interval: OptInterval::Steps(1),
            n_updates_per_opt: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            count_opt_interval: 0,
            train: false,
            prev_obs: RefCell::new(None),
            phantom: PhantomData,
        }
    }

    pub fn opt_interval(mut self, v: OptInterval) -> Self {
        self.opt_interval = v;
        self
    }

    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    pub fn min_transitions_warmup(mut self, v: usize) -> Self {
        self.min_transitions_warmup = v;
        self
    }

    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn discount_factor(mut self, v: f64) -> Self {
        self.gamma = v;
        self
    }

    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    // Adapted from dqn.rs
    fn push_transition(&mut self, step: Step<E>) {
        let next_obs = step.obs;
        let obs = self.prev_obs.replace(None).unwrap();
        let reward = Tensor::of_slice(&step.reward[..]);
        let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);
        self.replay_buffer.push(
            &obs,
            &step.act,
            &reward,
            &next_obs,
            &not_done,
        );
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn update_critic(&mut self, batch: &TchBatch<E, O, A>) {
        trace!("DDPG.update_critic()");

        let loss = {
            let o = &batch.obs;
            let a = &batch.actions;
            let next_o = &batch.next_obs;
            let r = &batch.rewards;
            let not_done = &batch.not_dones;
            // trace!("obs.shape      = {:?}", o.size());
            // trace!("next_obs.shape = {:?}", next_o.size());
            // trace!("act.shape      = {:?}", a.size());
            trace!("reward.shape   = {:?}", r.size());
            trace!("not_done.shape = {:?}", not_done.size());

            let tgt = {
                let next_q = no_grad(|| {
                    let next_a = self.actor_tgt.forward(&next_o);
                    self.critic_tgt.forward(&next_o, &next_a)
                });
                trace!("         r.size(): {:?}", r.size());
                trace!("  not_done.size(): {:?}", not_done.size());
                trace!("    next_q.size(): {:?}", next_q.size());
                r + not_done * Tensor::from(self.gamma) * next_q
            };
            let pred = self.critic.forward(&o, &a);

            let pred = pred.squeeze();
            let tgt = tgt.squeeze();
            debug_assert_eq!(pred.size().as_slice(), [self.batch_size as i64]);
            debug_assert_eq!(tgt.size().as_slice(), [self.batch_size as i64]);
            trace!("      pred.size(): {:?}", pred.size());
            trace!("       tgt.size(): {:?}", tgt.size());

            // let loss = pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0);
            let diff = tgt - pred;
            let loss = (&diff * &diff).mean(tch::Kind::Float);
            trace!("    critic loss: {:?}", loss);

            loss
        };

        self.critic.backward_step(&loss);
    }

    fn update_actor(&mut self, batch: &TchBatch<E, O, A>) {
        trace!("DDPG.update_actor()");

        let loss = {
            let obs = &batch.obs;
            let act = self.actor.forward(obs);
            let loss = -self.critic.forward(obs, &act).mean(tch::Kind::Float);

            // trace!("  obs.size(): {:?}", obs.size());
            // trace!("    a.size(): {:?}", a.size());
            // trace!("log_p.size(): {:?}", log_p.size());
            // trace!(" qval.size(): {:?}", qval.size());
            trace!("  actor loss: {:?}", loss);

            // let mut stdin = io::stdin();
            // let _ = stdin.read(&mut [0u8]).unwrap();

            loss
        };

        self.actor.backward_step(&loss);
    }

    fn soft_update(&mut self) {
        track(&mut self.critic_tgt, &mut self.critic, self.tau);
        track(&mut self.actor_tgt, &mut self.actor, self.tau);
    }
}

impl<E, Q, P, O, A> Policy<E> for DDPG<E, Q, P, O, A> where
    E: Env,
    // Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = A::SubBatch> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let act = tch::no_grad(|| self.actor.forward(&obs));
        if self.train {
            // TODO: parametrize output scale; 2.0 is for pendulum env
            self.action_noise.apply(&act).clip(-2.0, 2.0).into()
        }
        else {
            act.into()
        }
    }
}

impl<E, Q, P, O, A> Agent<E> for DDPG<E, Q, P, O, A> where
    E: Env,
    Q: Model2<Input1 = O::SubBatch, Input2 = A::SubBatch, Output = ActionValue> + Clone,
    P: Model1<Output = A::SubBatch> + Clone,
    E::Obs :Into<O::SubBatch>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = P::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>,
{
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn push_obs(&self, obs: &E::Obs) {
        self.prev_obs.replace(Some(obs.clone()));
    }

    fn observe(&mut self, step: Step<E>) -> bool {
        trace!("DDPG.observe()");

        let is_done_any = step.is_done.iter().fold(0, |x, v| x + *v as i32) > 0;
        if is_done_any {
            self.action_noise.update();
        }

        // Push transition to the replay buffer
        self.push_transition(step);
        trace!("Push transition");

        // Check if doing optimization
        let do_optimize = match self.opt_interval {
            OptInterval::Steps(interval) => {
                self.count_opt_interval += 1;
                if self.count_opt_interval == interval {
                    self.count_opt_interval = 0;
                    true
                }
                else {
                    false
                }
            },
            OptInterval::Episodes(interval) => {
                if is_done_any {
                    self.count_opt_interval += 1;
                    if self.count_opt_interval == interval {
                        self.count_opt_interval = 0;
                        true
                    }
                    else {
                        false
                    }
                }
                else {
                    false
                }
            }
        } && self.replay_buffer.len() >= self.min_transitions_warmup;

        // Do optimization
        if do_optimize {
            for _ in 0..self.n_updates_per_opt {
                let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                trace!("Sample random batch");

                self.update_critic(&batch);
                self.update_actor(&batch);
                self.soft_update();
                trace!("Update models");
            };
            true
        }
        else {
            false
        }
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        fs::create_dir(&path)?;
        self.critic.save(&path.as_ref().join("critic.pt").as_path())?;
        self.critic_tgt.save(&path.as_ref().join("critic_tgt.pt").as_path())?;
        self.actor.save(&path.as_ref().join("actor.pt").as_path())?;
        self.actor_tgt.save(&path.as_ref().join("actor_tgt.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.critic.load(&path.as_ref().join("critic.pt").as_path())?;
        self.critic_tgt.load(&path.as_ref().join("critic_tgt.pt").as_path())?;
        self.actor.load(&path.as_ref().join("actor.pt").as_path())?;
        self.actor_tgt.load(&path.as_ref().join("actor_tgt.pt").as_path())?;
        Ok(())
    }
}
