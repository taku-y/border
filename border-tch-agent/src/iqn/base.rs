//! IQN agent implemented with tch-rs.
use super::{average, IqnConfig, IqnExplorer, IqnModel, IqnSample};
use crate::{
    model::{ModelBase, SubModel},
    util::{quantile_huber_loss, track, OutDim},
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Configurable, Env, Policy, ReplayBufferBase, TransitionBatch,
};
use log::trace;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    convert::TryFrom,
    fs,
    marker::PhantomData,
    path::{Path, PathBuf},
};
use tch::{no_grad, Device, Tensor};

/// IQN agent implemented with tch-rs.
///
/// The type parameter `M` is a feature extractor, which takes
/// `M::Input` and returns feature vectors.
pub struct Iqn<E, F, M, R>
where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
    F::Config: DeserializeOwned + Serialize,
    M::Config: DeserializeOwned + Serialize,
{
    pub(in crate::iqn) soft_update_interval: usize,
    pub(in crate::iqn) soft_update_counter: usize,
    pub(in crate::iqn) n_updates_per_opt: usize,
    pub(in crate::iqn) batch_size: usize,
    pub(in crate::iqn) iqn: IqnModel<F, M>,
    pub(in crate::iqn) iqn_tgt: IqnModel<F, M>,
    pub(in crate::iqn) train: bool,
    pub(in crate::iqn) phantom: PhantomData<(E, R)>,
    pub(in crate::iqn) discount_factor: f64,
    pub(in crate::iqn) tau: f64,
    pub(in crate::iqn) sample_percents_pred: IqnSample,
    pub(in crate::iqn) sample_percents_tgt: IqnSample,
    pub(in crate::iqn) sample_percents_act: IqnSample,
    pub(in crate::iqn) explorer: IqnExplorer,
    pub(in crate::iqn) device: Device,
    pub(in crate::iqn) n_opts: usize,
}

impl<E, F, M, R> Iqn<E, F, M, R>
where
    E: Env,
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
    R: ReplayBufferBase,
    F::Config: DeserializeOwned + Serialize,
    M::Config: DeserializeOwned + Serialize + OutDim,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<F::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    fn update_critic(&mut self, buffer: &mut R) -> f32 {
        trace!("IQN::update_critic()");
        let batch = buffer.batch(self.batch_size).unwrap();
        let (obs, act, next_obs, reward, is_terminated, _is_truncated, _ixs, _weight) =
            batch.unpack();
        let obs = obs.into();
        let act = act.into().to(self.device);
        let next_obs = next_obs.into();
        let reward = Tensor::from_slice(&reward[..])
            .to(self.device)
            .unsqueeze(-1);
        let is_terminated = Tensor::from_slice(&is_terminated[..])
            .to(self.device)
            .unsqueeze(-1);

        let batch_size = self.batch_size as _;
        let n_percent_points_pred = self.sample_percents_pred.n_percent_points();
        let n_percent_points_tgt = self.sample_percents_tgt.n_percent_points();

        debug_assert_eq!(reward.size().as_slice(), &[batch_size, 1]);
        debug_assert_eq!(is_terminated.size().as_slice(), &[batch_size, 1]);
        debug_assert_eq!(act.size().as_slice(), &[batch_size, 1]);

        let loss = {
            // predictions of z(s, a), where a is from minibatch
            // pred.size() == [batch_size, 1, n_percent_points]
            let (pred, tau) = {
                let n_percent_points = n_percent_points_pred;

                // percent points
                let tau = self.sample_percents_pred.sample(batch_size).to(self.device);
                debug_assert_eq!(tau.size().as_slice(), &[batch_size, n_percent_points]);

                // predictions for all actions
                let z = self.iqn.forward(&obs, &tau);
                let n_actions = z.size()[z.size().len() - 1];
                debug_assert_eq!(
                    z.size().as_slice(),
                    &[batch_size, n_percent_points, n_actions]
                );

                // Reshape action for applying torch.gather
                let a = act.unsqueeze(1).repeat(&[1, n_percent_points, 1]);
                debug_assert_eq!(a.size().as_slice(), &[batch_size, n_percent_points, 1]);

                // takes z(s, a) with a from minibatch
                let pred = z.gather(-1, &a, false).squeeze_dim(-1).unsqueeze(1);
                debug_assert_eq!(pred.size().as_slice(), &[batch_size, 1, n_percent_points]);
                (pred, tau)
            };

            // target values with max_a q(s, a)
            // tgt.size() == [batch_size, n_percent_points, 1]
            // in theory, n_percent_points can be different with that for predictions
            let tgt = no_grad(|| {
                let n_percent_points = n_percent_points_tgt;

                // percent points
                let tau = self.sample_percents_tgt.sample(batch_size).to(self.device);
                debug_assert_eq!(tau.size().as_slice(), &[batch_size, n_percent_points]);

                // target values for all actions
                let z = self.iqn_tgt.forward(&next_obs, &tau);
                let n_actions = z.size()[z.size().len() - 1];
                debug_assert_eq!(
                    z.size().as_slice(),
                    &[batch_size, n_percent_points, n_actions]
                );

                // argmax_a z(s,a), where z are averaged over tau
                let y = z
                    .copy()
                    .mean_dim(Some([1].as_slice()), false, tch::Kind::Float);
                let a = y.argmax(-1, false).unsqueeze(-1).unsqueeze(-1).repeat(&[
                    1,
                    n_percent_points,
                    1,
                ]);
                debug_assert_eq!(a.size(), &[batch_size, n_percent_points, 1]);

                // takes z(s, a)
                let z = z.gather(2, &a, false).squeeze_dim(-1);
                debug_assert_eq!(z.size().as_slice(), &[batch_size, n_percent_points]);

                // target value
                let tgt: Tensor = reward + (1 - is_terminated) * self.discount_factor * z;
                debug_assert_eq!(tgt.size().as_slice(), &[batch_size, n_percent_points]);

                tgt.unsqueeze(-1)
            });

            let diff = tgt - pred;
            debug_assert_eq!(
                diff.size().as_slice(),
                &[batch_size, n_percent_points_tgt, n_percent_points_pred]
            );
            // need to convert diff to vec<f32>
            // buffer.update_priority(&ixs, &Some(diff));

            let tau = tau.unsqueeze(1).repeat(&[1, n_percent_points_tgt, 1]);

            quantile_huber_loss(&diff, &tau).mean(tch::Kind::Float)
        };

        self.iqn.backward_step(&loss);

        f32::try_from(loss).expect("Failed to convert Tensor to f32")
    }

    fn opt_(&mut self, buffer: &mut R) -> Record {
        let mut loss_critic = 0f32;

        for _ in 0..self.n_updates_per_opt {
            let loss = self.update_critic(buffer);
            loss_critic += loss;
        }

        self.soft_update_counter += 1;
        if self.soft_update_counter == self.soft_update_interval {
            self.soft_update_counter = 0;
            track(&mut self.iqn_tgt, &mut self.iqn, self.tau);
        }

        loss_critic /= self.n_updates_per_opt as f32;

        self.n_opts += 1;

        Record::from_slice(&[("loss_critic", RecordValue::Scalar(loss_critic))])
    }
}

impl<E, F, M, R> Policy<E> for Iqn<E, F, M, R>
where
    E: Env,
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        // Do not support vectorized env
        let batch_size = 1;

        let a = no_grad(|| {
            let action_value = average(
                batch_size,
                &obs.clone().into(),
                &self.iqn,
                &self.sample_percents_act,
                self.device,
            );

            if self.train {
                match &mut self.explorer {
                    IqnExplorer::Softmax(softmax) => softmax.action(&action_value),
                    IqnExplorer::EpsilonGreedy(egreedy) => egreedy.action(action_value),
                }
            } else {
                action_value.argmax(-1, true)
            }
        });

        a.into()
    }
}

impl<E, F, M, R> Configurable for Iqn<E, F, M, R>
where
    E: Env,
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
{
    type Config = IqnConfig<F, M>;

    /// Constructs [`Iqn`] agent.
    fn build(config: Self::Config) -> Self {
        let device = config
            .device
            .expect("No device is given for IQN agent")
            .into();
        let iqn = IqnModel::build(config.model_config, device).unwrap();
        let iqn_tgt = iqn.clone();

        Iqn {
            iqn,
            iqn_tgt,
            soft_update_interval: config.soft_update_interval,
            soft_update_counter: 0,
            n_updates_per_opt: config.n_updates_per_opt,
            batch_size: config.batch_size,
            discount_factor: config.discount_factor,
            tau: config.tau,
            sample_percents_pred: config.sample_percents_pred,
            sample_percents_tgt: config.sample_percents_tgt,
            sample_percents_act: config.sample_percents_act,
            train: config.train,
            explorer: config.explorer,
            device,
            n_opts: 0,
            phantom: PhantomData,
        }
    }
}

impl<E, F, M, R> Agent<E, R> for Iqn<E, F, M, R>
where
    E: Env + 'static,
    F: SubModel<Output = Tensor> + 'static,
    M: SubModel<Input = Tensor, Output = Tensor> + 'static,
    R: ReplayBufferBase + 'static,
    E::Obs: Into<F::Input>,
    E::Act: From<Tensor>,
    F::Config: DeserializeOwned + Serialize + Clone,
    M::Config: DeserializeOwned + Serialize + Clone + OutDim,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<F::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
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

    fn opt_with_record(&mut self, buffer: &mut R) -> Record {
        self.opt_(buffer)
    }

    fn save_params(&self, path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        let path1 = path.join("iqn.pt.tch").to_path_buf();
        let path2 = path.join("iqn_tgt.pt.tch").to_path_buf();
        self.iqn.save(&path1)?;
        self.iqn_tgt.save(&path2)?;
        Ok(vec![path1, path2])
    }

    fn load_params(&mut self, path: &Path) -> Result<()> {
        self.iqn.load(path.join("iqn.pt.tch").as_path())?;
        self.iqn_tgt.load(path.join("iqn_tgt.pt.tch").as_path())?;
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
    }
}
