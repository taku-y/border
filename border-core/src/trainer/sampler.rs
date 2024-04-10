//! Samples transitions and pushes them into a replay buffer.
use crate::{record::Record, Agent, Env, ReplayBufferBase, StepProcessor};
use anyhow::Result;

/// Encapsulates sampling steps. Specifically it does the followint steps:
///
/// 1. Samples an action from the [`Agent`], apply to the [`Env`] and takes [`Step`].
/// 2. Convert [`Step`] into a transition (typically a batch) with [`StepProcessor`].
/// 3. Pushes the trainsition to [`ReplayBufferBase`].
/// 4. Count episode length and pushes to [`Record`].
///
/// [`Step`]: crate::Step
/// [`StepProcessor`]: crate::StepProcessor
pub struct Sampler<E, P>
where
    E: Env,
    P: StepProcessor<E>,
{
    env: E,
    prev_obs: Option<E::Obs>,
    step_processor: P,
    /// Number of frames for counting frames per second.
    n_frames: usize,

    /// Total time of takes n_frames.
    time: f32,

    /// Number of frames in an episode
    n_frames_in_episode: usize,
}

impl<E, P> Sampler<E, P>
where
    E: Env,
    P: StepProcessor<E>,
{
    /// Creates a sampler.
    pub fn new(env: E, step_processor: P) -> Self {
        Self {
            env,
            prev_obs: None,
            step_processor,
            n_frames: 0,
            time: 0f32,
            n_frames_in_episode: 0,
        }
    }

    /// Samples transitions and pushes them into the replay buffer.
    ///
    /// The replay buffer `R_`, to which samples will be pushed, has to accept
    /// `PushedItem` that are the same with `Agent::R`.
    pub fn sample_and_push<A, R, R_>(&mut self, agent: &mut A, buffer: &mut R_) -> Result<Record>
    where
        A: Agent<E, R>,
        R: ReplayBufferBase<PushedItem = P::Output>,
        R_: ReplayBufferBase<PushedItem = R::PushedItem>,
    {
        let now = std::time::SystemTime::now();

        // Reset environment(s) if required
        if self.prev_obs.is_none() {
            // For a vectorized environments, reset all environments in `env`
            // by giving `None` to reset() method
            self.prev_obs = Some(self.env.reset(None)?);
            self.step_processor
                .reset(self.prev_obs.as_ref().unwrap().clone());
        }

        // Sample action(s) and apply it to environment(s)
        let act = agent.sample(self.prev_obs.as_ref().unwrap());
        let (step, mut record) = self.env.step_with_reset(&act);
        let is_done = step.is_done[0] == 1; // not support vectorized env
        self.n_frames_in_episode += 1;

        // Update previouos observation
        self.prev_obs = if is_done {
            Some(step.init_obs.clone())
        } else {
            Some(step.obs.clone())
        };

        // Create and push transition(s)
        let transition = self.step_processor.process(step);
        buffer.push(transition)?;

        // Reset step processor
        if is_done {
            self.step_processor
                .reset(self.prev_obs.as_ref().unwrap().clone());
            record.insert(
                "episode_length",
                crate::record::RecordValue::Scalar(self.n_frames_in_episode as _),
            );
            self.n_frames_in_episode = 0;
        }

        // For counting FPS
        if let Ok(time) = now.elapsed() {
            self.n_frames += 1;
            self.time += time.as_millis() as f32;
        }

        Ok(record)
    }

    /// Returns frames per second, including taking action, applying it to the environment,
    /// producing transition, and pushing it into the replay buffer.
    pub fn fps(&self) -> f32 {
        self.n_frames as f32 / self.time * 1000f32
    }

    /// Reset stats for computing FPS.
    pub fn reset_fps_counter(&mut self) {
        self.n_frames = 0;
        self.time = 0f32;
    }
}
