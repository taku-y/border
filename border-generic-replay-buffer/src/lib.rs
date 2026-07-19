#![doc = include_str!("../README.md")]
mod batch;
#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "tch")]
pub mod tch;
mod config;
mod iw_scheduler;
mod replay_buffer;
mod step_proc;
mod sum_tree;
pub use batch::{BatchBase, GenericTransitionBatch};
pub use config::{GenericReplayBufferConfig, PerConfig};
pub use iw_scheduler::IwScheduler;
pub use replay_buffer::GenericReplayBuffer;
pub use step_proc::{SimpleStepProcessor, SimpleStepProcessorConfig};
pub use sum_tree::WeightNormalizer;

// TODO: Consider to compile this module only for tests.
/// Agent and Env for testing.
pub mod test {
    use border_core::{record::Record, Act, Agent, Configurable, Env, Info, Obs, Policy, Step};
    use serde::{Deserialize, Serialize};

    /// Obs for testing.
    #[derive(Clone, Debug)]
    pub struct TestObs {
        obs: usize,
    }

    impl Obs for TestObs {
        fn len(&self) -> usize {
            1
        }
    }

    /// Batch of obs for testing.
    pub struct TestObsBatch {
        obs: Vec<usize>,
    }

    impl crate::BatchBase for TestObsBatch {
        fn new(capacity: usize) -> Self {
            Self {
                obs: vec![0; capacity],
            }
        }

        fn push(&mut self, i: usize, data: Self) {
            self.obs[i] = data.obs[0];
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let obs = ixs.iter().map(|ix| self.obs[*ix]).collect();
            Self { obs }
        }
    }

    impl From<TestObs> for TestObsBatch {
        fn from(obs: TestObs) -> Self {
            Self { obs: vec![obs.obs] }
        }
    }

    /// Act for testing.
    #[derive(Clone, Debug)]
    pub struct TestAct {
        act: usize,
    }

    impl Act for TestAct {}

    /// Batch of act for testing.
    pub struct TestActBatch {
        act: Vec<usize>,
    }

    impl From<TestAct> for TestActBatch {
        fn from(act: TestAct) -> Self {
            Self { act: vec![act.act] }
        }
    }

    impl crate::BatchBase for TestActBatch {
        fn new(capacity: usize) -> Self {
            Self {
                act: vec![0; capacity],
            }
        }

        fn push(&mut self, i: usize, data: Self) {
            self.act[i] = data.act[0];
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let act = ixs.iter().map(|ix| self.act[*ix]).collect();
            Self { act }
        }
    }

    /// Info for testing.
    pub struct TestInfo {}

    impl Info for TestInfo {}

    /// Environment for testing.
    pub struct TestEnv {
        state_init: usize,
        state: usize,
    }

    impl Env for TestEnv {
        type Config = usize;
        type Obs = TestObs;
        type Act = TestAct;
        type Info = TestInfo;

        fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> anyhow::Result<Self::Obs> {
            self.state = self.state_init;
            Ok(TestObs { obs: self.state })
        }

        fn reset_with_index(&mut self, _ix: usize) -> anyhow::Result<Self::Obs> {
            self.state = self.state_init;
            Ok(TestObs { obs: self.state })
        }

        fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = Step {
                obs: TestObs { obs: self.state },
                act: a.clone(),
                reward: vec![0.0],
                is_terminated: vec![0],
                is_truncated: vec![0],
                info: TestInfo {},
                init_obs: Some(TestObs {
                    obs: self.state_init,
                }),
            };
            return (step, Record::empty());
        }

        fn step(&mut self, a: &TestAct) -> (Step<Self>, Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = Step {
                obs: TestObs { obs: self.state },
                act: a.clone(),
                reward: vec![0.0],
                is_terminated: vec![0],
                is_truncated: vec![0],
                info: TestInfo {},
                init_obs: Some(TestObs {
                    obs: self.state_init,
                }),
            };
            return (step, Record::empty());
        }

        fn build(config: &Self::Config, _seed: i64) -> anyhow::Result<Self>
        where
            Self: Sized,
        {
            Ok(Self {
                state_init: *config,
                state: 0,
            })
        }
    }

    type ReplayBuffer = crate::GenericReplayBuffer<TestObsBatch, TestActBatch>;

    /// Agent for testing.
    pub struct TestAgent {}

    #[derive(Clone, Deserialize, Serialize)]
    /// Config of agent for testing.
    pub struct TestAgentConfig;

    impl Agent<TestEnv, ReplayBuffer> for TestAgent {
        fn train(&mut self) {}

        fn is_train(&self) -> bool {
            false
        }

        fn eval(&mut self) {}

        fn opt_with_record(&mut self, _buffer: &mut ReplayBuffer) -> Record {
            Record::empty()
        }

        fn save_params(&self, _path: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
            Ok(vec![])
        }

        fn load_params(&mut self, _path: &std::path::Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn as_any_ref(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    impl Policy<TestEnv> for TestAgent {
        fn sample(&mut self, _obs: &TestObs) -> TestAct {
            TestAct { act: 1 }
        }
    }

    impl Configurable for TestAgent {
        type Config = TestAgentConfig;

        fn build(_config: Self::Config) -> Self {
            Self {}
        }
    }
}
