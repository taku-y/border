use anyhow::Result;
use border_atari_env::{
    BorderAtariAct, BorderAtariActRawFilter, BorderAtariEnv, BorderAtariEnvConfig, BorderAtariObs,
    BorderAtariObsRawFilter,
};
use border_core::{DefaultEvaluator, Env as _, Evaluator, Policy};

type Obs = BorderAtariObs;
type Act = BorderAtariAct;
type ObsFilter = BorderAtariObsRawFilter<Obs>;
type ActFilter = BorderAtariActRawFilter<Act>;
type EnvConfig = BorderAtariEnvConfig<Obs, Act, ObsFilter, ActFilter>;
type Env = BorderAtariEnv<Obs, Act, ObsFilter, ActFilter>;

#[derive(Clone)]
struct RandomPolicyConfig {
    pub n_acts: usize,
}

struct RandomPolicy {
    n_acts: usize,
}

impl Policy<Env> for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(config: Self::Config) -> Self {
        Self {
            n_acts: config.n_acts,
        }
    }

    fn sample(&mut self, _: &Obs) -> Act {
        fastrand::u8(..self.n_acts as u8).into()
    }
}

fn env_config(name: String) -> EnvConfig {
    EnvConfig::default().name(name)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    // Pong environment configuration
    let env_config = env_config("pong".to_string()).render(true);

    // Creates a random policy
    let mut policy = {
        let policy_config = {
            let n_acts = {
                let env = Env::build(&env_config, 42)?;
                env.get_num_actions_atari()
            };
            RandomPolicyConfig {
                n_acts: n_acts as _,
            }
        };
        RandomPolicy::build(policy_config)
    };

    // Runs evaluation
    let env_config = env_config.render(true);
    let _ = DefaultEvaluator::new(&env_config, 0, 5)?.evaluate(&mut policy);

    Ok(())
}
