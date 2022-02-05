use crate::model::{SubModel, SubModel2};
use tch::{nn, nn::Module, Device, Tensor};
use super::{MLPConfig, mlp};

#[allow(clippy::clippy::upper_case_acronyms)]
/// Multilayer perceptron.
pub struct MLP {
    config: MLPConfig,
    device: Device,
    seq: nn::Sequential,
}

impl MLP {
    fn create_net(var_store: &nn::VarStore, config: &MLPConfig) -> nn::Sequential {
        let p = &var_store.root();
        let seq = mlp("cl", var_store, config).add(nn::linear(
            p / format!("cl{}", config.units.len() + 1),
            *config.units.last().unwrap(),
            config.out_dim,
            Default::default(),
        ));

        seq
    }
}

impl SubModel for MLP {
    type Config = MLPConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Tensor {
        self.seq.forward(&x.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let device = var_store.device();
        let seq = Self::create_net(var_store, &config);

        Self {
            config,
            device,
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let config = self.config.clone();
        let device = var_store.device();
        let seq = Self::create_net(&var_store, &config);

        Self {
            config,
            device,
            seq,
        }
    }
}

impl SubModel2 for MLP {
    type Config = MLPConfig;
    type Input1 = Tensor;
    type Input2 = Tensor;
    type Output = Tensor;

    fn forward(&self, input1: &Self::Input1, input2: &Self::Input2) -> Self::Output {
        let input1: Tensor = input1.to(self.device);
        let input2: Tensor = input2.to(self.device);
        let input = Tensor::cat(&[input1, input2], -1);
        self.seq.forward(&input.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let units = &config.units;
        let in_dim = *units.last().unwrap_or(&config.in_dim);
        let out_dim = config.out_dim;
        let p = &var_store.root();
        let seq = mlp("cl", var_store, &config).add(nn::linear(
            p / format!("cl{}", units.len() + 1),
            in_dim,
            out_dim,
            Default::default(),
        ));

        Self {
            config,
            device: var_store.device(),
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let config = self.config.clone();
        let device = var_store.device();
        let seq = Self::create_net(&var_store, &config);

        Self {
            config,
            device,
            seq,
        }
    }
}
