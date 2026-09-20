use super::key::{OpKey, SampleKey};
use super::mix::{RNG_ALGORITHM_VERSION, combine};
use super::stream::RandomStream;

pub const DOMAIN_SAMPLER: u64 = 0x5341_4D50_4C45_5201;
pub const DOMAIN_TRANSFORM: u64 = 0x5452_414E_5346_4D01;
pub const DOMAIN_TENSOR: u64 = 0x5445_4E53_4F52_0101;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RandomDomain {
    Sampler,
    Transform,
    Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomContext {
    global_seed: u64,
    epoch: u64,
}

impl RandomContext {
    pub const fn new(global_seed: u64) -> Self {
        Self {
            global_seed,
            epoch: 0,
        }
    }

    pub const fn with_epoch(self, epoch: u64) -> Self {
        Self { epoch, ..self }
    }

    pub const fn global_seed(self) -> u64 {
        self.global_seed
    }

    pub const fn epoch(self) -> u64 {
        self.epoch
    }

    pub fn domain_seed(self, domain: RandomDomain) -> u64 {
        let domain = match domain {
            RandomDomain::Sampler => DOMAIN_SAMPLER,
            RandomDomain::Transform => DOMAIN_TRANSFORM,
            RandomDomain::Tensor => DOMAIN_TENSOR,
        };
        let mut seed = combine(RNG_ALGORITHM_VERSION as u64, self.global_seed);
        seed = combine(seed, domain);
        combine(seed, self.epoch)
    }

    pub fn sampler_seed(self) -> u64 {
        self.domain_seed(RandomDomain::Sampler)
    }

    pub fn tensor_seed(self) -> u64 {
        self.domain_seed(RandomDomain::Tensor)
    }

    pub fn stream(self, sample: SampleKey, op: OpKey) -> RandomStream {
        let mut seed = self.domain_seed(RandomDomain::Transform);
        seed = combine(seed, sample.as_u64());
        RandomStream::from_seed(combine(seed, op.as_u64()))
    }
}

#[cfg(test)]
mod tests {
    use super::{DOMAIN_SAMPLER, DOMAIN_TENSOR, DOMAIN_TRANSFORM, RandomContext, RandomDomain};
    use crate::random::{OpKey, RNG_ALGORITHM_VERSION, SampleKey};

    #[test]
    fn semantic_stream_has_a_stable_v1_golden_sequence() {
        assert_eq!(RNG_ALGORITHM_VERSION, 1);
        let context = RandomContext::new(42).with_epoch(3);
        let key = OpKey::from_parts("RandomCrop", 0);
        let mut stream = context.stream(SampleKey::from_index(7), key);
        assert_eq!(stream.next_u64(), 5_533_641_612_311_874_201);
        assert_eq!(stream.next_u64(), 14_975_349_713_323_976_143);
        assert_eq!(stream.next_u64(), 6_276_863_558_590_808_471);
    }

    #[test]
    fn domains_epoch_samples_and_ops_are_separated() {
        let context = RandomContext::new(42).with_epoch(3);
        assert_ne!(
            context.domain_seed(RandomDomain::Sampler),
            context.domain_seed(RandomDomain::Transform)
        );
        assert_ne!(
            context.domain_seed(RandomDomain::Transform),
            context.domain_seed(RandomDomain::Tensor)
        );
        assert_ne!(
            context.domain_seed(RandomDomain::Sampler),
            context.domain_seed(RandomDomain::Tensor)
        );
        assert_ne!(
            context.domain_seed(RandomDomain::Transform),
            context.with_epoch(4).domain_seed(RandomDomain::Transform)
        );

        let op = OpKey::from_parts("RandomCrop", 0);
        let other_op = OpKey::from_parts("RandomHorizontalFlip", 0);
        let first = context.stream(SampleKey::from_index(10), op).next_u64();
        let other_sample = context.stream(SampleKey::from_index(11), op).next_u64();
        let other_op = context
            .stream(SampleKey::from_index(10), other_op)
            .next_u64();
        assert_ne!(first, other_sample);
        assert_ne!(first, other_op);
    }

    #[test]
    fn public_domain_constants_are_distinct() {
        assert_ne!(DOMAIN_SAMPLER, DOMAIN_TRANSFORM);
        assert_ne!(DOMAIN_TRANSFORM, DOMAIN_TENSOR);
        assert_ne!(DOMAIN_SAMPLER, DOMAIN_TENSOR);
    }
}
