use super::mix::combine;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleKey(u64);

impl SampleKey {
    pub const fn from_index(index: usize) -> Self {
        Self(index as u64)
    }

    pub const fn from_u64(index: u64) -> Self {
        Self(index)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpKey(u64);

impl OpKey {
    /// Hash a semantic operator kind and its occurrence without using a
    /// platform-dependent standard-library hasher.
    pub fn from_parts(kind: &str, occurrence: u32) -> Self {
        let mut value = combine(0x4F50_4B45_595F_5631, occurrence as u64);
        for byte in kind.as_bytes() {
            value = combine(value, u64::from(*byte));
        }
        Self(combine(value, kind.len() as u64))
    }

    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub fn derive(self, child: OpKey) -> OpKey {
        OpKey(combine(self.0, child.0))
    }
}
