/// Version of the semantic RNG algorithm used by Rivet.
pub const RNG_ALGORITHM_VERSION: u32 = 1;

#[inline]
pub const fn mix_u64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline]
pub const fn combine(seed: u64, value: u64) -> u64 {
    mix_u64(seed ^ mix_u64(value))
}
