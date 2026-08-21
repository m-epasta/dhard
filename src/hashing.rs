//! Internal hashing shared by the sharded map structures: a cheap multiply-xor hasher,
//! seeded per map so that bucket placement cannot be predicted from the public
//! algorithm alone.

use std::hash::{BuildHasher, Hasher};

pub(crate) const MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Clone, Default)]
pub(crate) struct ShardHasher {
    state: u64,
    seed: u64,
}

impl ShardHasher {
    #[inline]
    fn mix(&mut self, value: u64) {
        self.state = (self.state ^ value).wrapping_mul(MULTIPLIER);
    }
}

impl Hasher for ShardHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state ^ self.seed
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.mix(u64::from_ne_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.write_u8(i as u8);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.write_u16(i as u16);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.write_u32(i as u32);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.write_usize(i as usize);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ShardBuildHasher {
    pub(crate) seed: u64,
}

impl ShardBuildHasher {
    /// Derives an unpredictable per-map seed from a cryptographically seeded
    /// [`std::collections::hash_map::RandomState`]
    pub(crate) fn random() -> Self {
        use std::collections::hash_map::RandomState;
        Self {
            seed: RandomState::new().hash_one(0u64),
        }
    }
}

impl BuildHasher for ShardBuildHasher {
    type Hasher = ShardHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        ShardHasher {
            state: 0,
            seed: self.seed,
        }
    }
}
