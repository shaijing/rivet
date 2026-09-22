//! Statically compiled CUDA kernels used by Rivet's CUDA backend.
//!
//! The build script compiles each `.cu` file to PTX and generates the private
//! `ptx` module below. The public module descriptors provide stable indices for
//! the runtime module cache while keeping the generated PTX representation
//! behind this crate boundary.

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Copy,
    Fill,
    Binary,
    Unary,
}

pub const ALL_IDS: [Id; 4] = [Id::Copy, Id::Fill, Id::Binary, Id::Unary];

#[derive(Debug, Clone, Copy)]
pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub const fn index(self) -> usize {
        self.index
    }

    pub const fn ptx(self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut index = 0;
    while index < ALL_IDS.len() {
        if ALL_IDS[index] as u32 == id as u32 {
            return index;
        }
        index += 1;
    }
    panic!("kernel module id not found")
}

macro_rules! module {
    ($constant:ident, $id:ident) => {
        pub const $constant: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$constant,
        };
    };
}

module!(COPY, Copy);
module!(FILL, Fill);
module!(BINARY, Binary);
module!(UNARY, Unary);

#[cfg(test)]
mod tests {
    use super::{ALL_IDS, BINARY, COPY, FILL, UNARY};

    #[test]
    fn exposes_all_baseline_modules() {
        assert_eq!(ALL_IDS.len(), 4);
        assert_eq!(COPY.index(), 0);
        assert_eq!(FILL.index(), 1);
        assert_eq!(BINARY.index(), 2);
        assert_eq!(UNARY.index(), 3);
        assert!(COPY.ptx().contains(".version"));
        assert!(FILL.ptx().contains("fill_f32"));
        assert!(BINARY.ptx().contains("add_f32"));
        assert!(UNARY.ptx().contains("neg_f32"));
    }
}
