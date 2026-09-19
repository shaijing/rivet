/// Logical device locations. CPU is the only implemented backend for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        self == rhs
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}
