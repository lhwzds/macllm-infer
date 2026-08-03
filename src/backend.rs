#[cfg(feature = "metal")]
use anyhow::Context;
use anyhow::Result;
use candle_core::{DType, Device};
use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DeviceKind {
    Cpu,
    Metal,
}

impl DeviceKind {
    pub fn create(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).context("failed to initialize Apple Metal device 0")
                }
                #[cfg(not(feature = "metal"))]
                {
                    anyhow::bail!("this binary was built without the `metal` feature")
                }
            }
        }
    }

    pub fn validate_dtype(self, dtype: DTypeKind) -> Result<()> {
        if self == Self::Cpu && dtype != DTypeKind::F32 {
            anyhow::bail!(
                "the CPU/Accelerate backend requires --dtype f32; {dtype:?} matmul is unsupported"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DTypeKind {
    F32,
    F16,
    Bf16,
}

impl From<DTypeKind> for DType {
    fn from(value: DTypeKind) -> Self {
        match value {
            DTypeKind::F32 => Self::F32,
            DTypeKind::F16 => Self::F16,
            DTypeKind::Bf16 => Self::BF16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_rejects_unsupported_half_precision_early() {
        assert!(DeviceKind::Cpu.validate_dtype(DTypeKind::F32).is_ok());
        assert!(DeviceKind::Cpu.validate_dtype(DTypeKind::F16).is_err());
        assert!(DeviceKind::Cpu.validate_dtype(DTypeKind::Bf16).is_err());
        assert!(DeviceKind::Metal.validate_dtype(DTypeKind::F16).is_ok());
    }
}
