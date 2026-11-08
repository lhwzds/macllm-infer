#[cfg(any(feature = "metal", feature = "cuda"))]
use anyhow::Context;
use anyhow::Result;
use candle_core::{DType, Device};

/// Device the runtime executes on.
///
/// The backend selectors are plain data: the command-line layer parses their
/// spelling, so this module stays independent of the argument parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda,
}

impl DeviceKind {
    /// Canonical spelling of this device on the command line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// Every accepted spelling, in command-line order.
    #[must_use]
    pub const fn all() -> [&'static str; 3] {
        ["cpu", "metal", "cuda"]
    }

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
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).context("failed to initialize CUDA device 0")
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!("this binary was built without the `cuda` feature")
                }
            }
        }
    }

    /// Reject dtype combinations the selected backend cannot run.
    ///
    /// Only the CPU/Accelerate backend is limited to F32. Metal and CUDA
    /// accept F32, F16 and BF16.
    pub fn validate_dtype(self, dtype: DTypeKind) -> Result<()> {
        if self == Self::Cpu && dtype != DTypeKind::F32 {
            anyhow::bail!(
                "the CPU/Accelerate backend requires --dtype f32; {dtype:?} matmul is unsupported"
            );
        }
        Ok(())
    }
}

/// Serving label for the backend that actually runs a model.
///
/// The label follows the live device rather than the requested one, so a
/// model loaded on an NVIDIA GPU reports `cuda` instead of falling back to
/// `cpu`.
#[must_use]
pub fn device_label(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for DeviceKind {
    type Err = BackendParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(BackendParseError::new("device", other, &Self::all())),
        }
    }
}

/// Activation dtype of the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeKind {
    F32,
    F16,
    Bf16,
}

impl DTypeKind {
    /// Canonical spelling of this dtype on the command line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// Every accepted spelling, in command-line order.
    #[must_use]
    pub const fn all() -> [&'static str; 3] {
        ["f32", "f16", "bf16"]
    }
}

impl std::fmt::Display for DTypeKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for DTypeKind {
    type Err = BackendParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            other => Err(BackendParseError::new("dtype", other, &Self::all())),
        }
    }
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

/// A backend selector that does not name a known option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendParseError {
    kind: &'static str,
    value: String,
    accepted: Vec<String>,
}

impl BackendParseError {
    fn new(kind: &'static str, value: &str, accepted: &[&str]) -> Self {
        Self {
            kind,
            value: value.to_owned(),
            accepted: accepted.iter().map(|entry| (*entry).to_owned()).collect(),
        }
    }
}

impl std::fmt::Display for BackendParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unknown {} `{}`; expected one of: {}",
            self.kind,
            self.value,
            self.accepted.join(", ")
        )
    }
}

impl std::error::Error for BackendParseError {}

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

    #[test]
    fn cuda_accepts_every_supported_dtype() {
        assert!(DeviceKind::Cuda.validate_dtype(DTypeKind::F32).is_ok());
        assert!(DeviceKind::Cuda.validate_dtype(DTypeKind::F16).is_ok());
        assert!(DeviceKind::Cuda.validate_dtype(DTypeKind::Bf16).is_ok());
    }

    #[test]
    fn parses_the_documented_backend_spellings() {
        assert_eq!("cpu".parse::<DeviceKind>(), Ok(DeviceKind::Cpu));
        assert_eq!("metal".parse::<DeviceKind>(), Ok(DeviceKind::Metal));
        assert_eq!("cuda".parse::<DeviceKind>(), Ok(DeviceKind::Cuda));
        assert_eq!("f32".parse::<DTypeKind>(), Ok(DTypeKind::F32));
        assert_eq!("f16".parse::<DTypeKind>(), Ok(DTypeKind::F16));
        assert_eq!("bf16".parse::<DTypeKind>(), Ok(DTypeKind::Bf16));
        assert!("gpu".parse::<DeviceKind>().is_err());
        assert!("f64".parse::<DTypeKind>().is_err());
        assert_eq!(DeviceKind::Metal.to_string(), "metal");
        assert_eq!(DeviceKind::Cuda.to_string(), "cuda");
        assert_eq!(DTypeKind::Bf16.to_string(), "bf16");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_device_creation_requires_the_cuda_feature() {
        let error = DeviceKind::Cuda
            .create()
            .expect_err("a build without the `cuda` feature cannot create a CUDA device");
        assert!(
            error
                .to_string()
                .contains("this binary was built without the `cuda` feature")
        );
    }

    #[test]
    fn serving_labels_name_the_cpu_backend() {
        assert_eq!(device_label(&Device::Cpu), "cpu");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn serving_labels_name_the_cuda_backend() {
        // Candle exposes its stub CUDA device without the `cuda` feature, so
        // the CUDA label is covered on machines without an NVIDIA GPU.
        assert_eq!(device_label(&Device::Cuda(candle_core::CudaDevice)), "cuda");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn serving_labels_name_the_cuda_backend_on_a_cuda_build() {
        // A build machine without an NVIDIA GPU cannot create the device;
        // the stub build covers the label mapping itself.
        if let Ok(device) = Device::new_cuda(0) {
            assert_eq!(device_label(&device), "cuda");
        }
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn serving_labels_name_the_metal_backend() {
        assert_eq!(
            device_label(&Device::Metal(candle_core::MetalDevice)),
            "metal"
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn serving_labels_name_the_metal_backend_on_a_metal_build() {
        if let Ok(device) = Device::new_metal(0) {
            assert_eq!(device_label(&device), "metal");
        }
    }
}
