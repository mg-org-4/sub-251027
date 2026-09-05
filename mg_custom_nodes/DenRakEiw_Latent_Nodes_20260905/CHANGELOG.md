# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-13

### Added
- **Latent Color Match Node** with multiple algorithms:
  - Cubiq-based methods using kornia (LAB, YCbCr, LUV, YUV, XYZ, RGB)
  - Advanced algorithms using color-matcher library (hm-mkl-hm, mkl, hm, reinhard, mvgd, hm-mvgd-hm)
  - Simple version for basic color matching
- **Latent Image Adjust Node** with comprehensive adjustments:
  - Brightness adjustment (-1.0 to 1.0)
  - Contrast adjustment (0.0 to 3.0)
  - Hue shifting (-180° to 180°) with HSV conversion
  - Saturation adjustment (0.0 to 3.0)
  - Sharpness control (0.0 to 3.0) with unsharp masking
- **Automatic tensor shape handling**:
  - Support for 4D and 5D tensors
  - Intelligent dimension squeezing and restoration
  - Batch processing support
- **Device management**:
  - Auto, CPU, and GPU device selection
  - Efficient memory management
  - CUDA acceleration support
- **Error handling and fallbacks**:
  - Graceful degradation when dependencies are missing
  - Automatic fallback to simpler methods
  - Comprehensive error logging

### Technical Features
- **Kornia integration** for professional color space conversions
- **Color-matcher library** support for advanced algorithms
- **Anti-aliasing and smoothing** to prevent raster artifacts
- **Non-linear factor scaling** for more intuitive effect control
- **Per-channel processing** for multi-channel latents
- **Memory-efficient batch processing**

### Performance Optimizations
- Direct latent space processing (no VAE encoding/decoding)
- GPU-accelerated operations
- Efficient tensor operations with minimal memory overhead
- Batch processing for improved throughput

### Documentation
- Comprehensive README with usage examples
- Detailed method descriptions and comparisons
- Troubleshooting guide
- Performance tips and best practices

## [Unreleased]

### Planned Features
- Additional color matching algorithms
- Real-time preview capabilities
- Advanced masking support
- Integration with other ComfyUI nodes
- Performance profiling and optimization tools

---

## Version History

- **v1.0.0**: Initial release with full feature set
- **v0.9.x**: Development and testing phases
- **v0.1.x**: Early prototypes and concept validation
