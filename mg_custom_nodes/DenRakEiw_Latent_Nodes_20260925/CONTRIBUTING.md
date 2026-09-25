# Contributing to ComfyUI Latent Color Tools

Thank you for your interest in contributing to ComfyUI Latent Color Tools! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information**:
   - ComfyUI version
   - Python version
   - Operating system
   - Error logs and stack traces
   - Steps to reproduce
   - Expected vs actual behavior

### Suggesting Features
1. **Check the roadmap** in issues to see if it's already planned
2. **Create a feature request** with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach
   - Any relevant examples or references

### Code Contributions

#### Getting Started
1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/DenRakEiw/Latent_Nodes
   cd ComfyUI-Latent-Color-Tools
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

2. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

#### Code Standards
- **Follow PEP 8** style guidelines
- **Use meaningful variable names** and comments
- **Add docstrings** to all functions and classes
- **Keep functions focused** and reasonably sized
- **Handle errors gracefully** with appropriate logging

#### Testing
- **Test your changes** thoroughly in ComfyUI
- **Test with different tensor shapes** (4D and 5D)
- **Test with different devices** (CPU and GPU if available)
- **Test edge cases** and error conditions
- **Include test workflows** if adding new features

#### Documentation
- **Update README.md** if adding new features
- **Update CHANGELOG.md** with your changes
- **Add inline comments** for complex logic
- **Update docstrings** for modified functions

#### Pull Request Process
1. **Ensure your code follows the standards** above
2. **Update documentation** as needed
3. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots or examples if applicable
   - Testing information

4. **Respond to feedback** promptly and make requested changes

## 🏗️ Project Structure

```
ComfyUI-Latent-Color-Tools/
├── __init__.py              # Node registration
├── latent_colormatch.py     # Color matching nodes
├── latent_adjust.py         # Image adjustment node
├── requirements.txt         # Dependencies
├── README.md               # Main documentation
├── LICENSE                 # MIT license
├── CHANGELOG.md           # Version history
├── CONTRIBUTING.md        # This file
└── pyproject.toml         # Python packaging config
```

## 🎯 Development Guidelines

### Adding New Color Matching Methods
1. **Add the method** to the `METHODS` list in `latent_colormatch.py`
2. **Implement the algorithm** in a new method
3. **Handle tensor shapes** properly (4D/5D support)
4. **Add error handling** and fallbacks
5. **Test thoroughly** with different inputs
6. **Document the method** in README.md

### Adding New Adjustment Types
1. **Add the parameter** to `INPUT_TYPES` in `latent_adjust.py`
2. **Implement the adjustment** method
3. **Ensure latent-space compatibility**
4. **Add proper value ranges** and validation
5. **Test with various parameter values**
6. **Update documentation**

### Performance Considerations
- **Minimize memory allocations** in hot paths
- **Use in-place operations** where possible
- **Batch process** when applicable
- **Profile performance** for large tensors
- **Consider GPU memory limits**

## 🐛 Debugging Tips

### Common Issues
- **Tensor shape mismatches**: Check 4D vs 5D handling
- **Device mismatches**: Ensure all tensors are on the same device
- **Memory issues**: Use batch processing for large inputs
- **NaN/Inf values**: Add proper value clamping

### Debugging Tools
- **Print tensor shapes** and value ranges
- **Use torch.isnan()** and torch.isinf() for validation
- **Monitor GPU memory** with torch.cuda.memory_summary()
- **Add debug flags** for verbose logging

## 📋 Code Review Checklist

Before submitting a pull request, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] Error handling is implemented
- [ ] Tensor shapes are handled correctly
- [ ] Device management is proper
- [ ] Memory usage is optimized
- [ ] Documentation is updated
- [ ] Changes are tested in ComfyUI
- [ ] No breaking changes (or properly documented)
- [ ] CHANGELOG.md is updated

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **ComfyUI Discord**: For real-time community support

## 🙏 Recognition

Contributors will be:
- **Listed in CHANGELOG.md** for their contributions
- **Mentioned in release notes** for significant features
- **Added to contributors list** in README.md

Thank you for helping make ComfyUI Latent Color Tools better! 🎨
