# Changelog

All notable changes to the Keko project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Updated README.md hardware requirements to reflect RTX 4070 Super testing environment
- Increased recommended VRAM from 8GB+ to 12GB+

### Fixed
- Fixed all corrupted icons in README.md - replaced malformed binary characters with proper UTF-8 emoji
  - 🧠 Core Concept
  - 🏗️ Architecture
  - 📄 Documentation sections
  - 🎯 Database Schema
  - 📦 Requirements
  - 🔑 Key Features
  - 🐛 Monitoring & Debugging
  - 📚 Theoretical Framework
  - 🤝 Contributing
  - 🙏 Acknowledgments
- Converted ASCII diagrams to professional Mermaid flowcharts
  - Architecture diagram showing model flow with color-coded components
  - Column Lifecycle state diagram with transition flows
- Removed embedded null bytes and control characters from README.md
- Ensured entire README.md is valid UTF-8 encoding

### Added
- CHANGELOG.md to track project changes

## [0.1.0] - Initial Release

### Added
- Initial proof-of-concept implementation
- Progressive Neural Network (PNN) architecture with expandable columns
- Uncertainty-driven learning system
- Memory bank for episodic memory storage
- Three main implementation files:
  - `main.py` - Full PNN with memory integration
  - `pretraining.py` - Fertile ground initialization
  - `uncertain.py` - 24/7 uncertainty-driven operation
- Elastic Weight Consolidation (EWC) for continual learning
- Token hunger mechanism for context awareness
- Background training queue
- Complementarity scoring and column freezing
- Homogeneity detection to prevent redundant columns
- SQLite workflow database for experiment tracking