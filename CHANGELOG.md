# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-08

### Added

- Initial release: extract messages from ChatGPT JSON exports, build knowledge graphs
  with spaCy NER and TF-IDF keywords
- CLI with extract/graph/full modes
- Compact in-memory graph using plain dicts and sets (no NetworkX overhead)

[0.1.0]: https://github.com/daedalus/convo-tools/releases/tag/v0.1.0
