# PyGFWX Project Context for AI Assistants

## Project Overview

PyGFWX is an educational Python implementation of the GFWX wavelet codec, validated against the reference C++ SDK.

- **Reference SDK**: https://www.gfwx.org/ (also included at `/gfwx-sdk`)
- **Purpose**: Learn wavelet compression by implementing a readable Python codec

## Project Guidelines

### Documentation
- Check `/notes` folder for technical documentation before starting work
- See `notes/python_overview.md` for file listing to avoid duplicating functionality
- Keep notes focused on technical information, not process documentation

### Code Style
- Use NumPy for data structures, but implement core algorithms in readable Python
- Prioritize clarity over performance
- This is a uv-managed project: use `uv run` instead of `python` directly
- Line length limit: 120 characters
- Lint with `uv run ruff check <path>`, format with `uv run ruff format <path>`

### Testing
- Check for related test files in `/tests` when modifying code
- Use test image generators from `utils/image_io.py` for test images

### Platform
- Primary development on Windows; support Linux where possible
