"""Standalone daily macro regime engine.

Scores the macro tape across six pillars, computes each researched stock's
statistical macro betas, and biases each name's EV (tilt + conviction +
global gate). Decoupled from the research pipeline — reads finished report
dirs only. See docs/superpowers/specs/2026-06-04-macro-regime-engine-design.md
"""
