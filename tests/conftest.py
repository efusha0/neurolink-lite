"""Shared pytest fixtures.

Tests never download data. Phase 1 will add synthetic ``mne.io.RawArray`` fixtures here
that mimic the EEGMMIDB layout (64 channels, 160 Hz, T0/T1/T2 annotations).
"""
