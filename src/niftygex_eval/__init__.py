"""Evaluation layer: walk-forward harness, results store, and Stage A/B models.

Reads only feature/target Parquet and stage manifests produced by the data layer (``niftygex``); it never
writes under ``data/``.
"""
