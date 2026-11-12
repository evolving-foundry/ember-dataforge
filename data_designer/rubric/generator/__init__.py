"""Rubric generator (migrated from the RuACT refinement side; decoupled from
verl/asyncio). - :mod:`client`: Generator HTTP transport (credentials injected
independently; budget exhaustion fails explicitly). - :mod:`demonstrations`:
reviewed-case demonstration library (few-shot injection policy). - :mod:`candidates`:
candidate JSON schema, prompt construction, structural validation, and discriminative
filtering (child pass_count in [filter_low, filter_high], default [2,6]). -
:mod:`polarity`: polarity triage for negative-weight parents (convert_to_positive /
keep_negative); Generator and Judge never see rollout statistics.
"""
