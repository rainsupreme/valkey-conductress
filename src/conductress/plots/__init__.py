"""Matplotlib figure builders for Conductress results.

matplotlib is an optional dependency (``conductress[plots]``); it is imported
inside each builder module, not here, so importing this package never requires
it. Builders are pure functions of the loaded result data; the CLI
(``conductress plot ...``) loads results and writes the PNG.
"""
