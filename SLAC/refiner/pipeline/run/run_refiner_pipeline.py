"""
Main end-to-end refiner pipeline runner.

Pipeline:
raw file -> read -> normalize/clean -> rule chunk0 -> build refiner input
-> refiner inference -> export refined chunks
"""
