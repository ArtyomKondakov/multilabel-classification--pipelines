"""This constants is for the pipeline."""
import os

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH_JOIN = os.path.join(PROJECT_PATH, '..')
ABSOLUTE_PROJECT_PATH = os.path.abspath(PROJECT_PATH_JOIN)
EXPERIMENTS_PATH = os.path.join(ABSOLUTE_PROJECT_PATH, 'experiments')
