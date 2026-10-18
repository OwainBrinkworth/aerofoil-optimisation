#!/bin/bash

shopt -s globstar
pylint ./**/*.py
