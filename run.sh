#!/bin/bash

accelerate launch python/training.py
python python/oof.py