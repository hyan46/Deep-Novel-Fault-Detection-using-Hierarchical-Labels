#!/bin/bash
parallel --jobs 800% python hypothesis1_log.py --no-pylint run --preSleep True --modelType hier --leftOut A61_PassWear_I --numEpochs 300 --weightDecay 0 --beta {3}  --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001 ::: 0.1 1 10 100
parallel --jobs 400% python hypothesis1_log.py --no-pylint run --preSleep True --modelType flat --leftOut A61_PassWear_I --numEpochs 300 --weightDecay 0 --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001
parallel --jobs 800% python hypothesis1_log.py --no-pylint run --preSleep True --modelType hier --leftOut A31_Chipmark_W --numEpochs 300 --beta {3} --weightDecay 0 --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001 ::: 0.1 1 10 100
parallel --jobs 400% python hypothesis1_log.py --no-pylint run --preSleep True --modelType flat --leftOut A31_Chipmark_W --numEpochs 300 --weightDecay 0 --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001
parallel --jobs 800% python hypothesis1_log.py --no-pylint run --preSleep True --modelType hier --leftOut A40_Crack_B --numEpochs 300 --weightDecay 0 --beta {3} --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001 ::: 0.1 1 10 100
parallel --jobs 400% python hypothesis1_log.py --no-pylint run --preSleep True --modelType flat --leftOut A40_Crack_B --numEpochs 300 --weightDecay 0 --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001
parallel --jobs 800% python hypothesis1_log.py --no-pylint run --preSleep True --modelType hier --leftOut A12_Overfill_b --numEpochs 300 --weightDecay 0 --beta {3} --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001 ::: 0.1 1 10 100
parallel --jobs 400% python hypothesis1_log.py --no-pylint run --preSleep True --modelType flat --leftOut A12_Overfill_b --numEpochs 300 --weightDecay 0 --learningRate {2} --trainSeed {1} ::: 2 3 4 ::: 0.0001
python hypothesis1_derivatives_log.py
