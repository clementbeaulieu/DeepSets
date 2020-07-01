# Experiments Check List

## Exp. Batch Size

### DigitSumImage.

- Batch Size = 64.

    + DONE on instance-p4. With 1st model. Validation 5/50. StepLR. Set-weight exp.

    	```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //Tests DONE with BEST and LATEST.

        BEST. ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

        LATEST. ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```
		
	+ DONE on instance-p4. With 1st model. Validation 2/10. ReduceLROnPlateau. Set-weight mean. ERROR in set-weight.

		```python3 run.py --name digitsum_image_batch64_val210_reducelronplateau --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --set-weight mean --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Test done with best but ERROR.
        //TODO test with latest.
		
	+ DONE on instance-p4. With batch handling. With 1st model. Validation 2/10. ReduceLROnPlateau. Set-weight mean.
		
        ```python3 run_batch.py --name digitsum_image_batch64_val210_reducelronplateau_batch --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50_batch --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --set-weight mean --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Test DONE with latest.
        //TODO test with best.

    + INTERRUPTED. With 2nd model (last layer dropout). Validation 2/10. With 2nd model. StepLR.
        
        ```python3 run.py --name digitsum_image_batch64_val210 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

- Batch Size = 32.

    + TO BE CONTINUED (run on google colab). With 1st model. Validation 5/50. ReduceLROnPlateau.
        ```python3 run.py --name digitsum_image_batch32 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 32 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //TODO tests.

- Batch Size = 16.

    + DONE on instance-p4. With 1st model. Validation 5/50. ReduceLROnPlateau. Set-weight exp? (not mentionned)
        
        ```python3 run.py --name digitsum_image_batch16 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 16 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //TODO tests.

- Batch Size = 8.

    + DONE on instance-p4. With 1st model. Validation 5/50. ReduceLROnPlateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 8 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Test DONE with latest.

        LATEST. ```python3 run.py --name digitsum_image_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 8 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

- Batch Size = 1.

    + DONE on instance-p4. With 1st model. Validation 5/50. ReduceLROnPLateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```
        
        //TODO tests.

        //TODO. LATEST. ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

    + DONE on instance-p4. With 1st model. Validation 5/50. StepLR. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch1_stepLR --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler StepLR --lr-decay 0.5 --tensorboard```
        
        //Tests DONE with best and latest.

        BEST. ```python3 run.py --name digitsum_image_batch1_stepLR --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler StepLR --lr-decay 0.5 --tensorboard```

        LATEST. ```python3 run.py --name digitsum_image_batch1_stepLR --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler StepLR --lr-decay 0.5 --tensorboard```

    + INTERRUPTED on instance-16cpu. With 3rd model. Full dropouts. Validation 5/50. ReduceLROnPlateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch1_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```


## Exp. Exp. High variance on training input sizes/Low variance on testing input sizes.

Training size = 2/10. Testing sizes = 2/2, 5/5, 10/10.

- Testing size = 2.

    + Batch Size = 64. //TODO

        ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 2 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

    + Batch Size = 1. //TODO

        ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 2 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

- Testing size = 5.

    + Batch Size = 64. //TODO

        ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 5 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

    + Batch Size = 1. //TODO
        
        ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 5 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

- Testing size = 10.

    + Batch Size = 64. //TODO
        
        ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 10 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

    + Batch Size = 1. //TODO

        ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 10 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```


## Exp. Low variance on training input sizes/High variance on testing input sizes.

Training size = 5/5. Testing size = 2/10, 2/20, 2/50.

### Testing 2/10.

- Training 5/5. Validation 2/10. Testing 2/10. Mean Set-weight. DONE.

    + DONE on local. Batch Size = 64. With 1st model. Validation 2/10. StepLR. Mean Set-weight (BEST ERROR ?)
        
        ```python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 2000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //TODO test with LATEST.
        //TODO test with BEST. ERROR BEST.

        LATEST.
        BEST.

    + DONE on local. Batch Size = 64. With 1st model. Validation 2/10. ReduceLROnPlateau. Mean Set-weight (BEST ERROR ?)
        
        ```python3 run.py --name digitsum_image_trainlowvar_reducelronplateau --train-type regression --val-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 50000 --dataset-size-val 5000 --workers 8 --step 20 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //TODO Test with LATEST. 
        //TODO test with BEST. ERROR BEST.

        LATEST.
        BEST.

- Training 5/5. Validation 5/5. Testing 2/10.

    - 2nd model (last layer dropout).
    
    + DONE on local. With 2nd model (last layer dropout). Validation 5/5. ReduceLROnPlateau. (ERROR BEST ?)

        ```python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

        //Test DONE with BEST. ERROR BEST.

        BEST. ```python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

    - 3rd model (all dropouts in phi).

    + DONE on local. With 3rd model (full dropouts). Batch Size = 64. Validation 5/5. ReduceLROnPlateau.
        
        ```python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //Test DONE with LATEST.

        LATEST. ```python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --test-type digitsum --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

    + DONE on instance-16cpu. With 3rd model (full dropouts). Batch Size = 8. Validation 5/5. ReduceLROnPlateau.
        
        ```python3 run.py --name digitsum_image_lowvar_fulldropouts_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 8 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //TODO Tests with BEST and LATEST.

        LATEST. //TODO. ```python3 run.py --name digitsum_image_lowvar_fulldropouts_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 8 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```


# Testing 2/10.//TODO
python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 100 --print-freq-val 10 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 1000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5

# Testing 2/20.//TODO
python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 100 --print-freq-val 10 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 20 --dataset-size-train 10000 --dataset-size-val 1000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5
