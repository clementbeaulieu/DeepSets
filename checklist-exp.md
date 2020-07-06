# Experiments Check List

## Exp. Batch Size

### DigitSumImage.

- Batch Size = 64.

    + DONE on instance-p4. With 1st model. Validation 5/50. StepLR. Set_mAP OK with set-weight exp.

    	```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //Tests DONE with BEST and LATEST.

        BEST. Epoch 28. ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```

        LATEST. ```python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5```
		
	+ DONE on instance-p4. With 1st model. Validation 2/10. ReduceLROnPlateau. ERROR in set_MAP ?

		```python3 run.py --name digitsum_image_batch64_val210_reducelronplateau --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --set-weight mean --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Test done with best but ERROR.
        //TODO test with latest.
		
	+ DONE on instance-p4. With batch handling. With 1st model. Validation 2/10. ReduceLROnPlateau. Set_mAP OK with Set-weight mean.
		
        ```python3 run_batch.py --name digitsum_image_batch64_val210_reducelronplateau_batch --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50_batch --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --set-weight mean --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Tests DONE with latest and BEST.

        LATEST. ```python3 run_batch.py --name digitsum_image_batch64_val210_reducelronplateau_batch --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50_batch --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --set-weight mean --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        BEST. Epoch 74. ```python3 run_batch.py --name digitsum_image_batch64_val210_reducelronplateau_batch --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50_batch --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --set-weight mean --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.001 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

    + INTERRUPTED. With 2nd model (last layer dropout). Validation 2/10. With 2nd model. StepLR. ERROR st_MAP.
        
        ```python3 run.py --name digitsum_image_batch64_val210 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

- Batch Size = 32.

    + TO BE CONTINUED (run on google colab). With 1st model. Validation 5/50. ReduceLROnPlateau.
        ```python3 run.py --name digitsum_image_batch32 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 32 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //TODO tests.

- Batch Size = 16.

    + DONE on instance-p4. With 1st model. Validation 5/50. ReduceLROnPlateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch16 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 16 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Test DONE with LATEST.

        LATEST. ```python3 run.py --name digitsum_image_batch16 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 16 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

- Batch Size = 8.

    + DONE on instance-p4. With 1st model. Validation 5/50. ReduceLROnPlateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 8 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Test DONE with latest.

        LATEST. ```python3 run.py --name digitsum_image_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 8 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

- Batch Size = 1.

    + DONE on instance-p4. With 1st model. Validation 5/50. ReduceLROnPLateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //Tests DONE with LATEST and BEST.

        LATEST. ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        BEST. (Epoch 160) ```python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

    + DONE on instance-p4. With 1st model. Validation 5/50. StepLR. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch1_stepLR --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler StepLR --lr-decay 0.5 --tensorboard```
        
        //Tests DONE with best and latest.

        BEST. (Epoch 70.) ```python3 run.py --name digitsum_image_batch1_stepLR --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler StepLR --lr-decay 0.5 --tensorboard```

        LATEST. ```python3 run.py --name digitsum_image_batch1_stepLR --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler StepLR --lr-decay 0.5 --tensorboard```

    + INTERRUPTED on instance-16cpu. With 3rd model. Full dropouts. Validation 5/50. ReduceLROnPlateau. Set-weight exp.
        
        ```python3 run.py --name digitsum_image_batch1_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```

        //TODO test LATEST.

        //TODO. LATEST. ```python3 run.py --name digitsum_image_batch1_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10000 --print-freq-val 1000 --dataset digitsum_image --test --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard```


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

    + DONE on local. Batch Size = 64. With 1st model. Validation 2/10. StepLR. Mean Set-weight.
        
        ```python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 2000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //Tests DONE with best and latest.

        LATEST. ```python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --resume latest --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```
        BEST. Epoch 6. ```python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --resume best --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

    + DONE on local. Batch Size = 64. With 1st model. Validation 2/10. ReduceLROnPlateau. ERROR set_mAP. Mean set-weight.
        
        ```python3 run.py --name digitsum_image_trainlowvar_reducelronplateau --train-type regression --val-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 50000 --dataset-size-val 5000 --workers 8 --step 20 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //TODO Test with LATEST. 
        //BEST. ERROR BEST.

        LATEST. ```python3 run.py --name digitsum_image_trainlowvar_reducelronplateau --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --resume latest --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 50000 --dataset-size-val 50000 --workers 8 --step 20 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        BEST. ERROR mean set-weight.

- Training 5/5. Validation 5/5. Testing 2/10.

    - 1st model (low number of dropouts)
    
    + DONE on local. With 1st model. Validation 5/5. ReduceLROnPlateau.

        ```python3 run.py --name digitsum_image_lowvar_model1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --set-weight mean --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

        //Tests DONE with LATEST and BEST.
        
        LATEST. ```python3 run.py --name digitsum_image_lowvar_model1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --set-weight mean --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

        BEST. (Epoch 99)```python3 run.py --name digitsum_image_lowvar_model1 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --set-weight mean --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```


    - 2nd model (last layer dropout).
    
    + DONE on local. With 2nd model (last layer dropout). Validation 5/5. ReduceLROnPlateau. (ERROR BEST set_mAP = 0) 

        ```python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

        //Test DONE with BEST.
        //ERROR BEST.

        BEST. ERROR Mean set-weight. ```python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

        LATEST.

        ```python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard```

    - 3rd model (all dropouts in phi).

    + DONE on local. With 3rd model (full dropouts). Batch Size = 64. Validation 5/5. ReduceLROnPlateau.
        
        ```python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //Test DONE with LATEST.
        //ERROR with BEST. Handled manually Epoch 84.

        LATEST. ```python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --test-type digitsum --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        BEST. Epoch 84. ```python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --test-type digitsum --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume 85 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

    + DONE on instance-16cpu. With 3rd model (full dropouts). Batch Size = 8. Validation 5/5. ReduceLROnPlateau.
        
        ```python3 run.py --name digitsum_image_lowvar_fulldropouts_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 8 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```

        //TODO Tests with BEST and LATEST.

        LATEST. //TODO. ```python3 run.py --name digitsum_image_lowvar_fulldropouts_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 8 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard```


# Testing 2/10.//TODO
python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 100 --print-freq-val 10 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 1000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5

# Testing 2/20.//TODO
python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 100 --print-freq-val 10 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 20 --dataset-size-train 10000 --dataset-size-val 1000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5
