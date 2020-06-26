'''***** Batch Size Experiment *****'''

    # DigitSumImage training.

#Batch Size = 64.DONE avec val size 5/50.

-- DONE. With 1st model.
python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard

-- INTERRUPTED. With last layer dropout. With 2nd model.
python3 run.py --name digitsum_image_batch64_val210 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard

#Batch Size = 32. TO BE CONTINUED avec val size 5/50. 

-- TO BE CONTINUED. With 1st model.
python3 run.py --name digitsum_image_batch32 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 32 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard

#Batch Size = 16. DONE. With 1st model and val size = 5/50.

-- DONE. With 1st model.
python3 run.py --name digitsum_image_batch16 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 16 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard

#Batch Size = 8.

-- IN PROGRESS on instance-p4. With first model. Validation 5/50.
python3 run.py --name digitsum_image_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --set-weight exp --workers 8 --step 20 --batch-size 8 --epochs 200 --lr 0.01 --wd 0.005 --scheduler ReduceLROnPlateau --lr-decay 0.5 --tensorboard

#Batch Size = 1.

python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard


    # DigitSumImage testing on best and latest epochs. To be run locally with checkpoints downloaded locally.

#Batch Size = 64. DONE. Without last layer dropout.

-- DONE. With 1st model.
python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5
python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 50 --dataset-size-train 100000 --dataset-size-val 100000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5


'''TODO testing with batch size 16'''


'''***** Generalization Accuracy and variability of data set sizes. *****'''


    # High variance input sizes. Low variance output sizes (fixed size).
    
    # Training size = 2/10. Testing size = 2/2, 5/5, 10/10.
    # Resume from latest checkpoint.

# Testing size = 2.
Batch Size = 64.
python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 2 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5
Batch Size = 1.
python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 2 --max-size-val 2 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5

# Testing size = 5.
Batch Size = 64.
python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 5 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5
Batch Size = 1.
python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 5 --max-size-val 5 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5

# Testing size = 10.
Batch Size = 64.
python3 run.py --name digitsum_image_batch64 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 10 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 64 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5
Batch Size = 1.
python3 run.py --name digitsum_image_batch1 --train-type regression --val-type digitsum --print-freq-train 1000 --print-freq-val 100 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 2 --max-size-train 10 --min-size-val 10 --max-size-val 10 --dataset-size-train 100000 --dataset-size-val 10000 --workers 8 --step 20 --batch-size 1 --epochs 200 --lr 0.01 --wd 0.005 --lr-decay 0.5


    # Low variance input sizes (fixed size). High variance output sizes.
    # Training size = 5/5. Smaller training dataset = 20k inputs. Testing size = 2/10, 2/20, 2/50.

# Batch Size = 64.

# Training 5/5. Validation 2/10 or 5/5. Testing 2/10. Mean set_MAP. DONE.
-- DONE. With 1st model. Validation 2/10.
Training. python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 2000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard
-- DONE. With 1st model. Validation 2/10.
Training. python3 run.py --name digitsum_image_trainlowvar_reducelronplateau --train-type regression --val-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 50000 --dataset-size-val 5000 --workers 8 --step 20 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard

'''TODO. Testing. Resume from best.'''

#New network (last layer dropout) and val selection with same set sizes.
-- DONE. With last layer dropout. With 2nd model. Validation 5/5.
Training. python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard

-- DONE. Testing. Resume from best. With 2nd model.
python3 run.py --name digitsum_image_lowvar --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --test --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.95 --tensorboard

#All dropouts in phi. With val 5/5 and val 2/10.

-- IN PROGRESS on local. Validation 5/5. With 3rd model.
python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard
#TODO Testing resume from best.
python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard

-- IN PROGRESS on instance-16cpu. Validation 5/5. With 3rd model. Low batch size = 8.
python3 run.py --name digitsum_image_lowvar_fulldropouts_batch8 --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --min-size-train 5 --max-size-train 5 --min-size-val 5 --max-size-val 5 --dataset-size-train 20000 --dataset-size-val 2000 --workers 8 --step 10 --batch-size 8 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard
#TODO Testing resume from best.
python3 run.py --name digitsum_image_lowvar_fulldropouts --train-type regression --val-type digitsum --test-type digitsum --print-freq-train 10 --print-freq-val 100 --dataset digitsum_image --root-dir /Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data --arch digitsum_image --model-name digitsum_image50 --resume best --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 20000 --dataset-size-val 20000 --workers 8 --step 10 --batch-size 64 --epochs 100 --scheduler ReduceLROnPlateau --lr 0.01 --wd 0.005 --lr-decay 0.5 --tensorboard


# Testing 2/10.//TODO
python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 100 --print-freq-val 10 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 10 --dataset-size-train 10000 --dataset-size-val 1000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5

# Testing 2/20.//TODO
python3 run.py --name digitsum_image_trainlowvar --train-type regression --val-type digitsum --print-freq-train 100 --print-freq-val 10 --dataset digitsum_image --test --root-dir /home/jupyter/data --arch digitsum_image --model-name digitsum_image50 --resume latest --min-size-train 5 --max-size-train 5 --min-size-val 2 --max-size-val 20 --dataset-size-train 10000 --dataset-size-val 1000 --workers 8 --step 20 --batch-size 64 --epochs 100 --lr 0.01 --wd 0.005 --lr-decay 0.5

### . ###
