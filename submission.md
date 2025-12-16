#### Briefly describe your approach
- Fast Networks Track: 
- Large Networks Track: 

#### Command to train your model from scratch
- Fast Networks Track
```bash
python -m src.main --model FastModel1 --epochs 60 --data-augmentation \\resize\_to\_64x64 --use-all-data-to-train
```
- Large Networks Track
```bash
python -m src.main --model LargeModel1 --epochs 60 --data-augmentation \\resize\_to\_64x64 --use-all-data-to-train
```