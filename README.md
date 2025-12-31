# VHeat3D Model

## Environment Setup

### Requirements
- Python 3.9
- Required packages

### Installation
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the dataset
2. Create a `data` folder in the project root directory
3. Extract and place the dataset in the `data` folder

The file structure should look like:
```
data/IndianPine.mat
```

## Project Structure

### Main Files
- `vheat3d_model.py` - Model implementation
- `demo.py` - Demo script
- `dataset.py` - Dataset handling
- `util.py` - Utility functions

## Usage

### Training
```bash
chmod +x run_train.sh
./run_train.sh
```

### Testing
```bash
chmod +x run_test.sh
./run_test.sh
```

## License

[Add your license information here]

## Citation

[Add citation information if applicable]
