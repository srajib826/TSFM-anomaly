# Setup Instructions

To Run the Code you need to be in the `TSFM-anomaly\Chronos_Finetuning\rajib_work_space`

Follow these steps to create a Miniconda environment and install the required dependencies:

1. **Create the Miniconda environment** (Python 3.10):
   ```bash
   conda create -n test_env python=3.10 -y
   ```

2. **Activate the environment**:
   ```bash
   conda activate test_env
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **DATA PREPARATION**:
   ```bash
   bash run_prepare_labeled.sh
   ```

4. **MODEL FINE-TUNING**:
   ```bash
   bash run_finetune.sh
   ```