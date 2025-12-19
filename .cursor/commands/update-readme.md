You are tasked with updating the README.md file for a BERT-based Named Entity Recognition (NER) project for medical symptom extraction. 

**Project Context:**
- This is a medical NER system that extracts symptoms from text using transformer models (DistilBERT, BioBERT)
- The project uses a pipeline of Jupyter notebooks for data preparation and a Python script for training
- Training uses HuggingFace Transformers with custom metrics evaluation
- The project supports multiple model architectures and hyperparameter configurations

**Your Task:**
Update the README.md to be clear, concise, and practical. The README should:

1. **Quick Start Section**: Provide clear, copy-paste commands to:
   - Install dependencies (requirements.txt)
   - Set up environment variables (.env file)
   - Run the training script (6.trainer.py)
   - Load and use a trained model
   - Etc.. more files may appear each time you are prompted to updating the readme

2. **Project Structure**: Briefly explain the pipeline flow:
   - Notebooks 1-5: Data preparation (symptom dictionary, synthetic data generation, tokenization, splits, HuggingFace Hub upload)
   - Script 6.trainer.py: Model training and evaluation
   - Key files: metrics.py, hyperparam_sets.py, label mappings

3. **How to Run**: 
   - Explain how to configure hyperparameters (hyperparam_sets.py)
   - Show the basic training command/usage
   - Mention device selection (CUDA/MPS/CPU) is automatic
   - Explain output locations (runs/ directory)

4. **Logic Explanation** (keep concise):
   - BIO tagging scheme for NER
   - How WordPiece tokenization is handled
   - Evaluation metrics (micro/macro F1, per-label metrics)
   - Model selection logic (best model based on macro F1)

5. **Key Features**:
   - Supported models (DistilBERT, BioBERT)
   - Hyperparameter configurations
   - Automatic evaluation on validation and test sets
   - Metrics visualization and saving
   - HuggingFace Hub integration

**Style Guidelines:**
- Be direct and practical, not academic or verbose
- Use code blocks for all commands and code examples
- Keep explanations brief but complete
- Focus on what users need to know to run and understand the project
- Remove any outdated information
- Add any new features or changes that have been made

**Output:**
Provide the complete updated README.md content. Ensure all sections are accurate based on the current codebase structure and functionality.

Here is the readme file: @README.md