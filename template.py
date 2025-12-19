import logging.handlers
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name="ML_Model_Copilot"
list_of_files=[
    #".github/workflows/.gitkeep",
    f"src/{project_name}/data/raw",
    f"src/{project_name}/data/processed",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/feature_engineering.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluator.py"
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/inference_pipeline.py",
    f"src/{project_name}/genai/llm_explainer.py"
    f"src/{project_name}/genai/error_analyser.py",
    f"src/{project_name}/configs/config.yaml",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "mlruns",
    "artifacts",
    "app.py",
    f"tests/",
    f"docker/Dockerfile",
    "requirement.txt",
    "setup.py",
    
    
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory:{filedir} for file {filename}")
        
    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)):
        with open(filepath,"w") as f:
            pass
            logging.info(f"creating empty file:{filepath}")
    
    else:
        logging.info(f"{filename}: already exists")