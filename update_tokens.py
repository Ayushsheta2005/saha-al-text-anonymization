import os
import glob
import json

files = [
    "Seq2Seq_fine_tuned_on_tab/finetune_tab.ipynb",
    "adversarial_training_pipeline_maskfill/dataset.py",
    "inference_all_combos.ipynb",
    "model_inversion/generate_dataset.py",
    "model_inversion/push_inverter_to_hub.py",
    "model_inversion/query_bart.py",
    "model_inversion_on_pipeline_maskfill/combo1_inverter_roberta_deberta.ipynb",
    "model_inversion_on_pipeline_maskfill/combo2_inverter_roberta_bart.ipynb",
    "model_inversion_on_pipeline_maskfill/combo2_inverter_roberta_bart_rtx3050.ipynb",
    "model_inversion_on_pipeline_maskfill/config.py",
    "model_inversion_on_pipeline_maskfill/generate_pairs.py",
    "model_inversion_on_pipeline_maskfill/make_notebooks.py",
    "model_inversion_on_pipeline_maskfill/upload_to_hub.py",
    "pipeline_maskfill/kaggle_evaluation.ipynb",
    "pipeline_maskfill/kaggle_evaluation.py",
    "pipeline_maskfill/kaggle_evaluation_sample.ipynb",
    "pipeline_maskfill/kaggle_notebook copy.ipynb",
    "pipeline_maskfill/kaggle_notebook.ipynb",
    "pipeline_maskfill/kaggle_notebook.py",
    "tab-finetuning/tab_finetune.ipynb",
    "tab-finetuning/tab_finetune.py",
    "tab-finetuning/tab_finetune_bart.ipynb",
    "tab-finetuning/tab_finetune_bart.py"
]

for fpath in files:
    if not os.path.exists(fpath): continue
    
    if fpath.endswith(".ipynb"):
        with open(fpath, "r", encoding="utf-8") as f:
            notebook = json.load(f)
            
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                new_source = []
                for line in source:
                    if '***HF_TOKEN***' in line:
                        new_source.append("import os\n")
                        new_source.append("from dotenv import load_dotenv\n")
                        new_source.append("load_dotenv()\n")
                        line = line.replace('\"***HF_TOKEN***\"', 'os.getenv("HF_TOKEN")')
                        line = line.replace('\'***HF_TOKEN***\'', 'os.getenv("HF_TOKEN")')
                    new_source.append(line)
                cell["source"] = new_source
                
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)
            
    elif fpath.endswith(".py"):
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        if '***HF_TOKEN***' in content:
            new_lines = []
            for line in content.split("\n"):
                if '***HF_TOKEN***' in line:
                    new_lines.append("import os")
                    new_lines.append("from dotenv import load_dotenv")
                    new_lines.append("load_dotenv()")
                    line = line.replace('\"***HF_TOKEN***\"', 'os.getenv("HF_TOKEN")')
                    line = line.replace('\'***HF_TOKEN***\'', 'os.getenv("HF_TOKEN")')
                new_lines.append(line)
            
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
