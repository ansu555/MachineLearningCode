import json
import base64
import os
import re

def extract_from_notebook(notebook_path, images_dir, metrics_file):
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    print(f"Processing {notebook_name}...")
    
    metrics = []
    image_count = 0
    
    for cell_index, cell in enumerate(notebook['cells']):
        if 'outputs' in cell:
            for output in cell['outputs']:
                # Extract Images
                if 'data' in output and 'image/png' in output['data']:
                    image_data = output['data']['image/png']
                    if isinstance(image_data, list):
                        image_data = "".join(image_data)
                    
                    image_bytes = base64.b64decode(image_data)
                    
                    # Construct a filename
                    # We'll try to be smart about naming based on cell content if possible, 
                    # but for now, unique names are safer.
                    image_filename = f"{notebook_name}_img_{image_count}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    
                    with open(image_path, 'wb') as img_f:
                        img_f.write(image_bytes)
                    
                    print(f"  Saved image: {image_filename}")
                    image_count += 1
                
                # Extract Metrics (Text)
                text_content = ""
                if 'text' in output:
                    text_content = "".join(output['text'])
                elif 'data' in output and 'text/plain' in output['data']:
                    text_content = "".join(output['data']['text/plain'])
                
                if text_content:
                    # Simple heuristic to find lines with metrics
                    lines = text_content.split('\n')
                    for line in lines:
                        if any(k in line for k in ["RMSE", "MAE", "R2", "Accuracy", "Score", "MSE"]):
                            clean_line = line.strip()
                            if len(clean_line) < 200: # Avoid dumping huge logs
                                metrics.append({
                                    "notebook": notebook_name,
                                    "cell": cell_index,
                                    "content": clean_line
                                })

    return metrics

if __name__ == "__main__":
    notebooks_dir = "StockPricePrediction/src/main/python/com.app.stock"
    notebook_paths = [
        os.path.join(notebooks_dir, "core/Stock_Price_Prediction_Data_Visualization.ipynb"),
        os.path.join(notebooks_dir, "process/Stock_Price_Prediction_Forecasting(Next_Day's_Price).ipynb"),
        os.path.join(notebooks_dir, "process/Stock_Price_Prediction_Forecasting(Next_Week_Month_Price).ipynb")
    ]
    
    images_output_dir = "latex/images/results"
    metrics_output_file = "latex/metrics.json"
    
    os.makedirs(images_output_dir, exist_ok=True)
    
    all_metrics = []
    
    for nb_path in notebook_paths:
        if os.path.exists(nb_path):
            nb_metrics = extract_from_notebook(nb_path, images_output_dir, metrics_output_file)
            all_metrics.extend(nb_metrics)
        else:
            print(f"Warning: Notebook not found: {nb_path}")
            
    with open(metrics_output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Extraction complete. Metrics saved to {metrics_output_file}")
