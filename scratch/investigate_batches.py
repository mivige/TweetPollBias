import os
import ast
import json
import time
import concurrent.futures
from loguru import logger
from dotenv import load_dotenv

from google import genai
from google.genai import types

from bias_analysis.dataset import get_base_dataset
from bias_analysis.llm_bias_extraction import SYSTEM_INSTRUCTION, process_single_batch

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60*1000)
    )
    
    logger.info("Caricamento dataset us20...")
    df = get_base_dataset("us20")
    
    # I due batch falliti:
    # Batch 1: da 4640 a 4659
    # Batch 2: da 7140 a 7159
    
    batch_indices = list(range(4640, 4660)) + list(range(7140, 7160))
    
    logger.info(f"Ho estratto {len(batch_indices)} record per l'indagine.")
    
    # Prepariamo gli item
    items_to_process = []
    for idx in batch_indices:
        row = df.iloc[idx]
        poll_options = []
        try:
            options_data = ast.literal_eval(str(row.get('poll_options', '[]')))
            poll_options = [opt.get('label', '') for opt in options_data]
        except Exception:
            pass
        
        clean_item = {
            "id": str(row.get('tweet_id')),
            "text": str(row.get('tweet_text', '')),
            "options": poll_options
        }
        items_to_process.append(clean_item)
    
    output_path = "data/processed/us20/cognitive_biases.jsonl"
    
    # Ora elaboriamo gli elementi SINGOLARMENTE per trovare i colpevoli e salvare i buoni
    logger.info("Avvio elaborazione 1 alla volta (batch_size=1)...")
    success_count = 0
    fail_count = 0
    
    with open(output_path, "a", encoding="utf-8") as f:
        for i, item in enumerate(items_to_process):
            logger.info(f"Elaborazione poll {i+1}/40 ID: {item['id']}")
            
            # Usiamo process_single_batch
            success = False
            for attempt in range(3):
                success = process_single_batch(client, [item], f)
                if success:
                    break
                logger.warning(f"Tentativo {attempt+1} fallito. Ritento...")
                time.sleep(15)
            
            if success:
                success_count += 1
            else:
                logger.error(f"FALLIMENTO TOTALE per poll ID {item['id']}.")
                logger.error(f"Testo: {item['text']}")
                fail_count += 1
                
            time.sleep(4) # Rispetto rate limit
            
    logger.info(f"Completato! Successi: {success_count}, Fallimenti: {fail_count}")

if __name__ == '__main__':
    main()
