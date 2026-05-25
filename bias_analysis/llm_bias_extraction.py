import os
import json
import time
import ast

import typer
from loguru import logger
from dotenv import load_dotenv

from google import genai
from google.genai import types

from bias_analysis.dataset import get_base_dataset
from bias_analysis.config import get_election_paths

app = typer.Typer()

SYSTEM_INSTRUCTION = """You are an expert cognitive scientist, psychometrician, and survey methodology analyst specializing in cognitive bias detection within public discourse and crowdsourced polls. 

Your task is to analyze a batch of social media poll data provided in a JSON array. For each poll object, evaluate both the text prompt (how the question is framed) and the poll options (how responses are structured) to detect the presence of any of the following 6 cognitive biases:

1. Confirmation Bias: The poll text or structure seeks out, interprets, or primes information in a way that validates existing opinions/beliefs, often disregarding contradictory evidence or alternatives.
2. Anchoring: The poll text heavily relies on or introduces an initial piece of information (the "anchor"), disproportionately influencing how a respondent might value subsequent choices.
3. Availability Heuristic: The question prompts judgments based on immediate, sensational, or highly memorable events/information currently in public memory rather than objective data.
4. Social Desirability Bias: The poll options or phrasing pressure respondents to choose options that will be viewed favorably by others, over-reporting "good behavior" or under-reporting undesirable behavior.
5. Acquiescence Bias: The question or statement induces "yea-saying," nudging the respondent to passively agree with a premise (e.g., "Do you agree that...?") regardless of their true view.
6. Demand Characteristics Bias: The poll contains clear clues, leading words, or surveyor signals indicating what the surveyor "wants to hear," causing respondents to adjust answers accordingly.

### Instructions:
- Process every item in the provided JSON array.
- For each item, return its original "id".
- Identify which biases are present. If no biases are found, return an empty array for `biases_detected`.
- Do not add conversational text, introductory remarks, or markdown wrapping outside of the valid JSON object. Your output must strictly be a parseable JSON array.

### JSON Output Schema:
[
  {
    "id": <int/string matching input id>,
    "biases_detected": [
      {
        "bias_type": "Name of Bias (or 'None')",
        "confidence_score": <float between 0.0 and 1.0>,
        "evidence": "Quote string from the text or options showcasing this bias",
        "rationale": "Brief scientific explanation of why this bias applies here"
      }
    ]
  }
]"""

def process_single_batch(client: genai.Client, batch: list, outfile):
    user_prompt = f"Analyze the following batch of poll data according to your instructions. Return only the requested JSON array. Input data:\n{json.dumps(batch, indent=2)}"
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1
            ),
        )
        
        # Parse response and append to file
        results = json.loads(response.text)
        
        if len(results) != len(batch):
            logger.warning(f"Attention: llm has returned {len(results)} items instead of {len(batch)}.")
            return False

        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        outfile.flush()
        return True
            
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError during batch processing: {e}")
        try:
            # Fallback: attempt to strip markdown if the model mistakenly included it
            cleaned = response.text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            results = json.loads(cleaned.strip())
            
            if len(results) != len(batch):
                logger.warning(f"Attention: llm has returned {len(results)} items instead of {len(batch)}.")
                return False

            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            outfile.flush()
            return True
        except Exception as clean_e:
            logger.error(f"Could not recover from JSONDecodeError. Raw text starts with: {response.text[:100]!r}")
            return False
            
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        return False

@app.command()
def extract_biases(
    election: str = typer.Option("us20", help="Election code (e.g. 'us20')"),
    batch_size: int = typer.Option(20, help="Batch size for API requests"),
):
    """
    Extract cognitive biases from polls using Gemini LLM.
    """
    # Load env variables (API keys)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found. Please insert it in the .env file at the root of the project.")
        raise typer.Exit(code=1)

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60 * 1000)
    )
    paths = get_election_paths(election)
    output_jsonl_path = paths.processed_dir / "cognitive_biases.jsonl"
    
    logger.info(f"Loading polls for election [{election}] via get_base_dataset...")
    base_df = get_base_dataset(election=election)
    if base_df.empty:
        logger.warning(f"No polls found for election {election}.")
        return

    logger.info(f"Starting processing of {len(base_df)} valid records.")
    logger.info(f"Model: gemini-3.1-flash-lite | Batch Size: {batch_size}")
    
    batch = []
    total_processed = 0
    
    # Ensure the directory exists
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for idx, row in base_df.iterrows():
            poll_options = []
            try:
                # The poll_options column is saved as a string of dictionaries in get_base_dataset
                options_data = ast.literal_eval(str(row.get('poll_options', '[]')))
                poll_options = [opt.get('label', '') for opt in options_data]
            except Exception:
                pass
            
            clean_item = {
                "id": str(row.get('tweet_id')),
                "text": str(row.get('tweet_text', '')),
                "options": poll_options
            }
            batch.append(clean_item)
            
            if len(batch) == batch_size:
                success = False
                for attempt in range(3):
                    success = process_single_batch(client, batch, outfile)
                    if success:
                        break
                    logger.warning(f"Batch failed (attempt {attempt + 1}/3). Retrying in 15 seconds...")
                    time.sleep(15)
                
                if not success:
                    logger.error(f"CRITICAL: Failed to process batch after 3 attempts. Skipping {len(batch)} records.")
                
                total_processed += len(batch)
                logger.info(f"Total records processed so far: {total_processed} / {len(base_df)}")
                batch = []
                
                # Sleep to strictly respect limits
                time.sleep(4)
                
        # Process remaining
        if batch:
            success = False
            for attempt in range(3):
                success = process_single_batch(client, batch, outfile)
                if success:
                    break
                logger.warning(f"Final batch failed (attempt {attempt + 1}/3). Retrying in 15 seconds...")
                time.sleep(15)
            
            if not success:
                logger.error(f"CRITICAL: Failed to process final batch after 3 attempts. Skipping {len(batch)} records.")
                
            total_processed += len(batch)
            logger.info(f"Processing completed. Total records: {total_processed} / {len(base_df)}")

if __name__ == "__main__":
    app()
