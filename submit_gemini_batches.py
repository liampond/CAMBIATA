"""Submit Gemini batch jobs using species__file_id custom keys.

Results live on Google's servers and can be retrieved later with
download_batches.py, even after a network failure. One batch is
submitted per species (10 conditions each → 5 batches total).

Usage:
    python submit_gemini_batches.py
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from src.llm_fux.models.gemini import GeminiModel
from src.llm_fux.core.runner import PromptRunner
from src.llm_fux.config.config import get_max_tokens

load_dotenv()
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

FILES = ["Above_A", "Above_C", "Above_E", "Above_F", "Above_G",
         "Below_A", "Below_C", "Below_E", "Below_F", "Below_G"]

SPECIES_GUIDES = {
    "1st-species": "data/guides/1_first_species/Fux Rules- 1vs1 note - Schubert v1.7.txt",
    "2nd-species": "data/guides/2_second_species/Cantus 2notes vs 1 - Schubert version 2.0.txt",
    "3rd-species": "data/guides/3_third_species/Fux Rules- 3rd species - 4vs1 - Schubert v2.0.txt",
    "4th-species": "data/guides/4_fourth_species/Fux Rules- 4th species - Syncopatio - Schubert v1.7.txt",
    "5th-species": "data/guides/5_fifth_species/Fux Rules- 5th species - Florish - Schubert v2.4.txt",
}

TEMPERATURE = 1.0
BASE_DIRS = {
    "encoded": Path("data/encoded"),
    "prompts": Path("data/prompts"),
    "guides":  Path("data/guides"),
    "outputs": Path("outputs"),
}


def main():
    model = GeminiModel()

    for species, guide in SPECIES_GUIDES.items():
        batch_inputs = {}
        for file_id in FILES:
            custom_id = f"{species}__{file_id}"
            runner = PromptRunner(
                model=model,
                file_id=file_id,
                datatype="musicxml",
                context=True,
                guide=guide,
                base_dirs=BASE_DIRS,
                temperature=TEMPERATURE,
                max_tokens=get_max_tokens(),
                save=False,
            )
            batch_inputs[custom_id] = runner._build_prompt_input()

        logging.info("Submitting %s (%d conditions)...", species, len(batch_inputs))
        batch_id = model.submit_batch(batch_inputs)
        print(f"  {species}: {batch_id}")

    print("\nAll batches submitted. Run download_batches.py when they complete.")


if __name__ == "__main__":
    main()
