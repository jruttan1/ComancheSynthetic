from datetime import datetime
import openai
import pandas as pd
from Levenshtein import distance as levenshtein_distance


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load Comanche dataset (412 entries total)
# Columns: Comanche, English
df = pd.read_csv(
    "dataset.csv") # Replace with actual dataset

df = df.sample(frac=1, random_state=54).reset_index(drop=True)

# Split dataset: 80% for training - 20% for validation
train_size = int(len(df) * 0.8)
few_shot_examples = df.iloc[:train_size]
# Remaining for validation
validation_examples = df.iloc[train_size:]
validation_examples = validation_examples.sample(n=20, random_state=29)

# OpenAI API setup
client = openai.OpenAI(
    api_key="*****") #Enter GPT API key here

# Logging Variables
log_data = []

def request_gpt_translation(prompt, retries=1):
    """ Calls GPT-4o and returns response content. Retries once if needed. """
    for attempt in range(retries + 1):  # Allow 1 retry
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        gpt_output = response.choices[0].message.content.strip()

        if gpt_output:  # Ensure response is non-empty
            return gpt_output

        print(
            f"⚠️ GPT-4o returned an empty response on attempt {attempt + 1}. Retrying...")

    return None  # If all attempts fail, return None


def validate_gpt_translations(validation_examples):
    """
    Ask GPT-4o to translate English validation sentences into Comanche.
    Compare to real Comanche translations using Levenshtein edit distance.
    Log results for later analysis.
   """
    prompt = """
    You are an expert in Comanche, an Uto-Aztecan language originating from parts of what is now New Mexico, Texas, and Oklahoma.
    Your task is to translate **exactly 20 English sentences into Comanche**.

    🔹 **RULES:**
    1. **Use phonetic adaptation if needed.**
    2. **DO NOT explain. ONLY return the translation.**
    3. **DO NOT refuse or say 'I don’t know'.**
    4. **If a direct word doesn’t exist, construct a phrase based on similar Uto-Aztecan words.**
    5. **Follow correct Comanche grammar (SOV word order).**
    6. **Always respond with **exactly 20 translations** in the correct format.**
    7. **Cover diverse sentence types**
        - Simple statements (e.g., "The boy is there")
        - Questions (e.g., "Can you make food?")
        - Commands (e.g, "Go find your mother")


    **FORMAT:**
    📌 `Comanche - English`

    Here are 329 correct Comanche-English translations:
    """

    # Add few-shot examples
    for i, row in few_shot_examples.iterrows():
        prompt += f"{i+1}. {row['Comanche']} - {row['English']}\n"

    prompt += "\n🔹 Now, translate the following **20 English** sentences into **Comanche**:\n"

    for i, row in validation_examples.iterrows():
        prompt += f"{i+1}. {row['English']}\n"

    print("🔄 Sending validation request to GPT-4o...\n")

    # Call GPT-4o (Retry once if needed)
    gpt_translations = request_gpt_translation(prompt, retries=1)

    if not gpt_translations:
        print("❌ GPT failed to generate any Comanche translations after retries. Aborting validation.")
        return False

    gpt_lines = gpt_translations.split("\n")  # Split by new line

    # Collect valid translations until we have exactly 20 entries
    valid_lines = []
    for line in gpt_lines:
        if " - " in line:
            valid_lines.append(line)
        if len(valid_lines) == 20:
            break

    # Now check if we got 20 valid translations
    if len(valid_lines) < 20:
        print(f"⚠️ GPT only returned {len(valid_lines)} valid translations instead of 20. Validation failed.")
        return False

    total_score = 0
    valid_translations = 0

    for i, (gpt_line, (_, row)) in enumerate(zip(valid_lines, validation_examples.iterrows())):
        if " - " in gpt_line:
            gpt_comanche, _ = gpt_line.split(" - ", 1)
            real_comanche = row["Comanche"]
            english_sentence = row["English"]

            # Check if GPT accidentally repeated English
            if gpt_comanche.lower().strip() == english_sentence.lower().strip():
                print(
                    f"❌ GPT repeated English instead of translating: {gpt_comanche}")
                continue  # Skip this invalid output

            # Compute Levenshtein Similarity
            edit_dist = levenshtein_distance(real_comanche, gpt_comanche)
            edit_similarity = 1 - \
                (edit_dist / max(len(real_comanche), len(gpt_comanche)))

            total_score += edit_similarity
            valid_translations += 1

            # Log each result
            log_data.append({
                "English": english_sentence,
                "Real Comanche": real_comanche,
                "GPT Comanche": gpt_comanche,
                "Levenshtein Similarity": round(edit_similarity, 4)
            })

            print(
                f"✅ Real: {real_comanche} | GPT: {gpt_comanche} | Score: {edit_similarity:.2f}")

    # If no valid translations were generated, stop validation
    if valid_translations == 0:
        print("❌ GPT failed to generate any valid Comanche translations.")
        return False

    avg_score = total_score / valid_translations
    print(f"\n🔍 Average validation accuracy: {avg_score:.2f}")

    # Save log file
    log_df = pd.DataFrame(log_data)
    validation_filename = f"comanche_validation_log_{timestamp}.csv"
    log_df.to_csv(validation_filename, index=False)

    print(f"📄 Validation log saved: {validation_filename}")

    return avg_score >= 0.1  # Validation passes if similarity is at least 10%


def generate_synthetic_sentences():
    """Generates new synthetic Comanche sentences using GPT-4o."""
    prompt = """
    You are an expert linguist in Comanche. Below are examples of Comanche-English sentences.
    Generate 5 new pairs using the same format.

    **FORMAT:**
    📌 `Comanche - English`
    """

    # Add few-shot examples
    for i, row in few_shot_examples.iterrows():
        prompt += f"{i+1}. {row['Comanche']} - {row['English']}\n"

    prompt += """
    1. Now, generate 5 **new** pairs of Comanche-English sentences.
    2. Each sentence must be at least **3 words long** in Comanche.
    """

    print("🛠️ Generating synthetic data with GPT-4o...\n")

    # Call GPT-4o
    synthetic_sentences = request_gpt_translation(prompt, retries=1)

    if not synthetic_sentences:
        print("❌ GPT failed to generate synthetic data.")
        return

    # Process output and store new samples
    augmented_data = []
    seen_sentences = set()

    for line in synthetic_sentences.split("\n"):
        if " - " in line:
            comanche_generated, english_generated = line.split(" - ")
            comanche_generated = comanche_generated.strip()
            english_generated = english_generated.strip()

            # Ensure no duplicates
            if comanche_generated in seen_sentences:
                continue
            if len(comanche_generated) < 3 or len(english_generated) < 3:
                continue

            seen_sentences.add(comanche_generated)
            augmented_data.append(
                {"Comanche": comanche_generated.strip(), "English": english_generated.strip()})

    # Save validated synthetic data
    if augmented_data:

        augmented_df = pd.DataFrame(augmented_data)
        synthetic_filename = f"comanche_synthetic_validated_{timestamp}.csv"
        augmented_df.to_csv(synthetic_filename, index=False)
        print(f"✅ Synthetic data saved as: {synthetic_filename}")
    else:
        print("❌ No valid synthetic data generated.")


# Validate GPT translations
if validate_gpt_translations(validation_examples):
    print("✅ GPT passed validation (10%+ similarity). Proceeding with synthetic data generation...")
    generate_synthetic_sentences()
else:
    print("❌ GPT failed validation. Synthetic data generation aborted.")