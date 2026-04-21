import os
from pathlib import Path


# From https://github.com/Lightricks/LTX-Video-Trainer
def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )
    words = clean_text.split()
    result = []
    current_length = 0

    for word in words:
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

# From https://github.com/Lightricks/LTX-Video-Trainer
def get_unique_filename(
        base: str,
        ext: str,
        prompt: str,
        seed: int,
        resolution: tuple[int, int, int],
        dir: Path,
        endswith=None,
        index_range=1000,
) -> Path:
    if seed != -1:
        base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    else:
        base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{resolution[0]}x{resolution[1]}x{resolution[2]}"

    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )
