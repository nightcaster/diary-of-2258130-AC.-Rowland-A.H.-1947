import os
import cv2
import time
import base64
import numpy as np
import shutil  # <-- Required for moving files
import anthropic
from PIL import Image, ImageEnhance

# Initialize Anthropic client (reads ANTHROPIC_API_KEY from environment)
client = anthropic.Anthropic()

image_dir = r"C:\TEMP"
processed_dir = os.path.join(image_dir, "processed")
output_file = "test_transcript_claude.md"

# Ensure the processed folder exists
os.makedirs(processed_dir, exist_ok=True)

# Get sorted images from the input directory
images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if not images:
    print("No images found!")
    exit()

print(f"Processing with Claude Multi-View Pipeline & Auto-Archive...\n")

def encode_image(image_path):
    """Helper function to convert local image to base64 for Claude API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

with open(output_file, "w", encoding="utf-8") as out:
    for img_name in images:
        # Ignore temp and output files
        if img_name in ["temp_standard.jpg", "temp_blue.jpg", "tuned_preview.jpg", "temp_page.jpg", "test_transcript.md", "test_transcript_tuned.md", "test_transcript_twopass.md", "test_transcript_smart_twopass.md", "test_transcript_multiview.md", "test_transcript_claude.md"]:
            continue

        img_path = os.path.join(image_dir, img_name)
        print(f"--- Processing: {img_name} ---")
        
        start_time = time.time()
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                width, height = img.size
                
                # --- CROP BOUNDARIES ---
                left = int(width * 0.18)
                top = int(height * 0.17)
                right = int(width * 0.83)
                bottom = int(height * 0.88)
                
                cropped_img = img.crop((left, top, right, bottom))
                
                # --- IMAGE 1: STANDARD ENHANCED ---
                std_img = ImageEnhance.Contrast(cropped_img).enhance(1.20)
                std_img = ImageEnhance.Sharpness(std_img).enhance(1.30)
                std_img.thumbnail((2048, 2048))
                std_path = "temp_standard.jpg"
                std_img.save(std_path, "JPEG", quality=95)
                
                # --- IMAGE 2: BLUE CHANNEL ---
                _, _, blue = cropped_img.split()
                blue_rgb = blue.convert('RGB')
                blue_img = ImageEnhance.Contrast(blue_rgb).enhance(1.40)
                blue_img = ImageEnhance.Sharpness(blue_img).enhance(1.50)
                blue_img.thumbnail((2048, 2048))
                blue_path = "temp_blue.jpg"
                blue_img.save(blue_path, "JPEG", quality=95)

            # Base64 encode both views
            std_base64 = encode_image(std_path)
            blue_base64 = encode_image(blue_path)

            print("  -> Sending multi-view images to Claude...")
            
            # ==========================================
            # CLAUDE API MULTI-VIEW CALL
            # ==========================================
            response = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=4096,  # Increased to allow room for thinking blocks + text output
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": std_base64,
                                },
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": blue_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are an expert archivist and historian. Transcribe the handwritten WWII diary pages shown in these two images "
                                    "(Image 1 is the standard enhanced view, Image 2 is the blue-channel ink isolation view). \n\n"
                                    "CRITICAL TRANSCRIPTION RULES:\n"
                                    "1. **Verbatim & Exact Abbreviations**: Do NOT alter, expand, or modernize abbreviations or shorthand. "
                                    "Specifically, you MUST maintain exact abbreviations like 'P.O.' (e.g., '5/- P.O.'), 'No.1', 'No.2', and currency amounts (£3, £1). Never expand 'P.O.' to 'Post Office'.\n"
                                    "2. **Accurate Proper Nouns & Phrasing**: Verify names etc. "
                                    "Ensure phrases are read accurately from the ink strokes (e.g., 'and went' out to cinema, NOT 'at sweet' or 'sweetshop'; 'chit' not 'chil').\n"
                                    "3. **Formatting & Line Breaks**: "
                                    "   - Keep major headers (like date lines e.g., 'JANUARY 2 - Thursday') on their own line. "
                                    "   - Remove unnecessary forced line breaks mid-paragraph. Let each paragraph flow naturally as a block of text, exactly like standard markdown paragraphs. "
                                    "   - Keep standalone checklist/action lines (like 'Answered Betty's letter - No.1') on their own respective lines.\n"
                                    # "Output ONLY the final clean markdown transcript."
                                )
                            }
                        ]
                    }
                ]
            )
            
            # --- SAFELY EXTRACT TEXT (Filtering out Thinking Blocks) ---
            text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
            final_text = "\n".join(text_blocks) if text_blocks else str(response.content)
            
            print(final_text[:400] + "...\n")
            
            out.write(f"\n\n# Page: {img_name}\n\n")
            out.write(final_text)
            out.write("\n\n---\n")

            # --- MOVE FILE TO PROCESSED FOLDER ---
            destination_path = os.path.join(processed_dir, img_name)
            shutil.move(img_path, destination_path)
            print(f"  -> Moved {img_name} to ./processed folder.")
            
            elapsed_time = time.time() - start_time
            print(f"--- Finished {img_name} in {elapsed_time:.2f} seconds ---\n")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Cleanup temp files
for p in ["temp_standard.jpg", "temp_blue.jpg"]:
    if os.path.exists(p):
        os.remove(p)

print(f"Claude transcription complete! Check output file: {output_file}")