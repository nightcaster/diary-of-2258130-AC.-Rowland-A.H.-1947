import os
import cv2
import time  # <-- Import the time module
import numpy as np
import ollama
from PIL import Image, ImageEnhance

image_dir = r"C:\TEMP"
output_file = "test_transcript_multiview.md"

images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if not images:
    print("No images found!")
    exit()

print(f"Processing with Multi-View (Standard + Blue Channel) Pipeline...\n")

with open(output_file, "w", encoding="utf-8") as out:
    for img_name in images:
        if img_name in ["temp_standard.jpg", "temp_blue.jpg", "tuned_preview.jpg", "temp_page.jpg", "test_transcript.md", "test_transcript_tuned.md", "test_transcript_twopass.md", "test_transcript_smart_twopass.md", "test_transcript_multiview.md"]:
            continue

        img_path = os.path.join(image_dir, img_name)
        print(f"--- Processing: {img_name} ---")
        
        # --- START TIMER FOR THIS ITEM ---
        start_time = time.time()
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                width, height = img.size
                
                # --- CROP BOUNDARIES ---
                left = int(width * 0.25)
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
                
                # --- IMAGE 2: BLUE CHANNEL (Drops yellow paper stains, highlights ink) ---
                _, _, blue = cropped_img.split()
                blue_rgb = blue.convert('RGB')
                blue_img = ImageEnhance.Contrast(blue_rgb).enhance(1.40)
                blue_img = ImageEnhance.Sharpness(blue_img).enhance(1.50)
                blue_img.thumbnail((2048, 2048))
                blue_path = "temp_blue.jpg"
                blue_img.save(blue_path, "JPEG", quality=95)

            # ==========================================
            # MULTI-VIEW VISION TRANSCRIPTION
            # ==========================================
            print("  -> Sending multi-view images (Standard + Blue Channel) to model...")
            response = ollama.chat(
                model='qwen2.5vl:7b',
                # model='richardyoung/olmocr2:7b-q8', #Model is not as good as qwen2.5vl:7b
                messages=[{
                        'role': 'system',
                        'content': (
                            "You are an expert archivist and historian. Your task is to audit a raw OCR transcription "
                            "against the original diary image provided. \n\n"
                            "Instructions for your response:\n"
                            "1. **Audit & Re-analyze**: Look closely at the image wherever the raw text looks grammatically broken, "
                            "nonsensical, or out of place (e.g., misreading 'and went' as 'at sweet', 'Slater' as 'Slatet', or 'chit' as 'chil').\n"
                            "2. **Editorial Notes**: First, provide a short section detailing any corrections you made and why (checking context vs. pixels).\n"
                            "3. **Final Transcript**: Then, output a clean, formatted Markdown block containing the final verified verbatim transcript.\n\n"
                            # "Maintain exact line breaks and paragraph structure of the original diary."
                        )
                    },
                    {
                    'role': 'user',
                    'content': (
                        "You are an expert archivist and historian. Transcribe the handwritten WWII diary pages shown in these two images "
                        "(Image 1 is the standard enhanced view, Image 2 is the blue-channel ink isolation view). \n\n"
                        "CRITICAL TRANSCRIPTION RULES:\n"
                        "1. **Verbatim & Exact Abbreviations**: Do NOT alter, expand, or modernize abbreviations or shorthand. "
                        # "Specifically, you MUST maintain exact abbreviations like 'P.O.' (e.g., '5/- P.O.'), 'No.1', 'No.2', and currency amounts (£3, £1). Never expand 'P.O.' to 'Post Office'.\n"
                        # "2. **Accurate Proper Nouns & Phrasing**: Verify names like Slater (not Slatet), Betty, Doug, Holly, Mrs Wyke, Delhi, Globe, F. MacMurry, SMOKY. "
                        # "Ensure phrases are read accurately from the ink strokes (e.g., 'and went' NOT 'at sweet'; 'chit' not 'chil').\n"
                        "3. **Formatting & Line Breaks**: "
                        "   - Keep major headers (like date lines e.g., 'JANUARY 2 - Thursday') on their own line. "
                        "   - Remove unnecessary forced line breaks mid-paragraph. Let each paragraph flow naturally as a block of text, exactly like standard markdown paragraphs. "
                        "   - Keep standalone checklist/action lines (like 'Answered Betty's letter - No.1') on their own respective lines.\n"
                        # "Output ONLY the final clean markdown transcript."
                    ),
                    'images': [std_path, blue_path]
                }],
                options={
                    'num_ctx': 9056,
                    'num_predict': 2048,
                    'temperature': 0.0
                }
            )
            
            final_text = response['message']['content']
            print(final_text[:400] + "...\n")
            
            out.write(f"\n\n# Page: {img_name}\n\n")
            out.write(final_text)
            out.write("\n\n---\n")
            
            # --- CALCULATE AND PRINT ELAPSED TIME ---
            elapsed_time = time.time() - start_time
            print(f"--- Finished {img_name} in {elapsed_time:.2f} seconds ---\n")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Cleanup temp files
for p in ["temp_standard.jpg", "temp_blue.jpg"]:
    if os.path.exists(p):
        os.remove(p)

print(f"Multi-view transcription complete! Check output file: {output_file}")