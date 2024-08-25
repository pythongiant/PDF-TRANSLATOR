import streamlit as st
import PIL.Image
import json
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import pathlib
import os

# Initialize the model (ensure you have set the API key correctly)
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key="AIzaSyDVn7JNvECmdjtNuR24CMHiUla9K00B-e4")

# Directory to save the uploaded files
media = pathlib.Path(__file__).parent / "uploaded_files"
os.makedirs(media, exist_ok=True)

# Set up the Streamlit app
st.title("PDF to Image Uploader and Translator")

# Add a dropdown for language selection
selected_language = st.selectbox(
    "Select the language for translation:",
    ["Hindi", "French", "English"]
)

# Streamlit UI to upload a PDF file
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

def add_ruler(image):
    """ Add rulers to the top and left boundaries of the image to show pixel coordinates with tick marks every 10 pixels. """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Define ruler properties
    tick_interval = 10
    ruler_color = (0, 0, 0)  # Black color for the ruler

    # Draw a horizontal ruler line at the top
    draw.line([(0, 0), (width, 0)], fill=ruler_color, width=2)
    
    # Draw vertical tick marks and labels along the top
    for x in range(0, width + 1, tick_interval):
        draw.line([(x, 0), (x, 10)], fill=ruler_color, width=1)
        draw.text((x + 2, 12), str(x), fill=ruler_color)

    # Draw a vertical ruler line on the left
    draw.line([(0, 0), (0, height)], fill=ruler_color, width=2)
    
    # Draw horizontal tick marks and labels along the left
    for y in range(0, height + 1, tick_interval):
        draw.line([(0, y), (10, y)], fill=ruler_color, width=1)
        draw.text((12, y + 2), str(y), fill=ruler_color)

    return image

def fit_text_in_box(draw, text, bbox, max_width):
    """ Adjust the font size to fit text inside the bounding box. """
    font_size = 20  # Starting font size
    font_path = "arialbd.ttf"  # Replace with the path to a valid bold font file if needed
    font = ImageFont.truetype(font_path, font_size)
    
    # Break text into lines, ensuring each line fits within the bounding box width
    lines = []
    words = text.split(' ')
    line = ''
    
    for word in words:
        test_line = f"{line} {word}".strip()
        text_bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width > max_width:
            lines.append(line)
            line = word
        else:
            line = test_line
    lines.append(line)

    # Measure text height
    text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])

    # Decrease font size until the text fits within the bounding box
    while text_width > bbox['x2'] - bbox['x1'] or text_height > bbox['y2'] - bbox['y1']:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])
        if font_size < 10:  # Minimum font size
            break
    
    return font, lines

if uploaded_pdf is not None:
    # Open the uploaded PDF file
    pdf = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

    # Iterate through the pages and save them as images
    for page_number in range(len(pdf)):
        print("Done page number ", page_number)
        page = pdf.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Add ruler to the image
        final_img = add_ruler(img)

        # Save the image as 1.jpg, 2.jpg, etc.
        img_path = os.path.join(media, f"{page_number + 1}.jpg")
        final_img.save(img_path)

        # Load the saved image
        img = PIL.Image.open(img_path)
        draw = ImageDraw.Draw(img)

        # Generate content using Gemini API
        response = model.generate_content([
            f"""
            Reply with a JSON object showing the bounding box of each text in the image, along with its translation to {selected_language}.
            The Bounding Boxes should cover entire paragraphs if possible. Transcribe the text inside the images as a paragraph as well.
            The JSON format should be:
            [
                {{
                    "bounding_box": [{{x1, y1, x2, y2}}],
                    "translation": "translated_text"
                }}
            ]
            The bounding box should be in the format [x1, y1, x2, y2]. The translation should be in {selected_language}. The bounding box should correspond to the original text that will be replaced with the translated text.
            The vertical lines contain pixel coordinates for the y axis.
            Make sure the boxes cover the entire paragraph of the text accurately.
            The horizontal lines contain pixel coordinates for the x axis.
            Use the coordinates for reference while making the bounding box. MAKE SURE THE BOUNDING BOX FITS EXACTLY WHERE THE ORIGINAL TEXT WAS.
            """,
            img
        ])

        response_data = json.loads(response.text.replace("```", '').replace('json', ''))
        
        for item in response_data:
            bbox = item["bounding_box"][0]  # Access the first bounding box in the list
            translation = item["translation"]
            print(bbox)
            # Clear the area where the original text was by filling it with white
            draw.rectangle([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])], fill="white")

            # Fit the text in the bounding box and draw it
            font, lines = fit_text_in_box(draw, translation, bbox, bbox['x2'] - bbox['x1'])
            y = bbox['y1']
            
            for line in lines:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_height = text_bbox[3] - text_bbox[1]

                # Draw the translated text in the cleared area
                text_x = bbox['x1']  # Start at the left edge of the bounding box
                draw.text((text_x, y), line, fill="black", font=font)
                y += text_height

        # Optionally, you could display the image
        st.image(img, caption=f"Page {page_number + 1}", use_column_width=True)

    st.success(f"PDF has been converted to images, translated to {selected_language}, and saved in the '{media}' directory.")
