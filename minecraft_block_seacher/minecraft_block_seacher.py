import json
import os
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import requests
from io import BytesIO

def get_dominant_color(img: Image.Image):
    data = img.getdata()
    shape = img.size
    pixels = [d for d in data if d[3] > 0]
    pixel_area = 0.0
    if len(pixels) > 0:
        pixel_area = len(pixels) / len(data)
        r, g, b, a = np.clip(np.array([sum(color) // len(pixels) for color in zip(*pixels)])*2.5,0,255).astype(np.int32).tolist()
        return (r, g, b, a), pixel_area, shape
    return (0,0,0,0), 0.0, shape

def extract_block_data(html_content):
    exetracts = ['command_block','powder','shulker_box', 'tnt', 'piston', 'observer', 'sand', 'ice', 'leaves']
    img_dir = "images/"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table', attrs={"data-description": "Block IDs"})
    
    if not table:
        raise ValueError("Table with data-description='Block IDs' not found")
    
    block_data = {}
    
    for row in table.find_all('tr')[1:]:  # Skip header row
        columns = row.find_all('td')
        if len(columns) >= 3:
            image_tag = columns[0].find('img')
            resource_location = columns[1].text.strip()
            block_name = columns[2].text.strip()
            print(f"{block_name} loaded!")
            block_data[resource_location] = {
                'name' : block_name
            }
            if image_tag:
                image_url = 'https://minecraft.wiki' + image_tag['src']
                image_path = os.path.join(img_dir,image_tag['src'].split('/')[-2])
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    img = img.convert('RGBA')
                else:
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGBA')
                    img.save(image_path,format='PNG',quality=100)
                dominant_color, area, shape = get_dominant_color(img)
                # 0.767 < < 0.814
                is_block = (shape[0] == shape[1]) and (0.7411 < area < 0.8178) and (not any([resource_location.endswith(et) for et in exetracts]))
                print(f"color: {dominant_color}, area: {area}, is_block?: {is_block}, shape: {shape}")
                if area > 0.0:
                    block_data[resource_location].update(color=dominant_color)
                    block_data[resource_location].update(is_block=is_block)
                    block_data[resource_location].update(im_shape=shape)
                    block_data[resource_location].update(area=round(area*1000)/1000)
                
    return block_data

def download_html(url, filename):
    response = requests.get(url)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    return response.text

import json


def generate_stub(json_file: str, output_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f:
        f.write("from enum import Enum\n\n")
        f.write("class MBlocks(Enum):\n")
        for key,value in data.items():
            f.write(f"\t{key} = {value}\n")
        f.write("\n\tdef __str__(self):\n")
        f.write("\t\treturn self.name.lower()\n")

def main():
    url = "https://minecraft.wiki/w/Java_Edition_data_values/Blocks"
    filename = "minecraft_blocks.html"

    if os.path.exists(filename):
        print(f"Loading HTML from existing file: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        print(f"Downloading HTML from {url}")
        html_content = download_html(url, filename)

    print("Extracting block data...")
    try:
        block_data = extract_block_data(html_content)
    except ValueError as e:
        print(f"Error: {e}")
        return

    output_filename = 'minecraft_blocks.json'
    print(f"Saving extracted data to {output_filename}")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(block_data, f, ensure_ascii=False, indent=2)

    print(f"Data has been extracted and saved to {output_filename}")
    print(f"Total blocks extracted: {len(block_data)}")
    generate_stub('minecraft_blocks.json', '../NBT/MBlocks.py')

if __name__ == "__main__":
    main()