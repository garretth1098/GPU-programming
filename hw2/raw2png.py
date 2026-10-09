import argparse
import numpy as np
from PIL import Image

def convert_raw_to_png(input_path, output_path, width, height, mode):
    # Determine dtype and pixel format based on mode
    if mode == "gray":
        channels = 1
        pil_mode = "L"
        dtype = np.uint8
    elif mode == "rgba":
        channels = 4
        pil_mode = "RGBA"
        dtype = np.uint8
    else:
        raise ValueError("Unsupported mode. Use 'gray' or 'rgba'.")

    # Read raw data
    raw_data = np.fromfile(input_path, dtype=dtype)
    
    # Reshape based on channels
    if channels == 1:
        img_array = raw_data.reshape((height, width))
    else:
        img_array = raw_data.reshape((height, width, channels))

    # Save as PNG
    img = Image.fromarray(img_array, mode=pil_mode)
    img.save(output_path)
    print(f"Successfully converted {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw binary image files to PNG.")
    parser.add_argument("input", help="Path to input .raw file")
    parser.add_argument("output", help="Path to output .png file")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--mode", choices=["gray", "rgba"], default="gray", help="Pixel format")

    args = parser.parse_args()
    convert_raw_to_png(args.input, args.output, args.width, args.height, args.mode)
