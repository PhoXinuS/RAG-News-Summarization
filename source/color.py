def print_color_table():
    """Print all available ANSI colors with their codes."""
    # Standard colors (0-15)
    print("Standard and Bright Colors (0-15):")
    for i in range(16):
        print(f"\033[38;5;{i}m{i:3d}\033[0m", end=" ")
    print("\n")

    # 216 colors (16-231)
    print("216 Colors (16-231):")
    for i in range(16, 232, 6):
        for j in range(6):
            code = i + j
            print(f"\033[38;5;{code}m{code:3d}\033[0m", end=" ")
        print()
    print()

    # Grayscale (232-255)
    print("Grayscale (232-255):")
    for i in range(232, 256):
        print(f"\033[38;5;{i}m{i:3d}\033[0m", end=" ")
    print("\n")

def print_colored(text, color):
    """
    Print text with ANSI 256 color codes.

    Args:
        text (str): Text to print
        color (int/str): Color number (0-255) or name
    """
    colors = {
        'yellow': 227,
        'blue': 33,
        'green': 46,
        'red': 196,
        'reset': 0,
        'response': 117,
        'title': 225,
        'context': 189,
        'error': 215
    }
    color_code = colors.get(color, color) if isinstance(color, str) else color
    if not (isinstance(color_code, int) and 0 <= color_code <= 255):
        color_code = 0
    print(f"\033[38;5;{color_code}m{text}\033[0m")

if __name__ == "__main__":
    print_color_table()