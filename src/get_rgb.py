# RETURN RGB VALUE FOR THE GIVEN NAME (TEXT)
#   Reference: https://www.rapidtables.com/web/color/RGB_Color.html


def get_rgb(color_name):
    rgb_value = []
    if color_name == "White" or color_name == "white" or color_name == "WHITE":
        rgb_value = (255, 255, 255)
    elif color_name == "Black" or color_name == "black" or color_name == "BLACK":
        rgb_value = (0, 0, 0)
    elif color_name == "Red" or color_name == "red" or color_name == "RED":
        rgb_value = (255, 0, 0)
    elif color_name == "Blue" or color_name == "blue" or color_name == "BLUE":
        rgb_value = (0, 0, 255)
    elif color_name == "Yellow" or color_name == "yellow" or color_name == "YELLOW":
        rgb_value = (255, 255, 0)
    elif color_name == "Cyan" or color_name == "cyan" or color_name == "CYAN":
        rgb_value = (0, 255, 255)
    elif color_name == "Magenta" or color_name == "magenta" or color_name == "MAGENTA":
        rgb_value = (255, 0, 255)
    elif color_name == "Silver" or color_name == "silver" or color_name == "SILVER":
        rgb_value = (192, 192, 192)
    elif color_name == "Gray" or color_name == "gray" or color_name == "GRAY":
        rgb_value = (128, 128, 128)
    elif color_name == "Green" or color_name == "green" or color_name == "GREEN":
        rgb_value = (0, 255, 0)
    elif color_name == "Purple" or color_name == "purple" or color_name == "PURPLE":
        rgb_value = (128, 0, 128)

    # RED ~ ORANGE
    elif color_name == "Maroon" or color_name == "maroon" or color_name == "MAROON":
        rgb_value = (128, 0, 0)
    elif color_name == "Dark Red" or color_name == "dark red" or color_name == "DARK RED":
        rgb_value = (139, 0, 0)
    elif color_name == "Brown" or color_name == "brown" or color_name == "BROWN":
        rgb_value = (165, 42, 42)
    elif color_name == "Firebrick" or color_name == "firebrick" or color_name == "FIREBRICK":
        rgb_value = (178, 34, 34)
    elif color_name == "Crimson" or color_name == "crimson" or color_name == "CRIMSON":
        rgb_value = (220, 20, 60)
    elif color_name == "Tomato" or color_name == "tomato" or color_name == "TOMATO":
        rgb_value = (255, 99, 71)
    elif color_name == "Coral" or color_name == "coral" or color_name == "CORAL":
        rgb_value = (255, 127, 80)
    elif color_name == "Indian Red" or color_name == "indian red" or color_name == "INDIAN RED":
        rgb_value = (255, 127, 80)
    elif color_name == "Light Coral" or color_name == "light coral" or color_name == "LIGHT CORAL":
        rgb_value = (240, 128, 128)
    elif color_name == "Dark Salmon" or color_name == "dark salmon" or color_name == "DARK SALMON":
        rgb_value = (233, 150, 122)
    elif color_name == "Salmon" or color_name == "salmon" or color_name == "SALMON":
        rgb_value = (250, 128, 114)
    elif color_name == "Light Salmon" or color_name == "light salmon" or color_name == "LIGHT SALMON":
        rgb_value = (255, 160, 122)
    elif color_name == "Orange Red" or color_name == "orange red" or color_name == "ORANGE RED":
        rgb_value = (255, 69, 0)
    elif color_name == "Dark Orange" or color_name == "dark orange" or color_name == "DARK ORANGE":
        rgb_value = (255, 140, 0)
    elif color_name == "Orange" or color_name == "orange" or color_name == "ORANGE":
        rgb_value = (255, 165, 0)

    # YELLOW ~ GREEN
    elif color_name == "Gold" or color_name == "gold" or color_name == "GOLD":
        rgb_value = (255, 215, 0)
    elif color_name == "Dark Golden Rod" or color_name == "dark golden rod" or color_name == "DARK GOLDEN ROD":
        rgb_value = (184, 134, 11)
    elif color_name == "Golden Rod" or color_name == "golden rod" or color_name == "GOLDEN ROD":
        rgb_value = (218, 165, 32)
    elif color_name == "Pale Golden Rod" or color_name == "pale golden rod" or color_name == "PALE GOLDEN ROD":
        rgb_value = (238, 232, 170)
    elif color_name == "Dark Khaki" or color_name == "dark khaki" or color_name == "DARK KHAKI":
        rgb_value = (189, 183, 107)
    elif color_name == "Khaki" or color_name == "khaki" or color_name == "KHAKI":
        rgb_value = (240, 230, 140)
    elif color_name == "Olive" or color_name == "olive" or color_name == "OLIVE":
        rgb_value = (128, 128, 0)
    elif color_name == "Yellow Green" or color_name == "yellow green" or color_name == "YELLOW GREEN":
        rgb_value = (154, 205, 50)
    elif color_name == "Dark Olive Green" or color_name == "dark olive green" or color_name == "DARK OLIVE GREEN":
        rgb_value = (85, 107, 47)
    elif color_name == "Olive Drab" or color_name == "olive drab" or color_name == "OLIVE DRAB":
        rgb_value = (107, 142, 35)
    elif color_name == "Lawn Green" or color_name == "lawn green" or color_name == "LAWN GREEN":
        rgb_value = (124, 252, 0)
    elif color_name == "Green Yellow" or color_name == "green yellow" or color_name == "GREEN YELLOW":
        rgb_value = (173, 255, 47)
    elif color_name == "Dark Green" or color_name == "dark green" or color_name == "DARK GREEN":
        rgb_value = (0, 100, 0)
    elif color_name == "Forest Green" or color_name == "forest green" or color_name == "FOREST GREEN":
        rgb_value = (34, 139, 34)

    # BLUE
    elif color_name == "Dark Turquoise" or color_name == "dark turquoise" or color_name == "DARK TURQUOISE":
        rgb_value = (0, 206, 209)
    elif color_name == "Turquoise" or color_name == "turquoise" or color_name == "TURQUOISE":
        rgb_value = (64, 224, 208)
    elif color_name == "Corn Flower Blue" or color_name == "corn flower blue" or color_name == "CORN FLOWER BLUE":
        rgb_value = (100, 149, 237)
    elif color_name == "Deep Sky Blue" or color_name == "deep sky blue" or color_name == "DEEP SKY BLUE":
        rgb_value = (0, 191, 255)
    elif color_name == "Dodger Blue" or color_name == "dodger blue" or color_name == "DODGER BLUE":
        rgb_value = (0, 191, 255)
    elif color_name == "Light Blue" or color_name == "light blue" or color_name == "LIGHT BLUE":
        rgb_value = (173, 216, 230)
    elif color_name == "Sky Blue" or color_name == "sky blue" or color_name == "SKY BLUE":
        rgb_value = (135, 206, 235)
    elif color_name == "Light Sky Blue" or color_name == "light sky blue" or color_name == "LIGHT SKY BLUE":
        rgb_value = (135, 206, 250)
    elif color_name == "Midnight Blue" or color_name == "midnight blue" or color_name == "MIDNIGHT BLUE":
        rgb_value = (25, 25, 112)
    elif color_name == "Navy" or color_name == "navy" or color_name == "NAVY":
        rgb_value = (0, 0, 128)
    elif color_name == "Dark Blue" or color_name == "dark blue" or color_name == "DARK BLUE":
        rgb_value = (0, 0, 139)
    elif color_name == "Medium Blue" or color_name == "medium blue" or color_name == "MEDIUM BLUE":
        rgb_value = (0, 0, 205)
    elif color_name == "Royal Blue" or color_name == "royal blue" or color_name == "ROYAL BLUE":
        rgb_value = (65, 105, 225)

    elif color_name == "Deep Pink" or color_name == "deep pink" or color_name == "DEEP PINK":
        rgb_value = (255, 20, 147)
    elif color_name == "Slate Gray" or color_name == "slate gray" or color_name == "SLATE GRAY":
        rgb_value = (112, 128, 144)
    elif color_name == "Dark Slate Blue" or color_name == "dark slate blue" or color_name == "DARK SLATE BLUE":
        rgb_value = (72, 61, 139)
    elif color_name == "Medium Slate Blue" or color_name == "medium slate blue" or color_name == "MEDIUM SLATE BLUE":
        rgb_value = (123, 104, 238)

    # BLACK
    elif color_name == "Dim Gray" or color_name == "dim gray" or color_name == "DIM GRAY":
        rgb_value = (105, 105, 105)
    elif color_name == "Dark Gray" or color_name == "dark gray" or color_name == "DARK GRAY":
        rgb_value = (169, 169, 169)
    elif color_name == "Silver" or color_name == "silver" or color_name == "SILVER":
        rgb_value = (192, 192, 192)
    elif color_name == "Light Gray" or color_name == "light gray" or color_name == "LIGHT GRAY":
        rgb_value = (211, 211, 211)
    elif color_name == "Gainsboro" or color_name == "gainsboro" or color_name == "GAINSBORO":
        rgb_value = (220, 220, 220)
    elif color_name == "White Smoke" or color_name == "white smoke" or color_name == "WHITE SMOKE":
        rgb_value = (245, 245, 245)
    else:
        print("Do nothing.")

    rgb_value = tuple([x / 255 for x in rgb_value])

    return rgb_value
