import math

def extract_geometry(bbox, img_w, img_h):
    
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    length_px = math.sqrt(bbox_w ** 2 + bbox_h ** 2)
    width_px = min(bbox_w, bbox_h)
    relative_width = width_px / img_w

    return {
        "length_px": length_px,
        "width_px": width_px,
        "relative_width": relative_width
    } 