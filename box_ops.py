
def union_boxes(box1, box2, ext=0):
    obox = [0,0,0,0]
    obox[0] = min(box1[0], box2[0])
    obox[1] = max(box1[1], box2[1])
    obox[2] = min(box1[2], box2[2])
    obox[3] = max(box1[3], box2[3])
    return obox

def ext_box(img, box, ext=5):
    t,b,l,r = box
    h = img.shape[0]
    w = img.shape[1]
    return [max(0,t-ext), min(h,b+ext), max(0,l-ext), min(w,r+ext)]

def crop_img(img, box, ext=5):
    t,b,l,r = box
    h = img.shape[0]
    w = img.shape[1]
    return img[max(0,t-ext):min(h,b+ext), max(0,l-ext):min(w,r+ext)]

def check_overlap_box(x1, x2, x3, x4):
    L1 = x2 - x1
    L2 = x4 - x3
    overlap = max(0, min(x2, x4) - max(x1, x3))
    is_overlap = overlap > 0.8 * min(L1, L2)
    is_length = (0.70 * max(L1, L2)) <= min(L1, L2)
    return is_overlap and is_length
