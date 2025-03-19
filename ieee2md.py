import cv2
import ocr
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import sys
import os
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from pytesseract import image_to_string
from PIL import ImageGrab
from io import BytesIO
import numpy as np
import extract_with_text

WT = 0
DEBUG = False
CHECK = False

def estimate_text_lines(img, display=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    proj = np.sum(binary // 255, axis=1)
    w = binary.shape[1]
    proj_threshold = 0.05 * w
    text_rows = proj > proj_threshold
    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(text_rows):
        if val and not in_line:
            in_line = True
            start = i
        elif not val and in_line:
            in_line = False
            end = i - 1
            lines.append((start, end))
    if in_line:
        lines.append((start, len(text_rows) - 1))

    num_lines = len(lines)
    # Calculate each line's height and compute the average line height
    line_heights = [end - start + 1 for start, end in lines]
    avg_line_height = np.mean(line_heights) if line_heights else 0

    return num_lines, avg_line_height

def local_entropy(image, radius=3):
    # Convert to grayscale if needed.
    if image.ndim == 3:
        image = rgb2gray(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    ent_img = entropy(image, disk(radius))
    avg_entropy = np.mean(ent_img)
    sum_entropy = np.sum(ent_img)
    return sum_entropy, avg_entropy#, ent_img


def trim_edges(image, tblr=None, tol=0.025):
    #print("INPUT TBLR:", tblr)
    if tblr is None:
        top, bottom = 0, image.shape[0]
        left, right = 0, image.shape[1]
    else:
        top,bottom,left,right = tblr
    if bottom-top < 5 or right-left<5: return None
    max_val = 255  
    def row_is_uniform(row):
        #if np.all(np.std(row, axis=0)<5): return True
        channel_range = np.ptp(row, axis=0)
        return np.all(channel_range <= tol * max_val)
    def col_is_uniform(col):
        #if np.all(np.std(col, axis=0)<5): return True
        channel_range = np.ptp(col, axis=0)
        return np.all(channel_range <= tol * max_val)
    while True:
        if bottom-top<5 or right-left<5: return None
        if row_is_uniform(image[top, left:right]): top += 1
        elif row_is_uniform(image[bottom - 1, left:right]): bottom -= 1
        elif col_is_uniform(image[top:bottom, left]): left += 1
        elif col_is_uniform(image[top:bottom, right - 1]): right -= 1
        else: break
    return [top,bottom,left,right]

def invert_if_dark_bg(image, mode_threshold=128):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if np.issubdtype(gray.dtype, np.floating):
        if gray.max() <= 1:
            gray_scaled = (gray * 255).astype(np.uint8)
        else:
            gray_scaled = gray.astype(np.uint8)
    else:
        gray_scaled = gray

    hist = cv2.calcHist([gray_scaled], [0], None, [256], [0, 256])
    mode_intensity = int(np.argmax(hist))
    
    if mode_intensity < mode_threshold:
        if image.dtype == np.uint8:
            inverted = 255 - image
        elif np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1:
                inverted = 1 - image
            else:
                inverted = image.max() - image
        else:
            max_val = np.iinfo(image.dtype).max
            inverted = max_val - image
        return inverted
    else:
        return image

def segment_row_reverse(image, dw=50, tblr=None, end_flag=False, level=0):
    if DEBUG: print('get reverse row:', tblr, level, dw)
    if tblr is None:
        tblr = trim_edges(image)
    else:
        tblr = trim_edges(image, tblr=tblr)
    if tblr is None: return []
    tmp_img = image[tblr[0]:tblr[1], tblr[2]:tblr[3]]
    tmp_img= invert_if_dark_bg(tmp_img)
    en, de = local_entropy(tmp_img)
    if en<300: return []
    if en<30000: return [tblr]
    if en<100000*de and de>0.9: return [tblr]
    gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_sum = np.sum(binary, axis=1) / 255  
    height = binary.shape[0]
    width  = binary.shape[1]
    threshold = 0.02 
    if height>300 and width>image.shape[1]*0.35:
        ans = segment_column(image, tblr=tblr, dw=30, level=level)
        if len(ans)>1: return ans
        ans = segment_column(image, tblr=tblr, dw=25, level=level)
        if len(ans)>1: return ans
    if height>1000 and width>image.shape[1]*0.35:
        ans = segment_column(image, tblr=tblr, dw=20, level=level)
        if len(ans)>1: return ans
        ans = segment_column(image, tblr=tblr, dw=15, level=level)
        if len(ans)>1: return ans
        ans = segment_column(image, tblr=tblr, dw=10, level=level)
        if len(ans)>1: return ans

    in_text = False
    step = 1
    for i in range(height-dw-1, int((height-dw)*0.9), -step):
        if not in_text and np.max(horizontal_sum[i:i+dw]) >= threshold:
            in_text = True
            i0 = i
        elif in_text and np.max(horizontal_sum[i:i+dw]) < threshold:
            a = tmp_img[:i+1, :]
            b = tmp_img[i+dw-1:, :]
            if CHECK:
                ma = ocr.tesseract(a).strip()
                print(ma)
                if len(ma)<4: continue
                mb = ocr.tesseract(b).strip()
                print(mb)
                if len(mb)<4: continue
            t1 = tblr.copy(); t1[1] = t1[0]+i
            t2 = tblr.copy(); t2[0] = t2[0]+i+dw-1
            text = ocr.tesseract(b).lower()
            if 'license' in text or 'confidential' in text or 'ieee' in text:
                return segment_row_reverse(image, tblr=t1, dw=5, level=level) + segment_row(image, tblr=t2, level=level+1)
            else:
                return segment_row(image, tblr=tblr, level=level+1)
    if DEBUG: 
         print("REVERSE CAPTURE FAILED")
         cv2.imshow('FAILED', tmp_img)
         cv2.waitKey(WT)
         cv2.destroyAllWindows()
    return segment_row(image, tblr=tblr, level=level+1)

def segment_row(image, dw=50, tblr=None, end_flag=False, level=0):
    if DEBUG: print('get row:', tblr, level)
    if type(dw)==list: raise
    if tblr is None:
        tblr = trim_edges(image)
    else:
        tblr = trim_edges(image, tblr=tblr)
    if tblr is None: return []
    tmp_img = image[tblr[0]:tblr[1], tblr[2]:tblr[3]]
    tmp_img = invert_if_dark_bg(tmp_img)
    en, de = local_entropy(tmp_img)
    if en<300: return []
    if en<30000: return [tblr]
    if en<100000 and de>0.9: return [tblr]
    gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_sum = np.sum(binary, axis=1) / 255  
    height = binary.shape[0]
    width = binary.shape[1]
    threshold = 0.02 
    if DEBUG: print("Level:", level)
    if height>1000: # and width>image.shape[1]*0.2:
        ans = segment_column(image, tblr=tblr, dw=30, level=level)
        if len(ans)>1: return ans
        ans = segment_column(image, tblr=tblr, dw=20, level=level)
        if len(ans)>1: return ans
        ans = segment_column(image, tblr=tblr, dw=15, level=level)
        if len(ans)>1: return ans
        ans = segment_column(image, tblr=tblr, dw=10, level=level)
        if len(ans)>1: return ans

    if level<2 or height<dw*6: start_row = 0
    else: start_row = height // 3
    in_text = False
    i0 = None
    step = max(1, dw//10)
    for i in range(start_row, height - dw, step):
        if not in_text and np.max(horizontal_sum[i:i+dw]) >= threshold:
            in_text = True
        elif in_text and i0 is None and np.max(horizontal_sum[i:i+dw]) < threshold:
            i0 = i
        elif in_text and not i0 is None and np.max(horizontal_sum[i:i+dw]) >= threshold:
            a = tmp_img[:i0+1, :]
            b = tmp_img[i+dw-2:, :]
            mb = ocr.tesseract(b).strip()
            if len(mb)<20 and '(' in mb and ')' in mb:
                break
            t1 = tblr.copy(); t1[1] = t1[0]+i0
            t2 = tblr.copy(); t2[0] = t2[0]+i+dw-1
            if DEBUG:
                cv2.imshow('row 1', a)
                cv2.imshow('row 2', b)
                cv2.waitKey(WT)
                cv2.destroyAllWindows()
            if level==0:
                ans2 = segment_row_reverse(image, tblr=t2, dw=5, level=level)
                return segment_row(image, tblr=t1, level=level+1) + ans2
            if height>500:
                ans = segment_column(image, tblr=t2, dw=20, level=level+1)
            else:
                ans = segment_column(image, tblr=t2, dw=dw+10, level=level+1)
            if len(ans)>1:
                return segment_row(image, tblr=t1, level=level+1) + ans
            else:
                return segment_row(image, tblr=t1, level=level+1) + \
                        segment_row(image, tblr=t2, level=level+1)
   
    if start_row>0:
        in_text = False
        i0 = None
        step = max(1, dw//10)
        for i in range(0, (height*2)//3, step):
            if np.max(horizontal_sum[i:i+dw]) >= threshold and not in_text:
                in_text = True
                continue
            elif in_text and np.max(horizontal_sum[i:i+dw]) < threshold and i0 is None:
                i0 = i
            elif in_text and not i0 is None and np.max(horizontal_sum[i:i+dw]) >= threshold:
                a = tmp_img[:i0+1, :]
                b = tmp_img[i+dw-2:, :]
                mb = ocr.tesseract(b).strip()
                if len(mb)<4 and '(' in mb and ')' in mb:
                    break
                if CHECK:
                    ma = ocr.tesseract(a).strip()
                    print(ma)
                    if len(ma)<4: continue
                    print(mb)
                    if len(mb)<4: continue
                t1 = tblr.copy(); t1[1] = t1[0]+i0
                t2 = tblr.copy(); t2[0] = t2[0]+i+dw-1
                if DEBUG:
                    cv2.imshow('row 1', a)
                    cv2.imshow('row 2', b)
                    cv2.waitKey(WT)
                    cv2.destroyAllWindows()
                if level==0:
                    ans2 = segment_row_reverse(image, tblr=t2, dw=3, level=level+1)
                    return segment_row(image, tblr=t1, level=level+1) + ans
                if height>500:
                    ans = segment_column(image, tblr=t2, dw=15, level=level+1)
                else:
                    ans = segment_column(image, tblr=t2, dw=dw+10, level=level+1)
                if len(ans)>1:
                    return segment_row(image, tblr=t1, level=level+1) + ans
                else:
                    return segment_row(image, tblr=t1, level=level+1) + \
                            segment_row(image, tblr=t2, level=level+1)

    ans = segment_column(image, tblr=tblr, dw=dw+15, level=level+1)
    if len(ans)>1: 
        return ans
    elif dw>=15: 
        return segment_row(image, tblr=tblr, dw=dw-5, level=level)
    return [tblr]

def segment_column(image, tblr=None, dw=50, level=0):
    if DEBUG: print('get COL:', tblr, dw)
    if tblr is None:
        tblr = trim_edges(image)
    else:
        tblr = trim_edges(image, tblr=tblr)
    if tblr is None: return []
    tmp_img = image[tblr[0]:tblr[1], tblr[2]:tblr[3]]
    tmp_img= invert_if_dark_bg(tmp_img)
    en, de = local_entropy(tmp_img)
    if en<300: return []
    if en<30000: return [tblr]
    if en<100000 and de>0.9: return [tblr]
    gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_sum = np.sum(binary, axis=0) / 255  
    height = binary.shape[0]
    width = binary.shape[1]
    threshold = 0.02  
    if level>3 and dw>20: dw=20
    in_text0 = False
    j0 = None
    step = 1
    for j in range(0, len(vertical_sum)-dw, step):
        test_blank = (dw>20 and sum(vertical_sum[j:j+dw]) <= threshold*dw) or \
            (dw<=20 and np.max(vertical_sum[j:j+dw]) <= threshold)
        if not in_text0 and not test_blank:
            in_text0 = True
        elif in_text0 and test_blank:
            j0 = j
        elif in_text0 and not j0 is None and not test_blank:
            a = tmp_img[:, :j0]
            b = tmp_img[:, j+dw-1:]
            if CHECK:
                ma = ocr.tesseract(a).strip()
                print(ma)
                if len(ma)<4: continue
                mb = ocr.tesseract(b).strip()
                print(mb)
                if len(mb)<4: continue
                ea,sa = local_entropy(a)
                if ea<6000: 
                    in_text = False; j0=None
                    continue
                eb,sb = local_entropy(b)
                if eb<6000: 
                    in_text = False; j0=None
                    continue
            t1 = tblr.copy(); t1[3] = t1[2]+j0
            t1t = trim_edges(image, tblr=t1)
            t2 = tblr.copy(); t2[2] = t2[2]+j+dw-1
            t2t = trim_edges(image, tblr=t2)
            if t1t is None: 
                print('T1T is NONE')
                in_text = False
                continue #TEST
            if (t1t[1]-t1t[0])/(t1t[3]-t1t[2])>15 \
                    and t1t[2]>0.1*image.shape[1]:
                in_text = False; j0=None
                print("FAILED RATIO TEST")
                print((t1t[1]-t1t[0])/(t1t[3]-t1t[2]), t1t[2], image.shape[1])
                continue  # prevent grabbing the bullets or list numbers
            if t2t is None: 
                print('T2T is NONE')
                break
            kr = ((t2t[1]-t2t[0])*(t2t[3]-t2t[2])) / ((t1t[1]-t1t[0])*(t1t[3]-t1t[2]))
            if (kr>4 or 1/kr>5) and t1t[2]>0.05*image.shape[1]: 
                in_text = False; j0=None
                continue  
            if DEBUG:
                cv2.imshow('col 1', a)
                cv2.imshow('col 2', b)
                cv2.waitKey(WT)
                cv2.destroyAllWindows()
            return segment_row(image, tblr=t1, level=level+1) + segment_row(image, tblr=t2, level=level+1)
    return [tblr]

def draw_bb(image, boxes, show=False):
    image_with_boxes = image.copy()
    colors = [(255, 0, 0),   # Blue
              (0, 255, 0),   # Green
              (0, 0, 255),   # Red
              (128, 128, 0), # Cyan
              (128, 0, 128), # Magenta
              (0, 128, 128)] # Yellow
    print(image.shape)
    s = image.shape
    height = s[0]
    width = s[1]
    for i, (top, bottom, left, right) in enumerate(boxes):
        color = colors[i % len(colors)]
        cv2.rectangle(image_with_boxes, 
            (max(0, left-2), max(0, top-2)), 
            (min(width, right+2), min(height-1, bottom+2)), 
            color, thickness=2)
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 1
        text_x = left
        text_y = top - 5 if top - 5 > 10 else top + 20
        
        # Optional: Draw a black outline behind the text for better contrast.
        cv2.putText(image_with_boxes, text, (text_x, text_y), 
                font, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(image_with_boxes, text, (text_x, text_y), font, 
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if show:
        cv2.imshow("Image with bounding boxes", image_with_boxes)
        cv2.waitKey(WT)
        cv2.destroyAllWindows()
    else:
        return image_with_boxes

if __name__ == '__main__':
    import datetime
    if len(sys.argv)==2:
        images = convert_from_path(sys.argv[1])
        dtimes = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
        fn = os.path.split(sys.argv[1])[-1].replace(' ','_').replace('?','_').replace(":",'_')[:50]
        if not os.path.exists('outputs'): os.mkdir('outputs')
        dir_name = f'outputs/{dtimes}__{fn}'
        if not os.path.exists(dir_name): os.mkdir(dir_name)
        with open(f'{dir_name}/info.txt', 'w') as f:
            f.write(sys.argv[1])
        for i,img in enumerate(images):
            p = np.array(img)
            bbs = segment_row(p)
            anno_img = draw_bb(p, bbs)
            cv2.imwrite(f'{dir_name}/image_{i:03d}.jpeg', anno_img)
            with open(f'{dir_name}/bb_{i:03d}.txt', 'w') as f:
                 f.write('\n'.join(list(map(lambda x: str(x), bbs))))
        extract_with_text.extract(dir_name)
    else:
        import IPython 
        IPython.embed()
