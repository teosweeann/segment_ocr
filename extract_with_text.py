import os, sys, io, re, glob, cv2
import numpy as np
from pdf2image import convert_from_path

from page import Page
import ocr

FIG_DELTA = None
TABLE_DELTA = None

def figure_delta(image):
    global FIG_DELTA
    if not FIG_DELTA is None: return FIG_DELTA
    FIG_DELTA = ocr.figure_cap_position(image) 
    print("FIGURE DELTA:", FIG_DELTA)
    return FIG_DELTA

def table_delta(image):
    global TABLE_DELTA
    if not TABLE_DELTA is None: return TABLE_DELTA
    TABLE_DELTA = ocr.table_cap_position(image) 
    print("TABLE DELTA:", TABLE_DELTA)
    return TABLE_DELTA

def make_safe_name(r): # something that would not mess up the system would suffice. 
    s = r.replace(' ', '_')
    for _ in '.,;\'\"/$%\\^{}[]()?<>#@&*':
        s = s.replace(_, '')
    return  s.replace(":",'_').replace("\\",'').replace("?",'')

def extract(path, pages=None):
    bbox_files = glob.glob(os.path.join(path, 'bb_*.txt'))
    bbox_files.sort()
    fname = open(os.path.join(path, 'info.txt')).read().strip()
    print('PDF:', fname)

    images = convert_from_path(fname)
    images_hr = convert_from_path(fname, dpi=600)

    if pages is None:
        pn_list = range(len(images))
    else:
        pn_list = pages #list(map(lambda x: int(x), sys.argv[2:]))

    for pn in pn_list:  # For each page
        text_chunks = []
        p = open(bbox_files[pn]).read().strip().splitlines()
        # Get bounding boxes
        bnd_boxes = []
        with open(bbox_files[pn]) as _:
            for line in _:
                if line.strip()=='': continue
                bnd_boxes.append(tuple(map(lambda x: int(x), line.strip().strip('[').strip(']').split(','))))

        page = Page(
                np.array(images[pn]), 
                bnd_boxes, 
                page_hr=np.array(images_hr[pn])
        )
        page.clear_cache()
        
        fig_indexes = []
        cap_indexes = []
        fig_paths = {}
        table_indexes = []

        # Scan for captions of tables and figures
        for i in range(len(page)):
            # Get the figures first
            if (page.bnd_boxes[i][1]-page.bnd_boxes[i][0]) > page.height/2: 
                continue  # Skip very large chunks. 
            if page.text_density(i)<0.8: 
                #print(f"BOX {i} text density:", page.text_density(i))
                continue
            # Then this gotta be text
            cap_text = page.transcribe(i).replace("*",'').strip().lower()
            # skip if not a table or fig 
            print('#1', cap_text)
            if cap_text.startswith('fig.') or cap_text.startswith('figure'): 
                delta = figure_delta(page.page)
                mode = 'fig'
            elif cap_text.startswith('table'): 
                delta = table_delta(page.page)
                mode = 'table'
            else:
                continue

            j = i + delta # Walk the list to construct the figure list or tables
            print('#2', j, delta)
            obox = []
            while j>=0 and j<len(page):
                tx, bx, lx, rx = page.bnd_boxes[j]
                if delta<0 and bx > page.bnd_boxes[i][1]: break
                if delta>0 and tx < page.bnd_boxes[i][0]: break
                if j in fig_indexes or j in cap_indexes or j in table_indexes: break # Already in the list
                print('#3 fig', j, mode, page.is_table(j), page.is_small_block(j), delta*(j-i))
                if mode=='table' and not page.is_table(j) and not page.is_small_block(j) and delta*(j-i)>1: break
                if page.is_figure(j) or page.is_table(j) or page.is_small_block(j) or \
                        (j+delta>=0 and j+delta<len(page) and (page.is_figure(j+delta) or page.is_table(j+delta)) \
                        and not j+delta in fig_indexes):
                    fig_indexes.append(j)
                    obox.append(j)
                    j += delta
                else: 
                    break
            if len(obox)>0:
                if  mode=='fig':
                    cap_indexes.append(i) # Caption list, which also includes tables
                elif mode=='table':
                    table_indexes.append(i)
                box = np.array(page.union_boxes(obox, ext=10))*3  # upscaling to 600 dpi
                cropped_image = images_hr[pn].crop((box[2], box[0], box[3], box[1])) 
                img_name = make_safe_name(cap_text).replace("\\",'')[:32]+'.jpg'
                img_name = f'fig_{pn:02d}_{i:02d}__'+img_name
                fig_paths[i] = (img_name, (box[3]-box[2])//3)
                cropped_image.save(os.path.join(path, img_name))

        # Scan for the text, given that we know where the figures, tables and captions are. 
        concat_flag = False
        obox = []
        for i in range(len(page)):
            if not i in fig_indexes: 
                if i in cap_indexes:
                    width = min(1000, int(fig_paths[i][1]/page.width*1400))
                    text_chunks.append(f'<img src="./{fig_paths[i][0]}" alt="fig_{pn:02d}_{i:02d}" width="{width}">')
                    text_chunks.append('###### ' + page.transcribe(i))
                elif i in table_indexes:
                    width = min(1000, int(fig_paths[i][1]/page.width*1400))
                    text_chunks.append(f'<img src="./{fig_paths[i][0]}" alt="table_{pn:02d}_{i:02d}" width="{width}">')
                    text_chunks.append('###### ' + page.transcribe(i))
                else: text_chunks.append(page.transcribe(i))

        with open(os.path.join(path, f'text_{pn:02d}.md'), 'w') as f:
            _ = '\n\n'.join(text_chunks)
            _  = re.sub(r"\n\$\$(\S+[^\$]*\S+)\$\$\n", "\n$$\n\\1\n$$\n", _)
            f.write(_)

if __name__=='__main__':
    if len(sys.argv)==2:
        extract(sys.argv[1])
    else:
        extract(sys.argv[1], pages = list(map(lambda x: int(x), sys.argv[2:])))
