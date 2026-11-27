"""Cuts hand-picked frames, note papers and stickers out of the sheets in assets/
and writes assets/manifest.js. Boxes are [x0, y0, x1, y1] in sheet pixels.
Run from the project root:  python tools/extract.py
"""
import json, os, shutil, subprocess
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

A = 'assets'
HK, DENIM, ORN, CAM = 'frames/download (1).jpg', 'frames/download.jpg', 'frames/download (55.jpg', 'frames/download (54).jpg'
PAPER, DPAPER = 'frames/download (2).jpg', 'frames/download (3).jpg'
STITCH, OGGY, MINION = 'stickers/download (57).jpg', 'stickers/Oggy and the Cockroaches Cute Stickers Pack ✨.jpg', 'stickers/download (55).jpg'
PUCCA, SNOOPY, HIRONO = 'stickers/download (56).jpg', 'stickers/a02d71e02e65cff304dc099f5a04b1f2.jpg', 'stickers/download (54).jpg'
WINDOW, TV, NEWS, STRIP = 'frames/download (4).jpg', 'frames/download (6).jpg', 'frames/download (7).jpg', 'frames/idea.jpg'
FULL = [0, 0, 10000, 10000]  # whole sheet (single-item images)
RECORD, TONEARM = 'vinyl/Download premium png of PNG Electronics gramophone technology turntable by Teddy about vinyl record, black cd, vinyl, music disc, and vinyl cd 12066625.jpg', ''
# sheets cut automatically into the "add a sticker" library (items are well separated on white)
LIBRARY = ['stickers/Vintage Music Stickers for Creative Planners & Notebooks – Download Free Now!.jpg',
           'stickers/download.jpg', 'stickers/download (1).jpg', 'stickers/download (2).jpg',
           'frames/fcf3f0bc5b8e1863c1fc94ae7fbe04a0.jpg']

# (sheet, box, rotate_degrees_to_make_upright)
PAGES = [
    {'name': 'stitch', 'hue': 215, 'sat': 60,
     'frames': [(DENIM, [20, 448, 344, 872]), (WINDOW, FULL), (DENIM, [352, 898, 678, 1292])],
     'note': (DPAPER, [10, 395, 325, 930]),
     'stickers': [(STITCH, [520, 0, 736, 214], 0), (STITCH, [548, 585, 736, 798], 0), (STITCH, [228, 386, 355, 610], 0)]},
    {'name': 'oggy', 'hue': 205, 'sat': 55,
     'frames': [(ORN, [372, 1002, 702, 1252]), (CAM, [362, 280, 728, 515]), (ORN, [45, 78, 338, 338])],
     'note': (DPAPER, [310, 385, 735, 770]),
     'stickers': [(OGGY, [560, 25, 722, 282], 0), (OGGY, [5, 795, 195, 1012], 0), (OGGY, [5, 1040, 192, 1265], 0)]},
    {'name': 'minions', 'hue': 48, 'sat': 75,
     'frames': [(ORN, [448, 402, 702, 642]), (TV, FULL), (DENIM, [35, 28, 337, 430])],
     'note': (PAPER, [35, 0, 382, 345]), 'text': [0.24, 0.38, 0.52, 0.48],
     'stickers': [(MINION, [205, 20, 475, 268], 0), (MINION, [478, 28, 665, 292], 0), (MINION, [232, 898, 478, 1185], 0)]},
    {'name': 'pucca', 'hue': 355, 'sat': 60,
     'frames': [(CAM, [368, 1000, 728, 1250]), (CAM, [358, 515, 728, 742]), (ORN, [38, 1012, 338, 1248])],
     'note': (PAPER, [30, 995, 365, 1300]), 'text': [0.42, 0.4, 0.48, 0.42],
     'stickers': [(PUCCA, [246, 24, 474, 234], 0), (PUCCA, [476, 238, 736, 480], 0), (PUCCA, [492, 722, 712, 936], 0)]},
    {'name': 'snoopy', 'hue': 345, 'sat': 45,
     'frames': [(ORN, [362, 82, 688, 333]), (CAM, [8, 8, 358, 262]), (ORN, [378, 698, 702, 982])],
     'note': (PAPER, [375, 52, 712, 382]),
     'stickers': [(SNOOPY, [245, 220, 495, 390], 90), (SNOOPY, [4, 315, 250, 520], 90), (SNOOPY, [498, 230, 730, 400], 90)]},
    {'name': 'hirono', 'hue': 95, 'sat': 30,
     'frames': [(ORN, [33, 368, 412, 668]), (NEWS, FULL), (CAM, [362, 0, 728, 272])],
     'note': (PAPER, [10, 688, 358, 1000]),
     'stickers': [(HIRONO, [190, 0, 324, 230], 0), (HIRONO, [190, 455, 362, 695], 0), (HIRONO, [345, 445, 525, 692], 0)]},
]

SOFT_SHADOW = {NEWS}
_cache = {}


def sheet(path):
    if path not in _cache:
        _cache[path] = np.asarray(Image.open(os.path.join(A, path)).convert('RGB')).astype(int)
    return _cache[path]


def cut(path, box, gray_halo=False):
    x0, y0, x1, y1 = box
    a = sheet(path)[y0:y1, x0:x1]
    mx, mn = a.max(2), a.min(2)
    bg = (mn >= 222) & (mx - mn <= 9)
    if path in SOFT_SHADOW:  # grey drop shadow around the item counts as background
        bg |= (mn >= 175) & (mx - mn <= 8)
    if gray_halo:  # Hirono sheet: stickers have a thick grey outline, treat it as background
        bg |= (mx - mn <= 14) & (mn >= 95) & (mx <= 175)
    lab, _ = ndi.label(bg)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    outer = np.isin(lab, list(edge))
    fg = ~outer
    # keep only the main piece inside the box (drops bits of neighbours)
    cl, n = ndi.label(ndi.binary_dilation(fg, iterations=3))
    if n > 1:
        sizes = ndi.sum(fg, cl, range(1, n + 1))
        fg &= cl == (1 + int(np.argmax(sizes)))
    fg = ndi.binary_opening(fg, iterations=1)
    return a, fg, lab, edge, bg


def crop(a, alpha):
    ys, xs = np.where(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return np.dstack([a[y0:y1, x0:x1], alpha[y0:y1, x0:x1].astype(np.uint8) * 255]).astype(np.uint8), (x0, y0, x1, y1)


def frame(path, box):
    a, fg, lab, edge, bg = cut(path, box)
    # photo window = biggest enclosed light region
    best = None
    for l in np.unique(lab[fg & bg]):
        if l == 0 or l in edge:
            continue
        reg = lab == l
        if best is None or reg.sum() > best.sum():
            best = reg
    alpha = fg & ~best
    rgba, (x0, y0, x1, y1) = crop(a, fg)
    rgba[..., 3] = (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)
    wy, wx = np.where(best[y0:y1, x0:x1])
    w, h = x1 - x0, y1 - y0
    bw, bh = wx.max() + 1 - wx.min(), wy.max() + 1 - wy.min()
    win = [round(wx.min() / w, 4), round(wy.min() / h, 4), round(bw / w, 4), round(bh / h, 4), round(best.sum() / (bw * bh), 3)]
    return rgba, win


def piece(path, box, gray=False, thin=0):
    a, fg, *_ = cut(path, box, gray)
    if thin:  # drop thin bits such as stray sticks/lines
        fg = ndi.binary_opening(fg, iterations=thin)
        cl, n = ndi.label(fg)
        if n > 1:
            fg = cl == (1 + int(np.argmax(ndi.sum(fg, cl, range(1, n + 1)))))
    fg = ndi.binary_fill_holes(fg) if not gray else fg
    return crop(a, fg)[0]


# ---- character stickers ----
# Each sheet has its own look, so the "background" colour rules differ:
#   (light level, max grey saturation, gap-closing iterations, extra grey band treated as background)
STK = {STITCH: (185, 20, 2, None), OGGY: (236, 14, 1, None), MINION: (222, 10, 1, None),
       PUCCA: (185, 20, 2, None), SNOOPY: (185, 20, 2, None), HIRONO: (225, 14, 2, (95, 190))}
# Snoopy: white die-cut border on a grey sheet. The border runs a constant width outside the drawing,
# so it is measured and pulled in, which keeps Snoopy's white body that touches the border.
DIECUT = {SNOOPY: lambda a, mx, mn: (mn >= 214) & (mx <= 240) & (mx - mn <= 6)}
# Oggy: thin white border + grey line on a cream sheet: peel pale pixels off the outside (px).
PEEL = {OGGY: 9}
# Minions have no white parts except eye whites, so enclosed white gaps can be opened up.
GAPS = {MINION}
def sticker(path, box, pad=12):
    L, S, C, grey = STK[path]
    full = sheet(path); H, W, _ = full.shape
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad)
    a = full[y0:y1, x0:x1]; mx, mn = a.max(2), a.min(2)
    light = (mn >= L) & (mx - mn <= S)
    if grey:
        light |= (mx - mn <= S) & (mn >= grey[0]) & (mx <= grey[1])
    ink = ndi.binary_opening(~light, iterations=1)
    ink = ndi.binary_closing(np.pad(ink, C + 1), iterations=C)[C + 1:-C - 1, C + 1:-C - 1]
    fg = ndi.binary_fill_holes(ink)
    if path in GAPS:
        # white sheet showing through enclosed gaps (between head and headphones, arm and body).
        # Eye whites are also enclosed white, but they wrap round the iris, so keep holes that hold an island.
        hl, hn = ndi.label(fg & light)
        for i in range(1, hn + 1):
            m = hl == i
            if m.sum() >= 40 and ndi.binary_fill_holes(m).sum() - m.sum() < 0.05 * m.sum():
                fg &= ~m
    if path in DIECUT:
        lab0, _ = ndi.label(DIECUT[path](a, mx, mn))
        e = set(np.unique(np.concatenate([lab0[0], lab0[-1], lab0[:, 0], lab0[:, -1]]))) - {0}
        stk = ndi.binary_opening(ndi.binary_fill_holes(~np.isin(lab0, list(e))), iterations=2)
        to_ink = ndi.distance_transform_edt(~((mn < 200) | (mx - mn > 30)))
        inside = ndi.distance_transform_edt(stk)
        bw = float(np.median(to_ink[stk & (inside <= 1.5)]))
        fg = stk & (inside > max(0, bw - 1.5))
    if path in PEEL:
        pale = (mn >= 200) & (mx - mn <= 28)
        for _ in range(PEEL[path]):
            fg &= ~(fg & ~ndi.binary_erosion(fg, border_value=0) & pale)
        fg = ndi.binary_opening(fg, iterations=1)
    lab, n = ndi.label(fg)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    # main piece = the blob covering most of the middle of the hand-picked box
    bx, by, bw_, bh = box[0] - x0, box[1] - y0, box[2] - box[0], box[3] - box[1]
    core = np.zeros_like(fg); core[by + bh // 4:by + 3 * bh // 4, bx + bw_ // 4:bx + 3 * bw_ // 4] = True
    main = 1 + int(np.argmax(ndi.sum(core & fg, lab, range(1, n + 1))))
    keep = lab == main
    # small extras (music notes, petals) close to the piece that don't run off the crop edge
    near = ndi.binary_dilation(keep, iterations=30)
    for i in range(1, n + 1):
        if i == main or not 0.004 * sizes[main - 1] < sizes[i - 1] < 0.25 * sizes[main - 1]:
            continue
        m = lab == i; ys, xs = np.where(m)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == fg.shape[0] - 1 or xs.max() == fg.shape[1] - 1:
            continue
        if (m & near).any():
            keep |= m
    return crop(a, keep)[0]
def text_area(rgba):
    """Biggest plain light patch of a note paper = where the writing goes."""
    a = rgba[..., :3].astype(int)
    mx, mn = a.max(2), a.min(2)
    light = (rgba[..., 3] > 0) & (mn >= 170) & (mx - mn <= 45)
    light = ndi.binary_closing(light, iterations=4)   # bridge the ruled lines
    light = ndi.binary_opening(light, iterations=6)
    cl, n = ndi.label(light)
    reg = cl == (1 + int(np.argmax(ndi.sum(light, cl, range(1, n + 1)))))
    # largest rectangle-ish core: shrink bbox until mostly inside the patch
    ys, xs = np.where(reg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    for _ in range(40):
        if reg[y0:y1, x0:x1].mean() > 0.97:
            break
        dy, dx = (y1 - y0) * 0.02, (x1 - x0) * 0.02
        y0, y1, x0, x1 = int(y0 + dy), int(y1 - dy), int(x0 + dx), int(x1 - dx)
    H, W = reg.shape
    pad_x, pad_y = (x1 - x0) * 0.06, (y1 - y0) * 0.06
    return [round((x0 + pad_x) / W, 4), round((y0 + pad_y) / H, 4),
            round((x1 - x0 - 2 * pad_x) / W, 4), round((y1 - y0 - 2 * pad_y) / H, 4)]


# Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN, ncnn build) upscales the small Pinterest-size
# sheets 4x. Set RESR to its folder; without it, save() falls back to plain Lanczos resizing.
RESR = os.environ.get('RESR', 'C:/Temp/resr')
PHOTO, CARTOON = 'realesrgan-x4plus', 'realesrgan-x4plus-anime'


def upscale(rgb, model):
    exe = os.path.join(RESR, 'realesrgan-ncnn-vulkan.exe')
    if not os.path.exists(exe):
        return None
    src, dst = 'C:/Temp/_up_in.png', 'C:/Temp/_up_out.png'
    rgb.save(src)
    subprocess.run([exe, '-i', src, '-o', dst, '-n', model], cwd=RESR, capture_output=True)
    return Image.open(dst).convert('RGB') if os.path.exists(dst) else None


def save(rgba, out, maxw, model=PHOTO):
    """4x AI upscale (capped at maxw wide), smooth anti-aliased edges, saved as webp."""
    # paint transparent pixels with the nearest kept colour, so the soft edge doesn't blend in the
    # white sheet behind the sticker (that showed up as a white fringe)
    rgba = rgba.copy()
    clear = rgba[..., 3] < 128
    if clear.any() and not clear.all():
        iy, ix = ndi.distance_transform_edt(clear, return_distances=False, return_indices=True)
        rgba[..., :3] = rgba[iy, ix, :3]
    img = Image.fromarray(rgba, 'RGBA')
    k = min(4.0, maxw / img.width)
    size = (round(img.width * k), round(img.height * k))
    big = upscale(img.convert('RGB'), model)
    if big is None:
        rgb = img.convert('RGB').resize(size, Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.6, percent=70, threshold=2))
    else:
        rgb = big.resize(size, Image.LANCZOS)
    alpha = img.getchannel('A').resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.4 * k))
    alpha = alpha.point(lambda v: 0 if v < 60 else 255 if v > 200 else int((v - 60) * 255 / 140))
    out_img = rgb.convert('RGBA'); out_img.putalpha(alpha)
    out_img.save(out, 'WEBP', quality=92, method=6)
    return out_img.size

shutil.rmtree(f'{A}/cut', ignore_errors=True)
os.makedirs(f'{A}/cut')
manifest = []
for n, pg in enumerate(PAGES, start=1):
    out = {'name': pg['name'], 'hue': pg['hue'], 'sat': pg['sat'], 'frames': [], 'stickers': []}
    for k, (p, b) in enumerate(pg['frames']):
        rgba, win = frame(p, b)
        w, h = save(rgba, f'{A}/cut/p{n}_frame{k}.webp', 1400)
        out['frames'].append({'src': f'assets/cut/p{n}_frame{k}.webp', 'ar': round(w / h, 4), 'win': win})
    p, b = pg['note']
    rgba = piece(p, b, thin=3)
    w, h = save(rgba, f'{A}/cut/p{n}_note.webp', 1400)
    out['note'] = {'src': f'assets/cut/p{n}_note.webp', 'ar': round(w / h, 4), 'text': pg.get('text') or text_area(rgba)}
    for k, (p, b, rot) in enumerate(pg['stickers']):
        rgba = sticker(p, b)
        img = Image.fromarray(rgba, 'RGBA')
        if rot:
            img = img.rotate(rot, expand=True, resample=Image.BICUBIC)
        w, h = save(np.asarray(img), f'{A}/cut/p{n}_sticker{k}.webp', 900, CARTOON)
        out['stickers'].append({'src': f'assets/cut/p{n}_sticker{k}.webp', 'ar': round(w / h, 4)})
    manifest.append(out)
    print(n, pg['name'], 'ok')


# ---- photostrip for the introduction page: every film window becomes a photo slot ----
def strip(path):
    a, fg, lab, edge, bg = cut(path, FULL)
    H, W = fg.shape
    # the camera body is white on a white sheet, so the colour cut eats into it: add it back as a rounded box
    x0, y0, x1, y1, r = 161, 107, 578, 350, 20
    yy, xx = np.mgrid[:H, :W]
    dx = np.maximum(np.maximum(x0 + r - xx, xx - (x1 - r)), 0)
    dy = np.maximum(np.maximum(y0 + r - yy, yy - (y1 - r)), 0)
    fg |= (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & (np.hypot(dx, dy) <= r)
    wins = []
    for l in np.unique(lab[fg & bg]):
        if l == 0 or l in edge:
            continue
        reg = lab == l
        ys, xs = np.where(reg)
        bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
        # film windows only: the camera body above the strip is also an enclosed light area
        if reg.sum() > 0.01 * H * W and reg.sum() / (bw * bh) > 0.9 and ys.mean() > 0.33 * H:
            wins.append((ys.min(), reg))
    wins.sort(key=lambda t: t[0])
    alpha = fg.copy()
    for _, reg in wins:
        alpha &= ~reg
    rgba, (x0, y0, x1, y1) = crop(a, fg)
    rgba[..., 3] = (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)
    w, h = x1 - x0, y1 - y0
    out = []
    for _, reg in wins:
        ys, xs = np.where(reg[y0:y1, x0:x1])
        out.append([round(xs.min() / w, 4), round(ys.min() / h, 4), round((xs.max() + 1 - xs.min()) / w, 4), round((ys.max() + 1 - ys.min()) / h, 4)])
    return rgba, out


extra = {}
rgba, wins = strip(STRIP)
w, h = save(rgba, f'{A}/cut/intro_strip.webp', 1400)
extra['strip'] = {'src': f'{A}/cut/intro_strip.webp', 'ar': round(w / h, 4), 'wins': wins}
print('strip windows', len(wins))

# ---- sticker library ----
lib = []
for si, path in enumerate(LIBRARY):
    a = sheet(path)
    mx, mn = a.max(2), a.min(2)
    bg = (mn >= 228) & (mx - mn <= 14)
    lab, _ = ndi.label(bg)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    fg = ndi.binary_opening(~np.isin(lab, list(edge)), iterations=1)
    cl, n = ndi.label(ndi.binary_dilation(fg, iterations=2))
    H, W = fg.shape
    for k in range(1, n + 1):
        m = (cl == k) & fg
        ys, xs = np.where(m)
        if len(ys) < 0.004 * H * W:
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
            continue  # cut off by the sheet edge
        m = ndi.binary_fill_holes(m)
        rgba = crop(a, m)[0]
        out = f'{A}/cut/lib{si}_{len(lib)}.webp'
        w, h = save(rgba, out, 700, CARTOON)
        lib.append({'src': out, 'ar': round(w / h, 4)})
# pieces where two stickers touch on the sheet and came out merged
MERGED = {14, 21, 32, 44}
for k in sorted(MERGED, reverse=True):
    os.remove(lib[k]['src'])
    del lib[k]
print('library stickers', len(lib))

with open(f'{A}/manifest.js', 'w', encoding='utf-8') as f:
    f.write('window.PACKS=' + json.dumps(manifest) + ';\n')
    f.write('window.EXTRA=' + json.dumps(extra) + ';\n')
    f.write('window.LIBRARY=' + json.dumps(lib) + ';\n')

# contact sheet for checking the cuts
tiles = []
for n in range(1, 7):
    row = [Image.open(f'{A}/cut/{f}').convert('RGBA') for f in sorted(os.listdir(f'{A}/cut')) if f.startswith(f'p{n}_')]
    tiles.append(row)
H = 170
sheet_img = Image.new('RGB', (H * 7 + 80, H * 6), (120, 120, 140))
for r, row in enumerate(tiles):
    x = 0
    for t in row:
        t = t.copy(); t.thumbnail((H - 6, H - 6))
        sheet_img.paste(t, (x + 3, r * H + 3), t); x += H
os.makedirs('C:/Temp', exist_ok=True)
sheet_img.save('C:/Temp/contact.png')








