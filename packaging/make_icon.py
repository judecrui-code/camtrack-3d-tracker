"""Generate packaging/camtrack.ico (PNG-in-ICO, no extra dependencies besides OpenCV)."""
import os
import struct

import cv2
import numpy as np

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camtrack.ico")
imgs = []
for s in (256, 64, 48, 32, 16):
    S = s * 4
    im = np.zeros((S, S, 4), np.uint8)
    c = S // 2
    cv2.circle(im, (c, c), int(S * 0.47), (40, 40, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(im, (c, c), int(S * 0.34), (255, 255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(im, (c, c), int(S * 0.22), (40, 40, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(im, (c, c), int(S * 0.09), (255, 255, 255, 255), -1, cv2.LINE_AA)
    im = cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA)
    imgs.append((s, cv2.imencode(".png", im)[1].tobytes()))
hdr = struct.pack("<HHH", 0, 1, len(imgs))
off = 6 + 16 * len(imgs)
entries, data = b"", b""
for s, png in imgs:
    entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), off + len(data))
    data += png
open(out, "wb").write(hdr + entries + data)
print("wrote", out)
