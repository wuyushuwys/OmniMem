# Preparing videos for the gallery

The gallery videos are **not** committed to git — they live as assets on the GitHub
release [`assets-v1`](https://github.com/wuyushuwys/OmniMem/releases/tag/assets-v1).
The page streams them from there (see `assets_base` in [`videos.yaml`](videos.yaml)).

Each clip is shipped in **two formats** and the browser picks the best one it supports
(handled by [`gallery.js`](gallery.js)):

| Format | Codec | Role | Played by |
| --- | --- | --- | --- |
| `name.webm` | VP9 | preferred (smaller) | Chrome, Firefox, Edge, Safari 14+ |
| `name.mp4` | H.264 | fallback | everything, incl. old browsers / iOS |

> You only ever name the **`.mp4`** in `videos.yaml`. `gallery.js` derives the `.webm`
> name automatically by swapping the extension, so both files must share the same base name.

## 1. Compress a video

Requires [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`). From a folder of source
`.mp4` files, this produces a `.webm` + a re-compressed `.mp4` for each:

```bash
for src in *.mp4; do
  f="${src%.mp4}"
  # VP9 / WebM — preferred. CRF 32 ≈ visually clean at ~half the size; lower CRF = higher quality/bigger.
  ffmpeg -y -i "$src" -c:v libvpx-vp9 -crf 32 -b:v 0 \
    -row-mt 1 -tile-columns 2 -cpu-used 2 -deadline good \
    -pix_fmt yuv420p -an "${f}.webm"
  # H.264 / MP4 — universal fallback. +faststart lets it start playing before fully downloaded.
  ffmpeg -y -i "$src" -c:v libx264 -crf 23 -preset medium \
    -pix_fmt yuv420p -movflags +faststart -an "${f}.mp4"
done
```

Notes:
- `-an` drops audio (the clips are silent). Remove it if a video has sound to keep.
- **Quality knob:** lower `-crf` = higher quality + larger file. VP9 `30–34` and H.264 `21–25`
  are good ranges for these 480p clips. Check the result with
  `ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate yourfile.webm`.
- Reference result: the original 18 clips went from **274 MB → 130 MB** (`.webm`) at this setting.

## 2. Register it in the gallery

Add the clip under the relevant section in [`videos.yaml`](videos.yaml), using the **`.mp4`**
name only:

```yaml
- src: "my_new_clip.mp4"     # the .webm is found automatically
  prompt: "Text shown on hover."
```

(For side-by-side rows use `left:`/`right:`; see existing entries in the file.)

## 3. Upload to the release

Requires the [GitHub CLI](https://cli.github.com/) (`gh auth login` once). Upload **both**
files; `--clobber` overwrites if a same-named asset already exists:

```bash
gh release upload assets-v1 -R wuyushuwys/OmniMem --clobber my_new_clip.webm my_new_clip.mp4
```

## 4. Preview locally

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000/  → in DevTools → Network, confirm the .webm is the file fetched
```
