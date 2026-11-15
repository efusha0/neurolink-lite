# Recording the demo GIF

The README has a place for `docs/demo.gif` (commented out until the file exists). This is
how to record it on Windows. Aim for 15-25 seconds, about 900 px wide, under 10 MB.

## What to show

One uncut stretch of the running demo that contains:

1. a cue (`CUE: imagine the LEFT / RIGHT hand`) with the cursor reaching the target, and
2. the rest period after it, where the cursor often keeps moving although nothing is
   being imagined ("False command").

Do not cut out timeouts or false commands. The project reports what the decoder really
does, and the GIF should too. The page already says on screen that the subject is a best
case and what the average is; keep that footer in the frame.

## 1. Start the demo

```bat
cd C:\path\to\neurolink-lite
uv run neurolink demo
```

The browser opens `http://127.0.0.1:8000`. Make the window about 1200 px wide and zoom
out (Ctrl + minus) until the whole page, from the title to the footer, is visible without
scrolling. Wait until the game card says `Replay pass 2`: from then on the counters cover
a full pass. Keep the tab in the foreground while recording.

## 2. Record with ScreenToGif

[ScreenToGif](https://www.screentogif.com/) is free and open source.

1. Open ScreenToGif and choose **Recorder**.
2. Drag the frame over the page. Set the frame rate to **15 fps**.
3. Start a few seconds before a cue (the rest periods last about 4 s) and press **Record**
   (F7).
4. Record one trial and the following rest period, then **Stop** (F8).
5. In the editor: delete frames before and after the part you want (select, Delete).
6. **Image > Resize** to about 900 px wide.
7. **File > Save as**, type **Gif**, and save it as `docs\demo.gif` in the repository.

If the file is larger than 10 MB, lower the frame rate to 10 fps (**Edit > Reduce frame
count**) or shorten the clip.

## Alternative: record a video and convert it with ffmpeg

Record the browser window with the Xbox Game Bar (Win + Alt + R starts and stops; the
video is saved under `Videos\Captures`), then convert it:

```bat
ffmpeg -ss 5 -t 20 -i capture.mp4 -vf "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" docs\demo.gif
```

`-ss` is the start in seconds and `-t` the length. The palette filters keep the dark
background free of banding.

## 3. Put it in the README

In `README.md`, remove the comment markers (`<!--` with its three explanatory lines, and
`-->`) around the demo line near the top, so that it reads:

```markdown
![NeuroLink-lite demo: the cursor follows the decoder while subject S007's recording is replayed](docs/demo.gif)
```

Delete the screenshot line (`![The UI during a rest period ...](docs/ui.png)`) directly
below it: the GIF takes its place at the top, and the same screenshot is still shown in
the "Browser UI" section.

Then commit both files:

```bat
git add docs\demo.gif README.md
git commit -m "docs: add the demo GIF"
git push
```
