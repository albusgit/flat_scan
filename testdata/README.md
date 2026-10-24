# FlatScan test image set

A small set of real photos with open licences, used to measure:

- (a) the whiteboard / projector processing mode, and
- (b) automatic detection of fingers holding a page.

All images come from Wikimedia Commons. Attribution for every file (Commons title, source URL, author,
licence, licence URL, original size and a one-line description) is in `MANIFEST.json` (fingers) and
`MANIFEST_extra.json` (whiteboard, projector, control). CC BY-SA files are redistributed here under their
own licences; they are test data only and are not shipped in the app.

## Categories

| Folder | Purpose | Target | Count |
|---|---|---|---|
| `whiteboard/` | whiteboards with marker writing or drawings | >= 12 | 17 |
| `projector/` | projector screens / projected slides in a room | >= 4 | 14 |
| `fingers/` | fingers or hands holding a page, with ground-truth masks | >= 32 | 33 |
| `control/` | documents with no hands but with skin-like colours | >= 32 | 46 |

### Acceptance criteria

All categories contain photos of real scenes. Illustrations, paintings and flat scans are not accepted.

- **whiteboard**: whiteboards with marker writing or drawings, coloured markers preferred. Include
  photos taken at an angle or with uneven light or glare, some with weak or missing frames, and some
  close-ups where the board fills most of the frame.
- **projector**: projector screens or projected slides in a room. At least one has a dark slide
  background with bright content.
- **fingers**: one or more fingers, thumbs or hands hold a paper page, letter, receipt, magazine,
  newspaper or open book, and the fingers overlap the page edge. The page is a large part of the frame,
  as a scanner user would photograph it. The set mixes skin tones, lighting, and single versus several
  fingers. Every candidate was checked by eye. Photos where no finger touches the page, or where the
  page is tiny, were rejected.
- **control**: documents or pages with no fingers or hands anywhere in the frame. They contain content
  that can fool a skin-colour detector, for example:
  - orange, pink, salmon, peach or beige paper;
  - printed faces or skin;
  - pages on skin-coloured or wooden tables;
  - kraft or brown envelopes;
  - tan cardboard.

  Each image was checked by eye to confirm that no hand is visible.

## Files

- Images are named `<category>_<NN>.jpg`. They are JPEG at quality 90, with the long side at most 2000 px.
- `fingers/masks/<name>.png`: ground-truth mask. It is 8-bit, the same size as the photo,
  255 = hand/finger/thumb and 0 = everything else. Hands are masked up to about the wrist, and forearms
  are not included.
- `fingers/masks/polygons.json`: the hand-drawn outline polygons (image px) that each mask was rasterised
  from. `_meta.grabcut_refined` lists masks that were snapped to edges with OpenCV GrabCut seeded from the
  polygon. `_meta.polygon_only` lists masks where the GrabCut result was rejected and the plain polygon
  was kept. Every mask was checked visually as an overlay on its photo, aiming for an outline within
  about 2% of the image width.
- `fingers/masks/pages.json`: rough corners `[TL, TR, BR, BL]` of the page or book spread in each finger
  photo. If a photo holds two items, the quad is the main one (see the notes in `MANIFEST.json`).

## Licence policy

Only files whose Commons `LicenseShortName` is one of these were accepted:

- CC0
- Public domain
- CC BY (any version)
- CC BY-SA (any version)

Fair-use, non-commercial, no-derivatives and unknown licences were skipped. CC BY and CC BY-SA require
attribution, and CC BY-SA derivatives must be shared under the same licence. The author and licence for
each file are in `MANIFEST.json`.

## Other labels

`board_corners.json`: hand-labelled corners of the board or screen in each whiteboard and projector photo,
used by `tools/board/sheet.py` for the corner IoU (see the README section on whiteboard mode).
