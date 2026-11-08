# Synthetic computer-vision pipeline notes

This fictional engineering note describes a local document-image workflow.

## Image handling

The ingestion worker checks image bytes, applies OpenCV preprocessing, and sends
the prepared image to OCR when the optional image dependencies are installed.
OCR text becomes ordinary attributed document evidence for lexical and dense
search. The demonstration does not call a vision-language model for every
image.

## Retrieval example

A user might ask, “Which files discuss extracting labels from scanned images?”
The wording differs from the filename. Relevant evidence may come from these
notes and the ScoutX release brief because they mention OCR, document
processing, and local indexing.

## Sample calibration values

- Resize long edge: 1600 pixels.
- Grayscale contrast: 1.25.
- Minimum OCR confidence for review: 0.70.
