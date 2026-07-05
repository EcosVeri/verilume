# OCR

Verilume supports OCR for scanned and image-heavy local evidence.

## Supported Inputs

- Scanned PDF pages
- Image uploads such as PNG, JPG, JPEG, BMP, TIFF, and WebP
- Embedded images in PowerPoint files when needed

## How It Works

Normal PDFs use native text extraction first. If a page has little or no readable text, Verilume renders the page and applies OCR.

Image uploads are OCRed directly. OCR text is stored with local evidence metadata so it can be retrieved, cited, and shown as part of the answer.

## Safety

OCR can be imperfect. Verilume keeps OCR evidence visible and avoids guessing when no readable evidence exists.
