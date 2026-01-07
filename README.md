# 📚 Page Extractor (OpenCV Document Scanner)

A robust document scanner script built using **OpenCV** to automatically detect, extract, and correct the perspective of a book page from a screenshot or photograph. It is designed to work effectively even when the photo is taken at an angle.

It assumes the book page is the **largest 4-sided object** covering the center of the input image.

---

## 🚀 Usage

The script is executed via the command line with the following general syntax:

```bash
page_extractor.py [-h] [--postprocess {ON,OFF}] [--logging {ON,OFF}] [--logfile LOGFILE] image_file
