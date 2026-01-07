# page_extractor
Extracts pages of documents from screenshots, photos etc.

usage: page_extractor.py [-h] [--postprocess {ON,OFF}] [--logging {ON,OFF}] [--logfile LOGFILE] image_file

        A robust document scanner script using OpenCV to extract a book page
        from a screenshot (or photo) and correct its perspective.
        It assumes the book page is the largest 4-sided object covering the center of the image.
        

positional arguments:
  image_file            The path to the input screenshot/image file (e.g., photo_of_book.jpg).

options:
  -h, --help            show this help message and exit
  --postprocess {ON,OFF}
                        
                                                Controls the final image enhancement:
                                                ON: (Default) Converts the extracted page to B&W using adaptive thresholding (best for text/OCR).
                                                OFF: Saves the extracted page as a raw color image.
                                                
  --logging {ON,OFF}    Turn logging ON or OFF (Default: OFF).
  --logfile LOGFILE     
                                                (Optional) Specify an external log file path.
                                                If provided, the script appends to this file.
                                                If omitted, a new log file named after the input image is created/overwritten.
                                                
Examples:
  1. Process a single file with logging ON (log file: my_screenshot.log):
     python page_extractor.py my_screenshot.png --logging ON

  2. Batch process files and append all results to a single log file:
     for file in *.png; do python page_extractor.py "$file" --logging ON --logfile batch_run.log; done

  3. Process a single file with no logging (and color output):
     python page_extractor.py another_shot.jpg --logging OFF --postprocess OFF
