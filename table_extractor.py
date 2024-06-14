class TableExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pages = []
        self.tables = []
        self.image_list = []

        # Ensure the file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        # Check if the file is a PDF
        if not self.file_path.lower().endswith('.pdf'):
            raise ValueError("The file must be a PDF.")
        
        # Open the PDF document
        pdf_document = fitz.open(self.file_path)
        self.num_pages = pdf_document.page_count

        print(f'Initialized for {self.num_pages}-page PDF at {self.file_path}')

        # Instantiation of OCR
        self.ocr = TesseractOCR(n_threads=2, lang="eng")
        print(f'Using OCR: {self.ocr.__class__}')
    
        
    def extract_tables(self, selected_pages=None):

        if not selected_pages:
            selected_pages = list(range(self.num_pages))

        # Process the PDF in batches
        for p in selected_pages:

            # FIX FOR MULTIPLE PAGE COLLECTIONS

            pdf = PDF(self.file_path, 
                      pages=[p],
                      detect_rotation=False,
                      pdf_text_extraction=True)
            
            self.pages.append(pdf.images[0])
            gb_image = cv2.GaussianBlur(pdf.images[0], (3,3), 0)
            gb_image = cv2.bitwise_not(gb_image)

            adaptive_thresh = cv2.adaptiveThreshold(gb_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

            horizontal = adaptive_thresh.copy()
            vertical = adaptive_thresh.copy()
            scale = 20

            h_size = int(horizontal.shape[1]/scale)
            h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
            horizontal = cv2.erode(horizontal, h_struct)
            horizontal = cv2.dilate(horizontal, h_struct)

            v_size = int(vertical.shape[1]/scale)
            v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
            vertical = cv2.erode(vertical, v_struct, (-1, -1))
            vertical = cv2.dilate(vertical, v_struct, (-1, -1))

            self.mask = horizontal + vertical
            self.final_img = cv2.bitwise_and(horizontal, vertical)

            contours, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            self.contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in range(len(self.contours)):

                crit_area = cv2.contourArea(self.contours[c])

                if crit_area < 20000 or crit_area > self.final_img.size * 0.9:
                    continue

                eps = cv2.arcLength(self.contours[c], True)/10
                approx = cv2.approxPolyDP(self.contours[c], eps, True)
                x, y, w, h = cv2.boundingRect(approx)
                roi = self.final_img[int(y):int(y + h), int(x):int(x + w)]

                roi_contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                if len(roi_contours) < 4:
                    continue

                original_image = cv2.rectangle(pdf.images[0], (x, y), (x + w, y + h), (0, 0, 255), 2)
                cropped_image = original_image[y:(y + h), x:(x + w)]
                cv2.imwrite('image' + str(c) + '.png', cropped_image)
                self.image_list.append('image' + str(c) + '.png')

            for img in self.image_list:

                image = Image(img, detect_rotation=False)

                extracted_tables = image.extract_tables(ocr=self.ocr,
                                                        implicit_rows=False,
                                                        borderless_tables=False,
                                                        min_confidence=50)
                self.tables.append(extracted_tables)
