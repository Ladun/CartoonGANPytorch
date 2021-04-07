from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

argument = {"keywords": "Polygon portrait", 'limit':500, 'print_urls':True, 'chromedriver': 'C:/Dev/tools/chromedriver'}
paths = response.download(argument)



