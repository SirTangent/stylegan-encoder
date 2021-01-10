from selenium import webdriver
from wget import download

def scrape_images_webdriver(url, output_path=""):
    # TODO: Validate for URL

    # Start webdriver
    driver = webdriver.Firefox()
    driver.get(url)

    # Get all images in DOM
    # TODO: Make sure arrayed src items are unpacked
    elements = driver.execute_script("return document.images")
    srcs = [i.get_property("currentSrc") for i in elements]

    # Download image list
    count = 0
    for src in srcs:
        try:
            download(src, output_path)
            count += 1
        except Exception as e:
            print(f"{srcs} failed to download!")

    driver.close()

    print(f"{count} / {len(srcs)} images were successfully downloaded!")
    return count, len(srcs)
