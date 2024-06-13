from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def extract_text_with_selenium(url):
    try:
        # Configure Selenium WebDriver (adjust for your setup)
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        service = Service(executable_path='/usr/local/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Load the webpage
        driver.get(url)

        # Wait for the page to load (adjust wait time if needed)
        driver.implicitly_wait(10)

        # Get the page source after JavaScript execution
        page_source = driver.page_source

        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract text from the parsed HTML
        text = soup.get_text(separator=' ', strip=True)

        return text

    except Exception as e:
        
        return None  # Return None in case of an error

    finally:
        # Close the browser
        driver.quit()
