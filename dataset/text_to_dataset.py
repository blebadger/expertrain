import requests
from bs4 import BeautifulSoup
import datasets


def itar_loader(first_section, last_section):
    cfr_itar_data = [{"part": i, 
        "url": f"https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-22?chapter=I&subchapter=M&part={i}",
         "content": ""} 
         for i in range(first_section, last_section)
    ]
    
    for rec in cfr_itar_data:
        response = requests.get(rec['url'])
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = soup.findAll(text=True)
        rec['content'] = ''.join(texts[:-1])
    data = str(''.join([i['content'] for i in cfr_itar_data]))
    data = [{"text": text}]
    dataset = datasets.Dataset.from_list(data)
    dataset.save_to_disk('/dbfs/ml/ITAR_dataset')

if __name__ == "__main__":
    first_section, last_section = 120, 131
    itar_loader(first_section, last_section)